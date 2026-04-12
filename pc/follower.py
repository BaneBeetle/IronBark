"""
pc/follower.py — Project IronBark Phase 6 (VLA)
The Follow-Me Brain with situation-aware behavior + semantic exploration.

Phase 6A: VLM reads the scene every ~2.5s and sets a behavior mode
  (ACTIVE/GENTLE/CALM/PLAYFUL/SOCIAL) that modifies follow speed,
  arrival distance, bark behavior, and idle pose.
Phase 6B: When owner is lost, VLM guides navigation toward doorways
  and open spaces instead of just sweeping the head.
"""

import sys
import math
import time
import json
import struct
from enum import Enum
from pathlib import Path

import threading

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class State(Enum):
    IDLE = "IDLE"
    FOLLOW = "FOLLOW"
    SEARCH = "SEARCH"
    EXPLORE = "EXPLORE"


class Command:
    """
    Motor command sent Mac → Pi over ZeroMQ.

    head_mode controls who owns head tracking:
      "local"  — Pi computes head angles from owner_bbox (FOLLOW state).
                 head_yaw/head_pitch in this command are ignored by Pi.
      "remote" — Mac dictates head angles (SEARCH/EXPLORE/BARK states).
                 Pi applies head_yaw/head_pitch directly.

    owner_bbox (dict or None):
      When present, Pi updates its local head tracker with this bbox.
      When absent, Pi keeps using the last known bbox (coasting).
    """

    def __init__(self, action="stop", speed=80, head_yaw=0.0,
                 head_pitch=None, step_count=2, bark=False,
                 head_mode="remote", owner_bbox=None,
                 bark_volume=80, idle_pose=None, thinking=False):
        self.action = action
        self.speed = speed
        self.head_yaw = head_yaw
        self.head_pitch = head_pitch if head_pitch is not None else config.HEAD_DEFAULT_PITCH
        self.step_count = step_count
        self.bark = bark
        self.head_mode = head_mode      # "local" or "remote"
        self.owner_bbox = owner_bbox    # dict with x,y,w,h,frame_w,frame_h
        self.bark_volume = bark_volume  # Phase 6: dynamic volume from behavior mode
        self.idle_pose = idle_pose      # Phase 6: "stand"/"sit"/"lie" after bark hold
        self.thinking = thinking        # Phase 6B: triggers thinking animation on Pi

    def to_json(self):
        d = {
            "action": self.action, "speed": self.speed,
            "head_yaw": self.head_yaw, "head_pitch": self.head_pitch,
            "step_count": self.step_count, "bark": self.bark,
            "head_mode": self.head_mode,
            "bark_volume": self.bark_volume,
        }
        if self.owner_bbox is not None:
            d["owner_bbox"] = self.owner_bbox
        if self.idle_pose is not None:
            d["idle_pose"] = self.idle_pose
        if self.thinking:
            d["thinking"] = True
        return json.dumps(d).encode("utf-8")


class TelemetryReceiver:
    """Background thread that subscribes to Pi telemetry (ultrasonic, battery)."""

    def __init__(self, ctx):
        self.data = {"distance_cm": -1, "battery_v": 0.0, "danger": False}
        self.lock = threading.Lock()
        self.running = True

        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.connect(f"tcp://{config.PI_IP}:{config.REMOTE_TELEM_PORT}")
        self.sock.subscribe(b"")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"[Follower] Telemetry SUB connected to {config.PI_IP}:{config.REMOTE_TELEM_PORT}")

    def _worker(self):
        while self.running:
            try:
                raw = self.sock.recv_string()
                msg = json.loads(raw)
                with self.lock:
                    self.data = msg
            except zmq.Again:
                pass
            except Exception:
                pass

    def get_distance(self):
        with self.lock:
            return self.data.get("distance_cm", -1)

    def get_scan_state(self):
        with self.lock:
            return self.data.get("scan_state", "idle")

    def get_scan_result(self):
        with self.lock:
            return self.data.get("scan_result", None)

    def stop(self):
        self.running = False
        self.sock.close()


class Follower:
    def __init__(self):
        self.state = State.IDLE
        self.last_owner_seen = time.time()  # start the IDLE→EXPLORE countdown from boot
        self.search_start_time = 0.0
        self._unconfirmed_count = 0
        self._should_bark = False
        self._investigating = False  # True when approaching unconfirmed person
        self._last_cmd_time = 0.0
        self._cmd_interval = config.CMD_INTERVAL
        self._smoothed_head_yaw = 0.0

        # Arrival bark state
        self._last_bark_time = 0.0
        self._bark_cooldown = 3.0
        self._bark_hold_until = 0.0

        # Calibration logging (throttled to ~1/sec)
        self._last_follow_log = 0.0
        self._last_area_ratio = 0.0

        # Phase 6A: Situation-aware behavior mode
        self._behavior_mode = config.BEHAVIOR_DEFAULT_MODE
        self._behavior_params = config.BEHAVIOR_MODES[self._behavior_mode]
        self._pending_mode = None       # candidate awaiting confirmation
        self._pending_mode_count = 0    # consecutive readings of pending mode
        self._situation_seq_seen = 0

        # Body centering — tracks last follow action to prevent rapid
        # forward↔turn oscillation at threshold boundaries.
        self._last_follow_action = "forward"
        self._last_follow_action_time = 0.0

        # Obstacle avoidance — LiDAR arc-based (Phase 7) or VLM fallback
        self._scan_phase = None             # None / "backing" / "turning" / "clearing"
        self._scan_phase_until = 0.0
        self._scan_phase_cmd = None
        self._scan_result_direction = None  # LEFT / RIGHT
        self._lidar = None                  # set in main block if LiDAR available
        self._latest_nav_frame = None       # latest ribbon cam frame for VLM
        self._latest_nav_frame_time = 0.0   # timestamp for staleness check

        # Phase 6B: Semantic exploration
        self._explore_direction = None
        self._explore_direction_time = 0.0
        self._explore_seq_seen = 0
        self._explore_move_until = 0.0  # time until current explore move completes
        self._last_explore_cmd = None   # cached command during explore move

        # Pipeline reference (set after construction in main block)
        self.pipeline = None

        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.PUB)
        self.cmd_sock.bind(f"tcp://{config.PC_IP}:{config.CMD_PORT}")
        print(f"[Follower] Command PUB bound on {config.PC_IP}:{config.CMD_PORT}")

        self.telemetry = TelemetryReceiver(self.ctx)

    def update(self, perception_result):
        now = time.time()

        # ── Obstacle maneuver override ────────────────────────────
        # If a backup/turn/clearing maneuver is in progress, it takes
        # priority over ALL state logic. Without this, losing the owner
        # mid-backup causes the coast path to send forward at speed=98,
        # overriding the backup and slamming into the obstacle.
        if self._scan_phase is not None:
            cmd = self._handle_maneuver(now)
            if cmd is not None:
                if now - self._last_cmd_time >= self._cmd_interval:
                    self.send_command(cmd)
                    self._last_cmd_time = now
                return cmd

        owner = self._find_owner(perception_result)

        # "Identity bark" — person detected for 10+ frames but no face match
        bark = False
        if self._should_bark:
            bark = True
            self._should_bark = False
            self._unconfirmed_count = 0

        if owner is not None:
            self.last_owner_seen = now
            self._transition(State.FOLLOW)
            cmd = self._follow(owner, perception_result)

        elif self._bark_hold_until > now:
            # Arrival bark hold — sustain greeting while head looks up.
            bp = self._behavior_params
            do_bark = (bp["bark_enabled"] and
                       now - self._last_bark_time >= self._bark_cooldown)
            if do_bark:
                self._last_bark_time = now
            cmd = Command("stop", head_mode="remote",
                          head_yaw=0.0, head_pitch=config.BARK_HEAD_PITCH,
                          bark=do_bark, bark_volume=bp["bark_volume"],
                          idle_pose=bp["idle_pose"])

        elif bark:
            # Identity confirmation bark (unrecognized person nearby)
            cmd = Command("stop", head_mode="remote",
                          head_yaw=self._smoothed_head_yaw, bark=True)

        elif now - self.last_owner_seen < 2.0 and self.state == State.FOLLOW:
            # Coast forward briefly after losing owner — Pi keeps head
            # pointed at last known bbox (head_mode="local", no new bbox)
            cmd = Command("forward", speed=98, head_mode="local", step_count=2)

        elif now - self.last_owner_seen < config.SEARCH_TIMEOUT_S:
            if self.state != State.SEARCH:
                self._transition(State.SEARCH)
                self.search_start_time = now
            cmd = self._search(now)

        elif now - self.last_owner_seen < config.SEARCH_TIMEOUT_S + config.EXPLORE_TIMEOUT_S:
            if self.state != State.EXPLORE:
                self._transition(State.EXPLORE)
            cmd = self._explore(now, perception_result)

        else:
            # If IDLE for > 5 seconds with no one visible, start exploring.
            # Without this, the dog sits frozen forever when no person is
            # in frame at startup. A real dog would wander to find its owner.
            if self.state == State.IDLE and now - self.last_owner_seen > 5.0:
                self._transition(State.EXPLORE)
                cmd = self._explore(now, perception_result)
            elif self.state == State.EXPLORE:
                # Already exploring from IDLE — keep going
                cmd = self._explore(now, perception_result)
            else:
                self._transition(State.IDLE)
                cmd = Command("stop", head_mode="remote", head_yaw=0.0)

        if now - self._last_cmd_time >= self._cmd_interval:
            self.send_command(cmd)
            self._last_cmd_time = now
        return cmd

    def _handle_maneuver(self, now):
        """
        Handle obstacle turn hold. With LiDAR, this is just a timed turn —
        no backup, no clearing, no multi-phase state machine. The LiDAR
        re-checks every frame after the hold expires, so if the obstacle
        is still there, it triggers another turn.

        Called from update() before any state logic. Returns a Command if
        a turn hold is active, or None to fall through to normal logic.
        """
        if self._scan_phase == "turning":
            if now > self._scan_phase_until:
                # Turn hold expired — resume normal behavior.
                # LiDAR will re-check on the next frame and trigger
                # another turn if the obstacle is still in the forward arc.
                print("[Follower] Obstacle turn complete — resuming")
                self._scan_phase = None
                self._scan_phase_cmd = None
                self._scan_result_direction = None
                return None
            return self._scan_phase_cmd  # keep turning

        return None

    def _find_owner(self, result):
        if not hasattr(result, 'detections') or not result.detections:
            return None

        # Priority 1: confirmed owner face match
        if hasattr(result, 'face_matches') and result.face_matches:
            for i, match in enumerate(result.face_matches):
                if match.is_owner and i < len(result.detections):
                    self._unconfirmed_count = 0
                    self._investigating = False
                    return result.detections[i]

        # Priority 2: approach unconfirmed person to investigate
        # Only chase high-confidence YOLO detections (>0.7) to avoid
        # following furniture/cabinets that YOLO misclassifies as people.
        # Never bark during investigation — only bark if we were already
        # confirmed following the owner and lost the face match.
        high_conf = [d for d in result.detections if d.confidence > 0.7]
        if high_conf:
            if self.state in (State.IDLE, State.SEARCH, State.EXPLORE):
                largest = max(high_conf, key=lambda d: d.area)
                self._investigating = True
                return largest
            elif self.state == State.FOLLOW:
                # Keep following largest detection (likely same person, bad angle)
                self._unconfirmed_count += 1
                largest = max(high_conf, key=lambda d: d.area)
                return largest

        # Low-confidence detections in FOLLOW — might be losing the person.
        # Don't bark at furniture. Only bark if we WERE following a confirmed
        # owner (investigating=False) and lost them at close range.
        if result.detections and self.state == State.FOLLOW and not self._investigating:
            self._unconfirmed_count += 1
            if self._unconfirmed_count > 50:
                self._should_bark = True
            largest = max(result.detections, key=lambda d: d.area)
            return largest

        return None

    def _consume_situation(self, result):
        """
        Phase 6A: Process new situation response with 2-consecutive-same
        hysteresis to prevent mode flickering from VLM inconsistency.
        At 2.5s query intervals, fastest possible switch is 5 seconds.
        """
        sit = result.situation_response
        seq = result.situation_seq

        if sit is None or seq == self._situation_seq_seen:
            return

        self._situation_seq_seen = seq
        new_mode = sit.mode

        if new_mode == self._behavior_mode:
            # Same as current — reset pending
            self._pending_mode = None
            self._pending_mode_count = 0
            return

        if new_mode == self._pending_mode:
            self._pending_mode_count += 1
        else:
            self._pending_mode = new_mode
            self._pending_mode_count = 1

        if self._pending_mode_count >= 2:
            old = self._behavior_mode
            self._behavior_mode = new_mode
            self._behavior_params = config.BEHAVIOR_MODES[new_mode]
            self._pending_mode = None
            self._pending_mode_count = 0
            print(f"[Follower] Behavior: {old} → {new_mode}")

    def _consume_explore(self, result):
        """Phase 6B: Read latest explore direction from VLM."""
        exp = result.explore_response
        seq = result.explore_seq

        if exp is None or seq == self._explore_seq_seen:
            # Check staleness
            timeout = getattr(config, "VLM_EXPLORE_DIRECTION_TIMEOUT_S", 5.0)
            if (self._explore_direction is not None and
                    time.time() - self._explore_direction_time > timeout):
                self._explore_direction = None
            return

        self._explore_seq_seen = seq
        self._explore_direction = exp.direction
        self._explore_direction_time = time.time()
        print(f"[Follower] Explore VLM: {exp.direction} — {exp.reasoning[:50]}")

    def _follow(self, owner_detection, result):
        """
        Core follow logic. Called each frame when owner IS detected.

        Head tracking: Pi handles it locally from the bbox (head_mode="local").
        Body movement: Mac decides action + speed based on safety hierarchy.
        Arrival: Uses bbox area ratio as distance proxy (no ultrasonic needed).
        """
        now = time.time()
        cx, cy = owner_detection.center
        offset_x = cx - config.FRAME_CENTER_X

        # ── Area ratio: distance proxy (big bbox = close) ──────────
        area = owner_detection.area
        frame_w, frame_h = config.CAMERA_RESOLUTION
        frame_area = frame_w * frame_h
        area_ratio = area / frame_area
        self._last_area_ratio = area_ratio  # expose for display overlay

        # ── Owner bbox for Pi-side head tracking ───────────────────
        # Pi receives this and computes head yaw/pitch locally (~5ms)
        # instead of waiting for Mac to compute and send back (~50ms).
        x1, y1, x2, y2 = owner_detection.bbox
        owner_bbox = {
            "x": int(x1), "y": int(y1),
            "w": int(x2 - x1), "h": int(y2 - y1),
            "frame_w": frame_w, "frame_h": frame_h,
        }

        # ── Calibration log (1/sec — not every frame) ─────────────
        # Use this output to tune ARRIVAL_*_RATIO in config.py.
        # Stand at different distances and note the area values.
        if now - self._last_follow_log >= 1.0:
            distance = self.telemetry.get_distance()
            print(f"[FOLLOW] area={area_ratio:.3f} dist={distance:.0f}cm "
                  f"off_x={offset_x:.0f}")
            self._last_follow_log = now

        self._consume_situation(result)
        distance = self.telemetry.get_distance()
        bp = self._behavior_params   # current behavior mode params

        # ═══ Safety hierarchy for body movement ═══════════════════
        #   1. LiDAR obstacle    — arc-based, instant, no VLM needed
        #   2. Bark hold         — sustain greeting while head is up
        #   3. Arrival bark      — close enough (bbox area), celebrate
        #   4. Body centering    — turn toward owner if off-center
        #   5. Default forward   — cruise toward owner

        # Priority 1: LiDAR obstacle avoidance (Phase 7).
        # Check forward arc every frame — if blocked, immediately pick
        # the clearer side and turn. No VLM call, no 3-phase maneuver,
        # no backup. Just check arcs and steer away.
        # Falls back to ultrasonic if LiDAR unavailable.
        obstacle_cm = getattr(config, "LIDAR_OBSTACLE_CM", 35)
        if self._lidar is not None and self._lidar.has_data():
            fwd_dist = self._lidar.get_forward_distance()
            if fwd_dist < obstacle_cm and self._scan_phase is None:
                left_dist = self._lidar.get_left_distance()
                right_dist = self._lidar.get_right_distance()
                if left_dist > right_dist:
                    turn_dir = "turn_left"
                else:
                    turn_dir = "turn_right"
                turn_steps = getattr(config, "LIDAR_TURN_STEP_COUNT", 12)
                turn_hold = getattr(config, "LIDAR_TURN_HOLD_S", 2.0)
                print(f"[Follower] LIDAR OBSTACLE fwd={fwd_dist:.0f}cm "
                      f"L={left_dist:.0f}cm R={right_dist:.0f}cm → {turn_dir}")
                self._scan_phase = "turning"
                self._scan_phase_until = now + turn_hold
                self._scan_phase_cmd = Command(
                    turn_dir, speed=90, step_count=turn_steps,
                    head_mode="local", owner_bbox=owner_bbox)
                self._scan_result_direction = turn_dir
                return self._scan_phase_cmd
        elif 0 < distance < config.ULTRASONIC_STOP_CM and self._scan_phase is None:
            # Ultrasonic fallback (no LiDAR) — simple turn left
            print(f"[Follower] ULTRASONIC OBSTACLE {distance:.0f}cm — turning left (no LiDAR)")
            self._scan_phase = "turning"
            self._scan_phase_until = now + 2.0
            self._scan_phase_cmd = Command(
                "turn_left", speed=90, step_count=12,
                head_mode="remote", head_yaw=0,
                head_pitch=config.HEAD_DEFAULT_PITCH)
            return self._scan_phase_cmd

        # Priority 2: Active bark hold — sustain greeting.
        if self._bark_hold_until > now:
            do_bark = (bp["bark_enabled"] and
                       now - self._last_bark_time >= self._bark_cooldown)
            if do_bark:
                self._last_bark_time = now
            return Command("stop", head_mode="remote",
                           head_yaw=0.0, head_pitch=config.BARK_HEAD_PITCH,
                           bark=do_bark, bark_volume=bp["bark_volume"],
                           idle_pose=bp["idle_pose"])

        # Priority 3: Arrival — owner close enough, stop and bark.
        # Uses dynamic arrival_ratio from behavior mode (e.g. GENTLE=0.30).
        if area_ratio > bp["arrival_ratio"]:
            # Re-center body toward owner before barking.
            if abs(offset_x) > 80:
                turn = "turn_left" if offset_x < 0 else "turn_right"
                print(f"[Follower] Arrival — re-centering {turn} "
                      f"(offset={offset_x:.0f})")
                return Command(turn, speed=70, step_count=2,
                               head_mode="local", owner_bbox=owner_bbox)
            # Centered — initiate bark hold
            print(f"[Follower] Arrival! area={area_ratio:.3f} "
                  f"mode={self._behavior_mode}")
            self._bark_hold_until = now + 5.0
            do_bark = (bp["bark_enabled"] and
                       now - self._last_bark_time >= self._bark_cooldown)
            if do_bark:
                self._last_bark_time = now
            return Command("stop", head_mode="remote",
                           head_yaw=0.0, head_pitch=config.BARK_HEAD_PITCH,
                           bark=do_bark, bark_volume=bp["bark_volume"],
                           idle_pose=bp["idle_pose"])

        # Dynamic speed from behavior mode (ACTIVE=98, GENTLE=60, etc.)
        speed = bp["speed"]

        # Priority 4: Body centering — distance-adaptive dead zone.
        # Far away: wide dead zone (200px) → go mostly straight, avoid obstacles.
        # Close up: tight dead zone (60px) → fine centering for arrival.
        if area_ratio < 0.08:
            turn_threshold = 220   # very far — almost never turn, just go straight
        elif area_ratio < 0.15:
            turn_threshold = 160   # far — gentle corrections only
        elif area_ratio < 0.30:
            turn_threshold = 100   # medium — moderate steering
        else:
            turn_threshold = 60    # close — tight centering for arrival

        # Hysteresis: once turning, keep turning until well-centered.
        TURN_HYSTERESIS = 30
        if self._last_follow_action in ("turn_left", "turn_right"):
            effective_threshold = max(turn_threshold - TURN_HYSTERESIS, 30)
        else:
            effective_threshold = turn_threshold

        # Decide wanted action
        if offset_x < -effective_threshold:
            wanted_action = "turn_left"
        elif offset_x > effective_threshold:
            wanted_action = "turn_right"
        else:
            wanted_action = "forward"

        # Minimum hold: commit to an action for at least 1 second before
        # switching. Each forward↔turn switch causes a servo snap because
        # the gait start positions are very different. Longer holds mean
        # fewer transitions and smoother movement.
        if wanted_action != self._last_follow_action:
            if now - self._last_follow_action_time >= 1.0:
                self._last_follow_action = wanted_action
                self._last_follow_action_time = now
            else:
                wanted_action = self._last_follow_action

        # Higher step_count so each gait runs longer between re-evaluations.
        # step_count=4 for turns (~1s), step_count=6 for forward (~1.5s).
        turn_steps = getattr(config, "BODY_TURN_STEP_COUNT", 4)
        fwd_steps = getattr(config, "FORWARD_STEP_COUNT", 8)
        if wanted_action in ("turn_left", "turn_right"):
            return Command(wanted_action, speed=speed, step_count=turn_steps,
                           head_mode="local", owner_bbox=owner_bbox)

        # Priority 5: Default — forward toward owner.
        return Command("forward", speed=speed, step_count=fwd_steps,
                       head_mode="local", owner_bbox=owner_bbox)

    def _search(self, now):
        # Smooth sinusoidal sweep ±35° over 4-second period.
        # Mac owns the head here (remote mode) — Pi applies these angles.
        # Pitch down so we're looking forward/toward the floor for doorways,
        # not up at the ceiling (head was pitched up during FOLLOW).
        elapsed = now - self.search_start_time
        head_yaw = 35.0 * math.sin(2 * math.pi * elapsed / 4.0)
        self._smoothed_head_yaw = head_yaw
        pitch = getattr(config, "EXPLORE_HEAD_PITCH", -10)
        return Command("stop", head_mode="remote",
                       head_yaw=head_yaw, head_pitch=pitch)

    def _explore(self, now, result):
        """
        Phase 6B: VLM-guided exploration. When owner is lost, the VLM
        tells the dog which direction to explore (doorways, hallways).

        Loop: pause + "thinking" animation → VLM responds with direction
        → dog walks that direction for a few steps → repeat.

        Falls back to head sweep if VLM doesn't respond.
        """
        self._consume_explore(result)

        pitch = getattr(config, "EXPLORE_HEAD_PITCH", -10)

        # ── LiDAR obstacle check (or ultrasonic fallback) ──────────────
        obstacle_cm = getattr(config, "LIDAR_OBSTACLE_CM", 35)
        if self._lidar is not None and self._lidar.has_data():
            fwd_dist = self._lidar.get_forward_distance()
            if fwd_dist < obstacle_cm and self._scan_phase is None:
                left_dist = self._lidar.get_left_distance()
                right_dist = self._lidar.get_right_distance()
                turn_dir = "turn_left" if left_dist > right_dist else "turn_right"
                turn_steps = getattr(config, "LIDAR_TURN_STEP_COUNT", 12)
                turn_hold = getattr(config, "LIDAR_TURN_HOLD_S", 2.0)
                print(f"[Follower] EXPLORE LIDAR fwd={fwd_dist:.0f}cm "
                      f"L={left_dist:.0f}cm R={right_dist:.0f}cm → {turn_dir}")
                self._explore_direction = None
                self._explore_move_until = 0.0
                self._scan_phase = "turning"
                self._scan_phase_until = now + turn_hold
                self._scan_phase_cmd = Command(
                    turn_dir, speed=90, step_count=turn_steps,
                    head_mode="remote", head_yaw=0, head_pitch=pitch)
                return self._scan_phase_cmd
        else:
            distance = self.telemetry.get_distance()
            if 0 < distance < config.ULTRASONIC_STOP_CM and self._scan_phase is None:
                print(f"[Follower] EXPLORE ULTRASONIC {distance:.0f}cm — turning left (no LiDAR)")
                self._explore_direction = None
                self._explore_move_until = 0.0
                self._scan_phase = "turning"
                self._scan_phase_until = now + 2.0
                self._scan_phase_cmd = Command(
                    "turn_left", speed=90, step_count=12,
                    head_mode="remote", head_yaw=0, head_pitch=pitch)
                return self._scan_phase_cmd

        # ── Hold check — if we're mid-turn or mid-move, finish it ────
        if self._explore_move_until > now:
            return self._last_explore_cmd

        # Check if VLM gave us a direction
        if self._explore_direction is not None:
            direction = self._explore_direction
            self._explore_direction = None  # consumed — next cycle queries again

            speed = getattr(config, "VLM_EXPLORE_SPEED", 70)
            steps = getattr(config, "VLM_EXPLORE_STEP_COUNT", 3)

            if direction == "FORWARD":
                cmd = Command("forward", speed=speed, step_count=steps,
                              head_mode="remote", head_yaw=0, head_pitch=pitch)
            elif direction == "LEFT":
                cmd = Command("turn_left", speed=80, step_count=4,
                              head_mode="remote", head_yaw=-30, head_pitch=pitch)
            elif direction == "RIGHT":
                cmd = Command("turn_right", speed=80, step_count=4,
                              head_mode="remote", head_yaw=30, head_pitch=pitch)
            elif direction == "BACK":
                # 180° turn — PiDog backward gait is useless, turn instead
                back_steps = getattr(config, "VLM_EXPLORE_BACK_STEPS", 8)
                cmd = Command("turn_left", speed=90, step_count=back_steps,
                              head_mode="remote", head_yaw=0, head_pitch=pitch)
            else:
                cmd = Command("forward", speed=speed, step_count=steps,
                              head_mode="remote", head_yaw=0, head_pitch=pitch)

            # Hold this move for ~2 seconds before querying VLM again
            self._explore_move_until = now + 2.0
            self._last_explore_cmd = cmd
            print(f"[Follower] Explore: moving {direction}")
            return cmd

        # No direction yet — "thinking" animation while waiting for VLM.
        # Gentle head oscillation + thinking flag triggers Pi-side purple
        # RGB + tail wag so the dog looks alive, not frozen. Pitch stays
        # down so the ribbon cam / webcam sees something navigable.
        elapsed = now - self.search_start_time
        head_tilt = 15.0 * math.sin(2 * math.pi * elapsed / 2.5)
        self._smoothed_head_yaw = head_tilt
        return Command("stop", head_mode="remote", head_yaw=head_tilt,
                       head_pitch=pitch, thinking=True)

    def _transition(self, new_state):
        if self.state != new_state:
            print(f"[Follower] {self.state.value} → {new_state.value}")
            self.state = new_state
            # Phase 6: Switch VLM query type based on state
            if self.pipeline:
                if new_state in (State.FOLLOW, State.SEARCH, State.IDLE):
                    self.pipeline.set_vlm_query_type("situation")
                elif new_state == State.EXPLORE:
                    self._explore_direction = None  # reset for fresh VLM query
                    self._explore_move_until = 0.0
                    self.pipeline.set_vlm_query_type("explore")

    def send_command(self, cmd):
        self.cmd_sock.send(cmd.to_json())

    def close(self):
        # Send stop so the Pi doesn't keep executing the last action
        try:
            self.send_command(Command("stop"))
            time.sleep(0.15)  # give ZMQ time to deliver before socket closes
        except Exception:
            pass
        self.telemetry.stop()
        self.cmd_sock.close()
        self.ctx.term()


if __name__ == "__main__":
    import cv2
    sys.path.insert(0, str(Path(__file__).parent))
    from perception_pipeline import PerceptionPipeline

    print("=" * 60)
    print("IronBark Phase 6 — Follow-Me (Situation-Aware VLA)")
    print("=" * 60)

    ctx = zmq.Context()
    # Main (owner-tracking) stream — USB webcam mounted on the dog's head
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVHWM, 2)      # only buffer 2 frames max
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://{config.PC_IP}:{config.ZMQ_PORT}")
    print(f"[Main] PULL socket bound on {config.PC_IP}:{config.ZMQ_PORT} (webcam)")

    # Phase 6 dual-camera: navigation stream — ribbon camera on the nose.
    # Optional — if the Pi isn't running a second pi_sender on ZMQ_NAV_PORT,
    # nav_sock.recv returns zmq.Again and we fall back to the webcam.
    nav_sock = None
    if getattr(config, "USE_RIBBON_CAM", False):
        nav_sock = ctx.socket(zmq.PULL)
        nav_sock.setsockopt(zmq.CONFLATE, 1)
        nav_sock.setsockopt(zmq.RCVHWM, 2)
        nav_sock.setsockopt(zmq.RCVTIMEO, 100)
        nav_sock.bind(f"tcp://{config.PC_IP}:{config.ZMQ_NAV_PORT}")
        print(f"[Main] Nav PULL socket bound on {config.PC_IP}:{config.ZMQ_NAV_PORT} (ribbon cam)")

    pipeline = PerceptionPipeline(config)
    pipeline.start()
    follower = Follower()
    follower.pipeline = pipeline  # Phase 6: follower switches VLM query type

    # Phase 7: LiDAR obstacle avoidance (optional — falls back to ultrasonic)
    if getattr(config, "USE_LIDAR", False):
        try:
            from lidar_map import LidarMap
            lidar = LidarMap(ctx)
            lidar.start()
            follower._lidar = lidar
            print("[Main] LiDAR obstacle avoidance enabled")
        except ImportError:
            print("[Main] LiDAR module not found — using ultrasonic fallback")
        except Exception as e:
            print(f"[Main] LiDAR init failed: {e} — using ultrasonic fallback")

    window = "IronBark Follow-Me"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    nav_window = "IronBark Ribbon Cam"
    cv2.namedWindow(nav_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nav_window, 640, 480)

    def get_latest_frame(socket):
        msg = None
        while True:
            try:
                msg = socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
        return msg

    def decode_frame(raw):
        header_size = struct.calcsize("<qI")
        if len(raw) < header_size:
            return None
        timestamp, payload_len = struct.unpack_from("<qI", raw, 0)
        jpeg_data = raw[header_size:header_size + payload_len]
        return cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Latest nav frame, refreshed each loop iteration. Held across iterations
    # so we can forward the most recent available ribbon-cam frame to the
    # VLM even if the ribbon cam pushes slower than the webcam.
    latest_nav_frame = None

    try:
        while True:
            raw = get_latest_frame(sock)
            if raw is None:
                try:
                    raw = sock.recv()
                except zmq.Again:
                    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(waiting, "Waiting for Pi stream...",
                                (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                    cv2.imshow(window, waiting)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    continue

            frame = decode_frame(raw)
            if frame is None:
                continue

            # Drain any pending nav frames (non-blocking). Only the latest
            # survives — everything older is discarded (CONFLATE also helps).
            if nav_sock is not None:
                nav_raw = get_latest_frame(nav_sock)
                if nav_raw is not None:
                    decoded = decode_frame(nav_raw)
                    if decoded is not None:
                        latest_nav_frame = decoded
                        follower._latest_nav_frame = decoded
                        follower._latest_nav_frame_time = time.time()

            result = pipeline.process_frame(frame, nav_frame=latest_nav_frame)
            cmd = follower.update(result)

            display = pipeline.draw_overlay(frame, result)

            # Bottom status bar: state + command + ultrasonic
            dist = follower.telemetry.get_distance()
            if 0 < dist < config.ULTRASONIC_STOP_CM:
                dist_text = f"Dist: {dist:.0f}cm STOPPED"
                dist_color = (0, 0, 255)
            elif 0 < dist < config.ULTRASONIC_SLOW_CM:
                dist_text = f"Dist: {dist:.0f}cm SLOW"
                dist_color = (0, 180, 255)
            elif dist > 0:
                dist_text = f"Dist: {dist:.0f}cm"
                dist_color = (0, 255, 0)
            else:
                dist_text = "Dist: N/A"
                dist_color = (150, 150, 150)

            state_text = f"State: {follower.state.value} | Cmd: {cmd.action} spd={cmd.speed}"
            mode_text = f"Mode: {follower._behavior_mode} | spd={follower._behavior_params['speed']}"
            # Area ratio display
            ar = follower._last_area_ratio
            arrival_r = follower._behavior_params["arrival_ratio"]
            if ar > arrival_r:
                ar_text = f"Area: {ar:.3f} ARRIVED"
                ar_color = (0, 255, 255)
            elif ar > 0:
                ar_text = f"Area: {ar:.3f}"
                ar_color = (0, 255, 0)
            else:
                ar_text = ""
                ar_color = (150, 150, 150)
            cv2.putText(display, state_text, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(display, mode_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
            cv2.putText(display, dist_text, (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
            if ar_text:
                cv2.putText(display, ar_text, (450, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ar_color, 1)
            cv2.imshow(window, display)

            # Show ribbon cam in second window with action debug info
            if latest_nav_frame is not None:
                nav_display = latest_nav_frame.copy()
                cv2.putText(nav_display, "Ribbon Cam (obstacle detection)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

                # Ribbon window shows what the VLM/perception wants
                # vs what the follower is actually sending to the Pi
                cur_action = cmd.action.upper().replace("_", " ")
                vlm_action = follower._last_follow_action.upper().replace("_", " ")
                state_name = follower.state.value

                # Color code current action
                action_colors = {
                    "FORWARD": (0, 255, 0),       # green
                    "TURN LEFT": (255, 200, 0),    # cyan
                    "TURN RIGHT": (0, 200, 255),   # orange
                    "BACKWARD": (0, 0, 255),       # red
                    "STOP": (150, 150, 150),       # gray
                    "OBSTACLE SCAN": (0, 255, 255), # yellow
                }
                cur_color = action_colors.get(cur_action, (255, 255, 255))

                cv2.putText(nav_display, f"VLM wants:    {vlm_action}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                cv2.putText(nav_display, f"Actual cmd:   {cur_action}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cur_color, 2)
                cv2.putText(nav_display, f"State: {state_name}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                # Scan phase if active
                if follower._scan_phase is not None:
                    scan_dir = follower._scan_result_direction or "?"
                    cv2.putText(nav_display, f"Maneuver: {follower._scan_phase} -> {scan_dir}",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

                # Distance bar at bottom
                cv2.putText(nav_display, f"Dist: {dist:.0f}cm" if dist > 0 else "Dist: N/A",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)

                cv2.imshow(nav_window, nav_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        follower.close()
        pipeline.stop()
        sock.close()
        if nav_sock is not None:
            nav_sock.close()
        cv2.destroyAllWindows()
