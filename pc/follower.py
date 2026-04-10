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

    def stop(self):
        self.running = False
        self.sock.close()


class Follower:
    def __init__(self):
        self.state = State.IDLE
        self.last_owner_seen = 0.0
        self.search_start_time = 0.0
        self._unconfirmed_count = 0
        self._should_bark = False
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
        self.cmd_sock.bind(f"tcp://*:{config.CMD_PORT}")
        print(f"[Follower] Command PUB bound on port {config.CMD_PORT}")

        self.telemetry = TelemetryReceiver(self.ctx)

    def update(self, perception_result):
        now = time.time()
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
            self._transition(State.IDLE)
            cmd = Command("stop", head_mode="remote", head_yaw=0.0)

        if now - self._last_cmd_time >= self._cmd_interval:
            self.send_command(cmd)
            self._last_cmd_time = now
        return cmd

    def _find_owner(self, result):
        if not hasattr(result, 'detections') or not result.detections:
            return None
        if hasattr(result, 'face_matches') and result.face_matches:
            for i, match in enumerate(result.face_matches):
                if match.is_owner and i < len(result.detections):
                    self._unconfirmed_count = 0
                    return result.detections[i]
        if len(result.detections) > 0 and self.state == State.FOLLOW:
            self._unconfirmed_count += 1
            if self._unconfirmed_count > 10:
                self._should_bark = True
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
        #   1. Ultrasonic stop   — hard safety, always wins
        #   2. Bark hold         — sustain greeting while head is up
        #   3. Arrival bark      — close enough (bbox area), celebrate
        #   4. Body centering    — turn toward owner if off-center
        #   5. Default forward   — cruise toward owner

        # Priority 1: Ultrasonic obstacle — redirect turn toward owner.
        if 0 < distance < config.ULTRASONIC_STOP_CM:
            if offset_x < -30:
                return Command("turn_left", speed=90, step_count=3,
                               head_mode="local", owner_bbox=owner_bbox)
            else:
                return Command("turn_right", speed=90, step_count=3,
                               head_mode="local", owner_bbox=owner_bbox)

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
        step_count = 2

        # Priority 4: Body centering — turn toward owner.
        if offset_x < -config.BODY_TURN_THRESHOLD:
            return Command("turn_left", speed=speed, step_count=step_count,
                           head_mode="local", owner_bbox=owner_bbox)
        if offset_x > config.BODY_TURN_THRESHOLD:
            return Command("turn_right", speed=speed, step_count=step_count,
                           head_mode="local", owner_bbox=owner_bbox)

        # Priority 5: Default — forward toward owner.
        return Command("forward", speed=speed, step_count=step_count,
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

        # If currently executing a VLM-directed move, keep going
        if self._explore_move_until > now:
            return self._last_explore_cmd

        pitch = getattr(config, "EXPLORE_HEAD_PITCH", -10)

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
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    print(f"[Main] PULL socket bound on port {config.ZMQ_PORT} (webcam)")

    # Phase 6 dual-camera: navigation stream — ribbon camera on the nose.
    # Optional — if the Pi isn't running a second pi_sender on ZMQ_NAV_PORT,
    # nav_sock.recv returns zmq.Again and we fall back to the webcam.
    nav_sock = None
    if getattr(config, "USE_RIBBON_CAM", False):
        nav_sock = ctx.socket(zmq.PULL)
        nav_sock.setsockopt(zmq.CONFLATE, 1)
        nav_sock.setsockopt(zmq.RCVHWM, 2)
        nav_sock.setsockopt(zmq.RCVTIMEO, 100)
        nav_sock.bind(f"tcp://*:{config.ZMQ_NAV_PORT}")
        print(f"[Main] Nav PULL socket bound on port {config.ZMQ_NAV_PORT} (ribbon cam)")

    pipeline = PerceptionPipeline(config)
    pipeline.start()
    follower = Follower()
    follower.pipeline = pipeline  # Phase 6: follower switches VLM query type

    window = "IronBark Follow-Me"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

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
