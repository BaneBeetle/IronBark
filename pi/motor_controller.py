"""
pi/motor_controller.py — IronBark Unified Motor Controller (Pi Side)
Receives commands from BOTH follow-me (PC_IP:5556) and teleop (bind 5557),
executes via PiDog SDK, publishes telemetry on 5558.

NEW in this version:
  - LocalHeadTracker: Pi computes head yaw/pitch locally from owner bbox
    sent by the Mac. Eliminates ~50ms network round trip for head movement.
  - head_mode: "local" (Pi owns head from bbox) vs "remote" (Mac sends angles)
  - FOLLOW_GAIT_PAUSE_S pulled from config (Part A: reduced from 0.2 to 0.05)

Run on Pi with: sudo python3 motor_controller.py
"""

import os
import sys
import signal
import time
import json
import atexit
import subprocess

import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config



# ═══════════════════════════════════════════════════════════════════════════
# GPIO Cleanup — runs BEFORE PiDog init to kill orphaned sensor processes
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_orphaned_pidog():
    """
    Kill any leftover PiDog sensor subprocesses from a previous crash.

    When motor_controller is killed with SIGKILL (kill -9), _shutdown()
    never runs. PiDog's internal sensory_process (ultrasonic reader)
    becomes an orphan that holds GPIO pins, causing 'GPIO busy' on
    the next startup.

    This function finds and kills those orphans before we init PiDog.
    """
    try:
        # Find python3 processes that look like PiDog sensor subprocesses
        # They're spawned by multiprocessing and have 'pidog' in their cmdline
        result = subprocess.run(
            ["pgrep", "-f", "pidog.*sensory|sensory.*pidog|from multiprocessing"],
            capture_output=True, text=True, timeout=3
        )
        pids = result.stdout.strip().split("\n")
        my_pid = str(os.getpid())
        for pid in pids:
            pid = pid.strip()
            if pid and pid != my_pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"[Cleanup] Killed orphaned PiDog process {pid}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass

    # Also release GPIO via lgpio if available (Pi 5 uses lgpio, not RPi.GPIO)
    try:
        import lgpio
        for chip in range(5):  # gpiochip0-4
            try:
                h = lgpio.gpiochip_open(chip)
                lgpio.gpiochip_close(h)
            except Exception:
                pass
    except ImportError:
        pass

    # Small delay to let OS fully release resources
    time.sleep(0.3)
    print("[Cleanup] GPIO pre-cleanup done.")


from pidog import Pidog

# ── Config ──────────────────────────────────────────────────────────────────
DANGER_DISTANCE = getattr(config, "DANGER_DISTANCE", 15)
FOLLOW_CMD_PORT = getattr(config, "CMD_PORT", 5556)
TELEOP_CMD_PORT = getattr(config, "REMOTE_CMD_PORT", 5557)
TELEM_PORT = getattr(config, "REMOTE_TELEM_PORT", 5558)
PC_IP = config.PC_IP

# Part A: Reduced from 0.2 → 0.05 for snappier follow-me gait cycles.
# Teleop has natural gaps from key release/re-press; this only affects follow.
FOLLOW_GAIT_PAUSE_S = getattr(config, "FOLLOW_GAIT_PAUSE_S", 0.05)


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Pi-Side Head Tracking
# ═══════════════════════════════════════════════════════════════════════════

class LocalHeadTracker:
    """
    Computes head yaw/pitch from owner bbox received from Mac.
    Runs every loop iteration on the Pi (~20Hz) — no network round trip.

    How it works:
      1. Mac detects owner with YOLO+ArcFace, sends bbox in command JSON
      2. Pi receives bbox, computes target yaw/pitch
      3. Exponential smoothing creates smooth head movement
      4. head_move() called locally — instant response

    This makes head tracking feel 50ms faster (servo responds in ~20ms
    vs ~70ms when waiting for Mac to compute + send back).
    """

    def __init__(self):
        self.last_bbox = None           # Latest bbox from Mac
        self.current_yaw = 0.0          # Smoothed output yaw
        self.current_pitch = 15.0       # Smoothed output pitch (slightly up)

        # Smoothing factor: 0.7 = weight on old value.
        # 30% of new target applied each step. At 20Hz, takes ~5 steps
        # (~250ms) to converge. Matches existing follower.py feel.
        self.yaw_smoothing = 0.7
        self.pitch_smoothing = 0.7

        # Angle dead zone: don't update if target is within this many
        # degrees of current — prevents micro-jitter.
        self.angle_dead_zone = 10.0     # degrees

        # Hard limits (match PiDog servo range)
        self.yaw_min, self.yaw_max = -45, 45
        self.pitch_min, self.pitch_max = 5, 25

        # Used for pitch computation — how big "ideal distance" bbox is
        self.target_area_ratio = getattr(config, "TARGET_AREA_RATIO", 0.12)

    def update_bbox(self, bbox_data):
        """
        Feed a new owner bbox from the Mac.
        Called whenever a command with owner_bbox arrives.
        bbox_data is a dict: {x, y, w, h, frame_w, frame_h}
        """
        self.last_bbox = bbox_data

    def compute_head_angles(self):
        """
        Compute smoothed yaw/pitch from the latest bbox.
        Call this EVERY loop iteration for smooth tracking.
        Returns (yaw, pitch) or None if no target.
        """
        if self.last_bbox is None:
            return None

        bbox = self.last_bbox
        frame_w = bbox["frame_w"]
        frame_h = bbox["frame_h"]
        frame_cx = frame_w / 2

        # ── Bbox center ────────────────────────────────────────
        cx = bbox["x"] + bbox["w"] / 2

        # ── Yaw (horizontal tracking) ─────────────────────────
        # offset_x > 0 means owner is on the right side of frame.
        # We want head to look right → negative yaw on PiDog.
        # (The negative sign was confirmed to work in earlier testing.)
        offset_x = cx - frame_cx
        target_yaw = max(self.yaw_min,
                         min(self.yaw_max, -offset_x * 45 / frame_cx))

        # Angle dead zone: if change is small, hold position (anti-jitter)
        if abs(target_yaw - self.current_yaw) < self.angle_dead_zone:
            target_yaw = self.current_yaw

        # ── Pitch (vertical tracking) ─────────────────────────
        # PiDog is low and owner is standing → look up.
        # Bigger bbox = closer = pitch up more, smaller = farther = pitch less.
        area_ratio = (bbox["w"] * bbox["h"]) / (frame_w * frame_h)
        target_pitch = 20 - (area_ratio / self.target_area_ratio) * 15
        target_pitch = max(self.pitch_min, min(self.pitch_max, target_pitch))

        # ── Exponential smoothing ─────────────────────────────
        # smooth = old * alpha + new * (1 - alpha)
        # Higher alpha = smoother but slower to react.
        self.current_yaw = (self.yaw_smoothing * self.current_yaw +
                            (1 - self.yaw_smoothing) * target_yaw)
        self.current_pitch = (self.pitch_smoothing * self.current_pitch +
                              (1 - self.pitch_smoothing) * target_pitch)

        return (self.current_yaw, self.current_pitch)


# ═══════════════════════════════════════════════════════════════════════════
# Motor Controller
# ═══════════════════════════════════════════════════════════════════════════

class MotorController:
    """Unified Pi-side motor controller for follow-me and teleop."""

    ACTION_MAP = {
        "forward":    "forward",
        "backward":   "backward",
        "turn_left":  "turn_left",
        "turn_right": "turn_right",
    }

    def __init__(self):
        print("=" * 60)
        print("  IronBark — Unified Motor Controller (Pi)")
        print("=" * 60)

        self._shutdown_done = False  # prevent double-shutdown

        # ── Clean up orphaned GPIO from previous crash ─────────────
        cleanup_orphaned_pidog()

        # ── PiDog init ──────────────────────────────────────────────
        print("[Motor] Initializing PiDog...")
        self.dog = Pidog()
        self.dog.do_action("stand", speed=80)
        self.dog.wait_all_done()
        time.sleep(0.5)

        # ── Register cleanup for ANY exit (SIGTERM, atexit, etc.) ──
        # SIGTERM (plain `kill`) — triggers graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        # atexit — last resort if Python exits without signal
        atexit.register(self._shutdown)

        self.stand_angles = self.dog.legs_angle_calculation(
            [[0, 80], [0, 80], [30, 75], [30, 75]]
        )

        # ── Pi-side head tracker (Part B) ──────────────────────────
        self.head_tracker = LocalHeadTracker()
        self.head_mode = "remote"   # default: Mac controls head

        # ── ZMQ sockets ────────────────────────────────────────────
        self.ctx = zmq.Context()

        # SUB: follow-me commands — connect to PC's PUB on port 5556
        self.follow_sock = self.ctx.socket(zmq.SUB)
        self.follow_sock.setsockopt(zmq.CONFLATE, 1)
        self.follow_sock.connect(f"tcp://{PC_IP}:{FOLLOW_CMD_PORT}")
        self.follow_sock.subscribe(b"")
        print(f"[Motor] Follow SUB connected to tcp://{PC_IP}:{FOLLOW_CMD_PORT}")

        # SUB: teleop commands — bind on port 5557 (Mac connects to us)
        self.teleop_sock = self.ctx.socket(zmq.SUB)
        self.teleop_sock.setsockopt(zmq.CONFLATE, 1)
        self.teleop_sock.bind(f"tcp://*:{TELEOP_CMD_PORT}")
        self.teleop_sock.subscribe(b"")
        print(f"[Motor] Teleop SUB bound on tcp://*:{TELEOP_CMD_PORT}")

        # PUB: telemetry — bind on port 5558 (Mac connects to us)
        self.telem_sock = self.ctx.socket(zmq.PUB)
        self.telem_sock.bind(f"tcp://*:{TELEM_PORT}")
        print(f"[Motor] Telemetry PUB bound on tcp://*:{TELEM_PORT}")

        # Poller for multiplexing both command sources
        self.poller = zmq.Poller()
        self.poller.register(self.follow_sock, zmq.POLLIN)
        self.poller.register(self.teleop_sock, zmq.POLLIN)

        # ── State ───────────────────────────────────────────────────
        self.current_action = "stop"
        self.current_speed = 80
        self.current_step_count = 2
        self.source = None          # "follow" or "teleop"
        self.danger = False
        self.running = True
        self.last_telem_time = 0.0
        self.gait_done_time = 0.0   # tracks when last gait finished

        self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
        print("[Motor] Ready. Waiting for commands...")

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        try:
            while self.running:
                # Read ultrasonic
                distance = round(self.dog.read_distance(), 2)

                # Poll both command sources
                action, speed, step_count, head_yaw, head_pitch, bark, \
                    source, head_mode, owner_bbox, \
                    bark_volume, idle_pose, thinking = self._receive_command()

                # ── Update head tracker state ──────────────────────
                # If this command has a bbox, feed it to the tracker.
                # If not, the tracker keeps its last known bbox (coasting).
                if owner_bbox is not None:
                    self.head_tracker.update_bbox(owner_bbox)
                if head_mode is not None:
                    self.head_mode = head_mode

                # Ultrasonic danger zone (hard safety net)
                # Block forward movement, but ALLOW turns — the Mac sends
                # redirect turns during danger to pivot the dog away from
                # the obstacle. Turns include some forward movement but
                # the lateral component moves the dog out of danger.
                if 0 < distance < DANGER_DISTANCE:
                    if action == "forward":
                        # Pure forward into obstacle — block it
                        if not self.danger:
                            print(f"\033[0;31m[Motor] DANGER! {distance}cm "
                                  f"— blocking forward\033[m")
                            self.dog.body_stop()
                            self.dog.rgb_strip.set_mode("bark", "red", bps=2)
                            self.dog.legs_move([self.stand_angles], speed=70)
                            self.dog.wait_all_done()
                            self.danger = True
                            self._bark_warning()
                        action = "stop"
                    elif action in ("turn_left", "turn_right"):
                        # Redirect turn — allow it through (Mac is pivoting
                        # the dog away from the obstacle toward the owner)
                        if not self.danger:
                            print(f"\033[0;33m[Motor] DANGER {distance}cm "
                                  f"— redirecting {action}\033[m")
                        self.danger = True
                        self.dog.rgb_strip.set_mode("bark", "yellow", bps=2)
                        # action passes through to _execute() as-is
                else:
                    if self.danger:
                        print("\033[0;32m[Motor] Clear — danger resolved\033[m")
                        self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
                        self.danger = False

                # ── Head tracking (every iteration, independent of motors) ──
                # In "local" mode: Pi computes head from bbox (~0ms, no network)
                # In "remote" mode: Mac sent explicit angles (search/explore/bark)
                if self.head_mode == "local":
                    angles = self.head_tracker.compute_head_angles()
                    if angles:
                        yaw, pitch = angles
                        self.dog.head_move([[yaw, 0, pitch]], speed=80)
                elif head_yaw is not None or head_pitch is not None:
                    # Remote mode — apply Mac's commanded angles
                    yaw = head_yaw if head_yaw is not None else 0
                    pitch = head_pitch if head_pitch is not None \
                        else config.HEAD_DEFAULT_PITCH
                    self.dog.head_move([[yaw, 0, pitch]], speed=80)

                # Execute movement
                self._execute(action, speed, step_count, source, bark,
                              bark_volume, idle_pose, thinking)

                # Publish telemetry at ~5 Hz
                now = time.time()
                if now - self.last_telem_time >= 0.2:
                    self._send_telemetry(distance)
                    self.last_telem_time = now

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[Motor] Ctrl+C — shutting down...")
        finally:
            self._shutdown()

    # ── Command reception ───────────────────────────────────────────────────

    def _receive_command(self):
        """
        Poll both sockets. If both have data, last processed wins.
        Returns motor params + head_mode + owner_bbox from JSON.
        """
        action = self.current_action
        speed = self.current_speed
        step_count = self.current_step_count
        head_yaw = None
        head_pitch = None
        bark = False
        source = self.source
        head_mode = None        # None = no new command, keep current mode
        owner_bbox = None       # None = no new bbox, tracker keeps last known

        socks = dict(self.poller.poll(50))

        for sock, src_name in [(self.teleop_sock, "teleop"),
                               (self.follow_sock, "follow")]:
            if sock in socks:
                try:
                    raw = sock.recv()
                    msg = json.loads(raw.decode("utf-8"))
                    action = msg.get("action", "stop")
                    speed = msg.get("speed", 80)
                    step_count = msg.get("step_count", 2)
                    head_yaw = msg.get("head_yaw")
                    head_pitch = msg.get("head_pitch")
                    bark = msg.get("bark", False)
                    source = src_name

                    # Part B: head tracking mode + owner bbox
                    head_mode = msg.get("head_mode", "remote")
                    owner_bbox = msg.get("owner_bbox")

                    # Phase 6: new command fields
                    bark_volume = msg.get("bark_volume", 80)
                    idle_pose = msg.get("idle_pose")    # "stand"/"sit"/"lie"
                    thinking = msg.get("thinking", False)

                except Exception as e:
                    print(f"[Motor] Parse error ({src_name}): {e}")

        # Ensure Phase 6 vars exist even if no command received this cycle
        bark_volume = locals().get("bark_volume", 80)
        idle_pose = locals().get("idle_pose", None)
        thinking = locals().get("thinking", False)

        return (action, speed, step_count, head_yaw, head_pitch,
                bark, source, head_mode, owner_bbox,
                bark_volume, idle_pose, thinking)

    # ── Movement execution ──────────────────────────────────────────────────

    def _execute(self, action, speed, step_count, source, bark,
                 bark_volume=80, idle_pose=None, thinking=False):
        """Execute action via PiDog SDK. Respects speed/step_count from sender."""

        # Phase 6B: "Thinking" animation — dog is waiting for VLM explore
        if thinking:
            if self.current_action != "thinking":
                self.dog.body_stop()
                self.dog.rgb_strip.set_mode("breath", [128, 0, 128], bps=1.5)
                self.dog.do_action("wag_tail", step_count=3, speed=40)
                self.current_action = "thinking"
                print(f"[Motor] -> THINKING ({source})")
            return

        # Bark request — arrival celebration: head-only bark (no legs)
        if bark:
            self.dog.body_stop()
            self.dog.rgb_strip.set_mode("bark", "yellow", bps=2)
            self.dog.do_action("wag_tail", step_count=5, speed=99)
            # Head up — keep current yaw, pitch up to bark angle
            cur_yaw = self.dog.head_current_angles[0]
            bark_pitch = getattr(config, "BARK_HEAD_PITCH", 35)
            self.dog.head_move([[cur_yaw, 0, bark_pitch]], speed=80)
            self.dog.wait_head_done()
            # Bark sound — volume from behavior mode (ACTIVE=80, GENTLE=40)
            if bark_volume > 0:
                try:
                    self.dog.speak("single_bark_1", bark_volume)
                except Exception:
                    pass
                time.sleep(0.5)
            # Head back down
            self.dog.head_move([[cur_yaw, 0, config.HEAD_DEFAULT_PITCH]], speed=60)
            # Phase 6: idle pose after bark (GENTLE→sit, CALM→lie, etc.)
            if idle_pose in ("sit", "lie"):
                self.dog.wait_head_done()
                self.dog.do_action(idle_pose, speed=80)
                self.dog.wait_all_done()
                print(f"[Motor] -> BARK + {idle_pose.upper()} ({source})")
            else:
                print(f"[Motor] -> BARK arrival ({source})")
            self.dog.rgb_strip.set_mode("breath", "green", bps=1)
            self.current_action = "stop"
            return

        # Same action continuing
        if action == self.current_action:
            if action in self.ACTION_MAP:
                # Movement action — re-issue gait when legs finish
                if self.dog.is_legs_done():
                    # Follow-me pacing: brief pause between gaits
                    if source == "follow":
                        now = time.time()
                        if now - self.gait_done_time < FOLLOW_GAIT_PAUSE_S:
                            return
                        self.gait_done_time = now

                    sdk_name = self.ACTION_MAP[action]
                    self.dog.do_action(sdk_name, step_count=step_count, speed=speed)
            # Non-movement (stop/stand/sit/lie) — already in this state,
            # don't re-issue body_stop() or it cancels head movements.
            return

        # ── Action changed ──────────────────────────────────────────
        self.current_action = action
        self.current_speed = speed
        self.current_step_count = step_count
        self.source = source

        if action == "stop":
            self.dog.body_stop()
            self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
            print(f"[Motor] -> STOP ({source})")

        elif action in self.ACTION_MAP:
            sdk_name = self.ACTION_MAP[action]
            self.dog.do_action(sdk_name, step_count=step_count, speed=speed)
            self.dog.do_action("wag_tail", step_count=5, speed=99)
            self.dog.rgb_strip.set_mode("breath", "green", bps=1)
            self.gait_done_time = time.time()
            print(f"[Motor] -> {action.upper()} spd={speed} ({source})")

        elif action == "stand":
            self.dog.body_stop()
            self.dog.do_action("stand", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print(f"[Motor] -> STAND ({source})")

        elif action == "sit":
            self.dog.body_stop()
            self.dog.do_action("sit", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print(f"[Motor] -> SIT ({source})")

        elif action == "lie":
            self.dog.body_stop()
            self.dog.do_action("lie", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print(f"[Motor] -> LIE ({source})")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _bark_warning(self):
        """Danger bark — head-only, same pattern as arrival bark."""
        try:
            cur_yaw = self.dog.head_current_angles[0]
            bark_pitch = getattr(config, "BARK_HEAD_PITCH", 35)
            self.dog.head_move([[cur_yaw, 0, bark_pitch]], speed=80)
            self.dog.wait_head_done()
            self.dog.speak("single_bark_1", 80)
            time.sleep(0.4)
            self.dog.head_move([[cur_yaw, 0, config.HEAD_DEFAULT_PITCH]], speed=60)
        except Exception:
            pass

    def _send_telemetry(self, distance):
        try:
            battery = self.dog.get_battery_voltage()
        except Exception:
            battery = 0.0

        telem = {
            "distance_cm": distance,
            "battery_v": battery,
            "action": self.current_action,
            "danger": self.danger,
            "source": self.source,
            "timestamp": time.time(),
        }
        try:
            self.telem_sock.send_string(json.dumps(telem))
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """Handle SIGTERM (plain `kill` command) — triggers clean shutdown."""
        print(f"\n[Motor] Received signal {signum} — shutting down...")
        self.running = False
        self._shutdown()
        sys.exit(0)

    def _shutdown(self):
        """
        Clean shutdown: stop dog, kill sensor process, release GPIO, close ZMQ.
        Safe to call multiple times (atexit + signal + finally).
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        print("[Motor] Shutting down PiDog...")

        # Stop all movement
        try:
            self.dog.body_stop()
            self.dog.do_action("lie", speed=80)
            self.dog.wait_all_done()
        except Exception:
            pass

        # Kill the ultrasonic sensor subprocess (this is what holds GPIO!)
        sp = getattr(self.dog, "sensory_process", None)
        if sp is not None and sp.is_alive():
            sp.terminate()
            sp.join(timeout=2.0)
            if sp.is_alive():
                try:
                    os.kill(sp.pid, signal.SIGKILL)
                    print("[Motor] Force-killed sensor process")
                except Exception:
                    pass

        # Release PiDog hardware (GPIO, I2C, PWM)
        try:
            self.dog.close()
        except SystemExit:
            pass
        except Exception:
            pass

        # Close ZMQ sockets
        try:
            self.follow_sock.close()
            self.teleop_sock.close()
            self.telem_sock.close()
            self.ctx.term()
        except Exception:
            pass
        print("[Motor] Done.")


if __name__ == "__main__":
    controller = MotorController()
    controller.run()
