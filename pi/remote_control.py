"""
pi/remote_control.py — IronBark WASD Teleoperation (Pi Side)
Receives movement commands from Mac over ZMQ and drives PiDog motors.
Run on Pi with: sudo python3 remote_control.py
"""

import os
import sys
import signal
import time
import json

import zmq

# ── PiDog SDK ────────────────────────────────────────────────────────────────
from pidog import Pidog

# ── Config (shared with Mac) ─────────────────────────────────────────────────
# When deployed to Pi, config.py lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

try:
    from preset_actions import bark as bark_action
except ImportError:
    bark_action = None

# ── Constants ────────────────────────────────────────────────────────────────
DANGER_DISTANCE = getattr(config, "DANGER_DISTANCE", 15)
CMD_PORT = getattr(config, "REMOTE_CMD_PORT", 5557)
TELEM_PORT = getattr(config, "REMOTE_TELEM_PORT", 5558)


class RemoteController:
    """Receives WASD commands from Mac and drives the PiDog."""

    # Map action names to PiDog SDK calls
    ACTION_MAP = {
        "forward":    ("forward",    2, 98),
        "backward":   ("backward",   2, 98),
        "turn_left":  ("turn_left",  2, 98),
        "turn_right": ("turn_right", 2, 98),
    }

    def __init__(self):
        print("=" * 60)
        print("  IronBark — Remote Control (Pi Side)")
        print("=" * 60)

        # ── Init PiDog ───────────────────────────────────────────────
        print("[RC] Initializing PiDog...")
        self.dog = Pidog()
        self.dog.do_action("stand", speed=80)
        self.dog.wait_all_done()
        time.sleep(0.5)

        # Pre-compute stand pose for recovery after danger stop
        self.stand_angles = self.dog.legs_angle_calculation(
            [[0, 80], [0, 80], [30, 75], [30, 75]]
        )

        # ── ZMQ sockets ─────────────────────────────────────────────
        self.ctx = zmq.Context()

        # SUB socket: receive commands from Mac
        self.cmd_sock = self.ctx.socket(zmq.SUB)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)  # keep only latest
        self.cmd_sock.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        self.cmd_sock.bind(f"tcp://*:{CMD_PORT}")
        self.cmd_sock.subscribe(b"")
        print(f"[RC] CMD SUB bound on tcp://*:{CMD_PORT}")

        # PUB socket: send telemetry back to Mac
        self.telem_sock = self.ctx.socket(zmq.PUB)
        self.telem_sock.bind(f"tcp://*:{TELEM_PORT}")
        print(f"[RC] TELEM PUB bound on tcp://*:{TELEM_PORT}")

        # ── State ────────────────────────────────────────────────────
        self.current_action = "stop"
        self.danger = False
        self.running = True
        self.last_telem_time = 0.0

        self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
        print("[RC] Ready. Waiting for commands from Mac...")

    def run(self):
        """Main loop: receive commands, check safety, move dog."""
        try:
            while self.running:
                # ── Read ultrasonic distance ─────────────────────────
                distance = round(self.dog.read_distance(), 2)

                # ── Receive command from Mac ─────────────────────────
                action = self._receive_command()

                # ── Safety check ─────────────────────────────────────
                if 0 < distance < DANGER_DISTANCE and action == "forward":
                    if not self.danger:
                        print(f"\033[0;31m[RC] DANGER! {distance}cm — blocking forward\033[m")
                        self.dog.body_stop()
                        self.dog.rgb_strip.set_mode("bark", "red", bps=2)
                        self.dog.legs_move([self.stand_angles], speed=70)
                        self.dog.wait_all_done()
                        self.danger = True
                        # Bark warning
                        if bark_action:
                            try:
                                head_yaw = self.dog.head_current_angles[0]
                                bark_action(self.dog, [head_yaw, 0, 0])
                            except Exception:
                                pass
                    action = "stop"  # override forward with stop
                else:
                    if self.danger:
                        print("\033[0;32m[RC] Clear — danger resolved\033[m")
                        self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
                        self.danger = False

                # ── Execute action ───────────────────────────────────
                self._execute(action)

                # ── Send telemetry ───────────────────────────────────
                now = time.time()
                if now - self.last_telem_time >= 0.2:
                    self._send_telemetry(distance)
                    self.last_telem_time = now

                time.sleep(0.08)

        except KeyboardInterrupt:
            print("\n[RC] Ctrl+C — shutting down...")
        finally:
            self._shutdown()

    def _receive_command(self):
        """Read latest command from ZMQ, return action string."""
        try:
            raw = self.cmd_sock.recv()
            msg = json.loads(raw.decode("utf-8"))
            return msg.get("action", "stop")
        except zmq.Again:
            # No message — keep current action
            return self.current_action
        except Exception as e:
            print(f"[RC] CMD parse error: {e}")
            return "stop"

    def _execute(self, action):
        """Execute an action on the PiDog. Only issues new commands on change."""
        if action == self.current_action:
            # If dog is doing a stepping action and it's done, re-issue
            if action in self.ACTION_MAP and self.dog.is_legs_done():
                sdk_name, steps, speed = self.ACTION_MAP[action]
                self.dog.do_action(sdk_name, step_count=steps, speed=speed)
            return

        # ── Action changed ───────────────────────────────────────────
        self.current_action = action

        if action == "stop":
            self.dog.body_stop()
            self.dog.rgb_strip.set_mode("breath", "blue", bps=0.5)
            print("[RC] → STOP")

        elif action in self.ACTION_MAP:
            sdk_name, steps, speed = self.ACTION_MAP[action]
            self.dog.do_action(sdk_name, step_count=steps, speed=speed)
            self.dog.do_action("wag_tail", step_count=5, speed=99)
            self.dog.rgb_strip.set_mode("breath", "green", bps=1)
            print(f"[RC] → {action.upper()}")

        elif action == "bark":
            self.dog.body_stop()
            if bark_action:
                try:
                    head_yaw = self.dog.head_current_angles[0]
                    bark_action(self.dog, [head_yaw, 0, 0])
                except Exception:
                    pass
            self.current_action = "stop"

        elif action == "stand":
            self.dog.body_stop()
            self.dog.do_action("stand", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print("[RC] → STAND")

        elif action == "sit":
            self.dog.body_stop()
            self.dog.do_action("sit", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print("[RC] → SIT")

        elif action == "lie":
            self.dog.body_stop()
            self.dog.do_action("lie", speed=80)
            self.dog.wait_all_done()
            self.current_action = "stop"
            print("[RC] → LIE DOWN")

    def _send_telemetry(self, distance):
        """Publish telemetry data for the Mac TUI."""
        try:
            battery = self.dog.get_battery_voltage()
        except Exception:
            battery = 0.0

        telem = {
            "distance_cm": distance,
            "battery_v": battery,
            "action": self.current_action,
            "danger": self.danger,
            "timestamp": time.time(),
        }
        try:
            self.telem_sock.send_string(json.dumps(telem))
        except Exception:
            pass

    def _shutdown(self):
        """Clean shutdown: stop dog, kill sensor process, close ZMQ."""
        print("[RC] Shutting down PiDog...")
        try:
            self.dog.body_stop()
            self.dog.do_action("lie", speed=80)
            self.dog.wait_all_done()
        except Exception:
            pass

        # Kill ultrasonic sensor process (frees GPIO)
        sp = getattr(self.dog, "sensory_process", None)
        if sp is not None and sp.is_alive():
            sp.terminate()
            sp.join(timeout=2.0)
            if sp.is_alive():
                try:
                    os.kill(sp.pid, signal.SIGKILL)
                except Exception:
                    pass

        try:
            self.dog.close()
        except SystemExit:
            pass

        self.cmd_sock.close()
        self.telem_sock.close()
        self.ctx.term()
        print("[RC] Done.")


if __name__ == "__main__":
    controller = RemoteController()
    controller.run()
