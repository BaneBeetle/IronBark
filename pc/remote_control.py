"""
pc/remote_control.py — IronBark WASD Teleoperation (Mac Side)
Captures keyboard input and sends movement commands to Pi over ZMQ.
Run on Mac: python pc/remote_control.py
"""

import sys
import time
import json
import curses
import threading
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Ports ────────────────────────────────────────────────────────────────────
CMD_PORT = getattr(config, "REMOTE_CMD_PORT", 5557)
TELEM_PORT = getattr(config, "REMOTE_TELEM_PORT", 5558)
PI_IP = config.PI_IP

# ── Key mapping ──────────────────────────────────────────────────────────────
KEY_MAP = {
    ord("w"): "forward",
    ord("W"): "forward",
    ord("s"): "backward",
    ord("S"): "backward",
    ord("a"): "turn_left",
    ord("A"): "turn_left",
    ord("d"): "turn_right",
    ord("D"): "turn_right",
    ord(" "): "bark",
    ord("1"): "stand",
    ord("2"): "sit",
    ord("3"): "lie",
}


class TelemetryReceiver:
    """Background thread that receives telemetry from Pi."""

    def __init__(self, ctx, pi_ip, telem_port):
        self.data = {
            "distance_cm": -1,
            "battery_v": 0.0,
            "action": "stop",
            "danger": False,
            "timestamp": 0,
        }
        self.lock = threading.Lock()
        self.running = True

        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.connect(f"tcp://{pi_ip}:{telem_port}")
        self.sock.subscribe(b"")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

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

    def get(self):
        with self.lock:
            return dict(self.data)

    def stop(self):
        self.running = False
        self.sock.close()


def main(stdscr):
    # ── Curses setup ─────────────────────────────────────────────────────
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(80)  # ~12.5 Hz refresh

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # active movement
    curses.init_pair(2, curses.COLOR_RED, -1)      # danger
    curses.init_pair(3, curses.COLOR_CYAN, -1)     # info
    curses.init_pair(4, curses.COLOR_YELLOW, -1)   # heading
    curses.init_pair(5, curses.COLOR_WHITE, -1)    # normal

    # ── ZMQ setup ────────────────────────────────────────────────────────
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PUB)
    cmd_sock.connect(f"tcp://{PI_IP}:{CMD_PORT}")

    telem = TelemetryReceiver(ctx, PI_IP, TELEM_PORT)

    # Give ZMQ a moment to establish connection
    time.sleep(0.3)

    current_action = "stop"
    last_key_time = time.time()
    send_count = 0

    try:
        while True:
            # ── Read key ─────────────────────────────────────────────
            key = stdscr.getch()

            if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                cmd_sock.send_string(json.dumps({"action": "stop"}))
                break

            if key in KEY_MAP:
                action = KEY_MAP[key]
                last_key_time = time.time()
            else:
                # No WASD key pressed — if no key for >150ms, stop
                if time.time() - last_key_time > 0.15:
                    action = "stop"
                else:
                    action = current_action

            # ── Send command ─────────────────────────────────────────
            cmd = json.dumps({"action": action})
            cmd_sock.send_string(cmd)
            current_action = action
            send_count += 1

            # ── Draw TUI ─────────────────────────────────────────────
            t = telem.get()
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # Title
            title = "══════ IronBark Remote Control ══════"
            stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.color_pair(4) | curses.A_BOLD)

            # Controls help
            stdscr.addstr(2, 2, "Controls:", curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(3, 4, "W", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(3, 5, " = Forward", curses.color_pair(5))
            stdscr.addstr(3, 18, "A", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(3, 19, " = Turn Left", curses.color_pair(5))
            stdscr.addstr(4, 4, "S", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(4, 5, " = Backward", curses.color_pair(5))
            stdscr.addstr(4, 18, "D", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(4, 19, " = Turn Right", curses.color_pair(5))
            stdscr.addstr(5, 4, "Space", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(5, 9, " = Bark", curses.color_pair(5))
            stdscr.addstr(5, 18, "1/2/3", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(5, 23, " = Stand/Sit/Lie", curses.color_pair(5))
            stdscr.addstr(6, 4, "Q/ESC", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(6, 9, " = Quit", curses.color_pair(5))

            # Separator
            stdscr.addstr(8, 2, "─" * min(50, w - 4), curses.color_pair(5))

            # Current action (big display)
            action_display = current_action.upper().replace("_", " ")
            if current_action == "stop":
                action_color = curses.color_pair(5)
            elif t.get("danger", False):
                action_color = curses.color_pair(2) | curses.A_BOLD
                action_display = "⚠ DANGER STOP ⚠"
            else:
                action_color = curses.color_pair(1) | curses.A_BOLD

            stdscr.addstr(10, 2, "Action: ", curses.color_pair(3))
            stdscr.addstr(10, 10, f"  {action_display}  ", action_color)

            # WASD visual
            wa = "▲" if current_action == "forward" else "△"
            sa = "▼" if current_action == "backward" else "▽"
            aa = "◀" if current_action == "turn_left" else "◁"
            da = "▶" if current_action == "turn_right" else "▷"

            col = max(0, w // 2 - 2)
            stdscr.addstr(12, col, wa, curses.color_pair(1) if current_action == "forward" else curses.color_pair(5))
            stdscr.addstr(13, col - 2, aa, curses.color_pair(1) if current_action == "turn_left" else curses.color_pair(5))
            stdscr.addstr(13, col, sa, curses.color_pair(1) if current_action == "backward" else curses.color_pair(5))
            stdscr.addstr(13, col + 2, da, curses.color_pair(1) if current_action == "turn_right" else curses.color_pair(5))

            # Telemetry
            stdscr.addstr(15, 2, "─" * min(50, w - 4), curses.color_pair(5))
            stdscr.addstr(16, 2, "Telemetry:", curses.color_pair(4) | curses.A_BOLD)

            dist = t.get("distance_cm", -1)
            if dist < 0:
                dist_str = "N/A"
                dist_color = curses.color_pair(5)
            elif dist < 15:
                dist_str = f"{dist:.1f} cm  ⚠ DANGER"
                dist_color = curses.color_pair(2) | curses.A_BOLD
            elif dist < 50:
                dist_str = f"{dist:.1f} cm"
                dist_color = curses.color_pair(4)
            else:
                dist_str = f"{dist:.1f} cm"
                dist_color = curses.color_pair(1)

            battery = t.get("battery_v", 0)
            if battery < 6.5 and battery > 0:
                bat_color = curses.color_pair(2)
            else:
                bat_color = curses.color_pair(1)

            pi_action = t.get("action", "?")

            stdscr.addstr(17, 4, f"Distance:  ", curses.color_pair(3))
            stdscr.addstr(17, 15, dist_str, dist_color)
            stdscr.addstr(18, 4, f"Battery:   ", curses.color_pair(3))
            stdscr.addstr(18, 15, f"{battery:.2f}V", bat_color)
            stdscr.addstr(19, 4, f"Pi state:  ", curses.color_pair(3))
            stdscr.addstr(19, 15, pi_action.upper(), curses.color_pair(5))
            stdscr.addstr(20, 4, f"Cmds sent: ", curses.color_pair(3))
            stdscr.addstr(20, 15, str(send_count), curses.color_pair(5))

            # Connection status
            age = time.time() - t.get("timestamp", 0)
            if t.get("timestamp", 0) == 0:
                conn_str = "NO CONNECTION"
                conn_color = curses.color_pair(2)
            elif age < 1.0:
                conn_str = "CONNECTED"
                conn_color = curses.color_pair(1)
            else:
                conn_str = f"STALE ({age:.0f}s)"
                conn_color = curses.color_pair(4)

            stdscr.addstr(21, 4, f"Link:      ", curses.color_pair(3))
            stdscr.addstr(21, 15, conn_str, conn_color)

            stdscr.refresh()

    finally:
        # Send stop before exiting
        cmd_sock.send_string(json.dumps({"action": "stop"}))
        time.sleep(0.1)
        telem.stop()
        cmd_sock.close()
        ctx.term()


if __name__ == "__main__":
    print(f"Connecting to PiDog at {PI_IP}...")
    print(f"  CMD  → tcp://{PI_IP}:{CMD_PORT}")
    print(f"  TELEM ← tcp://{PI_IP}:{TELEM_PORT}")
    print("Starting TUI... (press Q to quit)")
    time.sleep(0.5)
    curses.wrapper(main)
