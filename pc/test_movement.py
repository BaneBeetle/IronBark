"""
pc/test_movement.py — Movement Debug Tool

Sends the EXACT same commands as follower.py, one tap at a time.
Each keypress sends one command and waits for it to finish.
Compare this to what follower.py claims to send.

Controls:
  W = forward  (step_count=8, speed=98, head_mode=remote)
  A = turn_left  (step_count=8, speed=98, head_mode=remote)
  S = backward (step_count=8, speed=80, head_mode=remote)
  D = turn_right (step_count=8, speed=98, head_mode=remote)
  SPACE = stop
  1 = forward with head_mode=local (like follower does)
  2 = turn_left with head_mode=local
  3 = turn_right with head_mode=local
  Q = quit

Run: .venv/bin/python pc/test_movement.py
"""

import sys
import json
import curses
import time
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

CMD_PORT = config.CMD_PORT
PC_IP = config.PC_IP

FWD_STEPS = getattr(config, "FORWARD_STEP_COUNT", 8)
TURN_STEPS = getattr(config, "BODY_TURN_STEP_COUNT", 8)
FWD_SPEED = 98
TURN_SPEED = 98
BACK_SPEED = 80


def send_cmd(sock, action, speed, step_count, head_mode="remote",
             head_yaw=0, head_pitch=15):
    cmd = {
        "action": action,
        "speed": speed,
        "step_count": step_count,
        "head_mode": head_mode,
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "bark": False,
    }
    sock.send_string(json.dumps(cmd))
    return cmd


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)  # blocking — waits for keypress

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{PC_IP}:{CMD_PORT}")
    time.sleep(0.3)  # let ZMQ connect

    history = []
    cmd_count = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 0, "═══ IronBark Movement Tester ═══",
                      curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(2, 0, "Sends EXACT same commands as follower.py",
                      curses.color_pair(2))
        stdscr.addstr(3, 0, f"FWD steps={FWD_STEPS} spd={FWD_SPEED}  "
                      f"TURN steps={TURN_STEPS} spd={TURN_SPEED}")

        stdscr.addstr(5, 0, "── Remote mode (head stays centered) ──")
        stdscr.addstr(6, 0, "  W = Forward    A = Turn Left",
                      curses.color_pair(1))
        stdscr.addstr(7, 0, "  S = Backward   D = Turn Right",
                      curses.color_pair(1))
        stdscr.addstr(8, 0, "  SPACE = Stop", curses.color_pair(4))

        stdscr.addstr(10, 0, "── Local mode (head tracks bbox — like follower) ──")
        stdscr.addstr(11, 0, "  1 = Forward    2 = Turn Left    3 = Turn Right",
                      curses.color_pair(2))

        stdscr.addstr(13, 0, "  Q = Quit")

        stdscr.addstr(15, 0, f"Commands sent: {cmd_count}",
                      curses.color_pair(3))

        # Show last 10 commands
        start_row = 17
        stdscr.addstr(start_row - 1, 0, "── History ──")
        for i, entry in enumerate(history[-10:]):
            row = start_row + i
            if row >= h - 1:
                break
            color = curses.color_pair(1) if "forward" in entry else \
                    curses.color_pair(2) if "turn" in entry else \
                    curses.color_pair(4)
            stdscr.addstr(row, 0, entry[:w-1], color)

        stdscr.refresh()
        key = stdscr.getch()

        cmd = None
        label = ""

        if key == ord('q') or key == ord('Q') or key == 27:
            send_cmd(sock, "stop", 0, 0)
            break

        elif key == ord('w') or key == ord('W'):
            cmd = send_cmd(sock, "forward", FWD_SPEED, FWD_STEPS, "remote")
            label = f"FORWARD  steps={FWD_STEPS} spd={FWD_SPEED} head=remote"

        elif key == ord('a') or key == ord('A'):
            cmd = send_cmd(sock, "turn_left", TURN_SPEED, TURN_STEPS, "remote")
            label = f"TURN_L   steps={TURN_STEPS} spd={TURN_SPEED} head=remote"

        elif key == ord('s') or key == ord('S'):
            cmd = send_cmd(sock, "backward", BACK_SPEED, FWD_STEPS, "remote")
            label = f"BACKWARD steps={FWD_STEPS} spd={BACK_SPEED} head=remote"

        elif key == ord('d') or key == ord('D'):
            cmd = send_cmd(sock, "turn_right", TURN_SPEED, TURN_STEPS, "remote")
            label = f"TURN_R   steps={TURN_STEPS} spd={TURN_SPEED} head=remote"

        elif key == ord(' '):
            cmd = send_cmd(sock, "stop", 0, 0)
            label = "STOP"

        elif key == ord('1'):
            cmd = send_cmd(sock, "forward", FWD_SPEED, FWD_STEPS, "local")
            label = f"FORWARD  steps={FWD_STEPS} spd={FWD_SPEED} head=LOCAL"

        elif key == ord('2'):
            cmd = send_cmd(sock, "turn_left", TURN_SPEED, TURN_STEPS, "local")
            label = f"TURN_L   steps={TURN_STEPS} spd={TURN_SPEED} head=LOCAL"

        elif key == ord('3'):
            cmd = send_cmd(sock, "turn_right", TURN_SPEED, TURN_STEPS, "local")
            label = f"TURN_R   steps={TURN_STEPS} spd={TURN_SPEED} head=LOCAL"

        if cmd is not None:
            cmd_count += 1
            history.append(f"#{cmd_count:3d} {label}")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    curses.wrapper(main)
