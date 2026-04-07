"""
pi/read_distance.py — Continuously print ultrasonic distance.
Run on Pi with: sudo python3 read_distance.py
"""

import os
import sys
import signal
import time
import atexit
import subprocess

# Kill orphaned PiDog sensor processes from previous crash
try:
    result = subprocess.run(
        ["pgrep", "-f", "pidog.*sensory|sensory.*pidog"],
        capture_output=True, text=True, timeout=3
    )
    my_pid = str(os.getpid())
    for pid in result.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    time.sleep(0.3)
except Exception:
    pass

from pidog import Pidog

dog = Pidog()


def cleanup(*args):
    """Release GPIO on any exit."""
    try:
        sp = getattr(dog, "sensory_process", None)
        if sp is not None and sp.is_alive():
            sp.terminate()
            sp.join(timeout=2.0)
            if sp.is_alive():
                os.kill(sp.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        dog.close()
    except Exception:
        pass


signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))
atexit.register(cleanup)

dog.do_action("stand", speed=80)
dog.wait_all_done()
dog.head_move([[0, 0, 35]], speed=80)  # look up, same as greeting pitch
dog.wait_all_done()

try:
    while True:
        dist = dog.read_distance()
        print(f"{dist:.1f} cm")
        time.sleep(0.2)
except KeyboardInterrupt:
    cleanup()
