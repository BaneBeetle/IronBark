#!/bin/bash
# run_cameras.sh — Start both IronBark camera streams on the Pi.
#
# Launches two pi_sender.py instances in the background:
#   • USB webcam  → ZMQ_PORT     (owner tracking, head-mounted, pitched up)
#   • Ribbon cam  → ZMQ_NAV_PORT (navigation,    nose-mounted,  forward)
#
# Usage:
#   ./run_cameras.sh            # start both, log to /tmp/pi_sender_*.log
#   ./run_cameras.sh stop       # stop both
#   ./run_cameras.sh status     # show running instances
#
# The Mac-side follower.py binds to both ports and automatically forwards
# the ribbon-cam frames to the VLM during SEARCH/EXPLORE so the dog sees
# the floor and doorways instead of the ceiling.
set -e

# Work from the directory containing this script. On the Pi, deploy.sh
# flattens pi/* into ironbark/, so both run_cameras.sh and pi_sender.py
# end up in ~/ironbark. In the local repo they both live in ironbark/pi/.
cd "$(dirname "$0")"

WEBCAM_LOG=/tmp/pi_sender_webcam.log
RIBBON_LOG=/tmp/pi_sender_ribbon.log

case "${1:-start}" in
  start)
    # Kill any existing pi_senders first (idempotent restart)
    pkill -f "pi_sender.py" 2>/dev/null || true
    sleep 0.3

    echo "[run_cameras] Starting USB webcam on port 50505 → $WEBCAM_LOG"
    nohup python3 -u pi_sender.py --source usb --port 50505 \
      > "$WEBCAM_LOG" 2>&1 &

    echo "[run_cameras] Starting ribbon cam on port 50506 → $RIBBON_LOG"
    nohup python3 -u pi_sender.py --source picamera --port 50506 \
      > "$RIBBON_LOG" 2>&1 &

    sleep 3
    echo ""
    echo "[run_cameras] Status:"
    pgrep -af pi_sender.py || echo "  (none running — check logs)"
    echo ""
    echo "[run_cameras] Tail webcam log:"
    tail -5 "$WEBCAM_LOG" 2>/dev/null || true
    echo ""
    echo "[run_cameras] Tail ribbon log:"
    tail -5 "$RIBBON_LOG" 2>/dev/null || true
    ;;

  stop)
    pkill -f "pi_sender.py" 2>/dev/null && echo "[run_cameras] Stopped."\
      || echo "[run_cameras] No pi_sender running."
    ;;

  status)
    pgrep -af pi_sender.py || echo "[run_cameras] No pi_sender running."
    ;;

  *)
    echo "Usage: $0 [start|stop|status]"
    exit 1
    ;;
esac
