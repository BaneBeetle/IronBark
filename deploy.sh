#!/bin/bash
# deploy.sh — Push IronBark code to the Raspberry Pi from Mac
# motor_controller.py is the unified controller (handles both follow-me + teleop).
# Run on Pi with: sudo python3 motor_controller.py
#
# Reads PI_USER, PI_IP, and PI_DEPLOY_PATH from .env (gitignored).
# Copy .env.example to .env and fill in your values before running.
set -e

# Source .env if it exists
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${PI_USER:?PI_USER not set — copy .env.example to .env and fill in values}"
: "${PI_IP:?PI_IP not set — copy .env.example to .env and fill in values}"

PI_TARGET="${PI_USER}@${PI_IP}"
REMOTE="${PI_DEPLOY_PATH:-~/ironbark}"

echo "Deploying IronBark to ${PI_TARGET}:${REMOTE}..."
ssh "$PI_TARGET" "mkdir -p $REMOTE"
scp config.py .env "${PI_TARGET}:${REMOTE}/"
scp pi/pi_sender.py pi/motor_controller.py pi/remote_control.py pi/lidar_reader.py pi/test_lidar.py "${PI_TARGET}:${REMOTE}/"

echo "Verifying..."
ssh "$PI_TARGET" "ls -la $REMOTE/"
echo "Deploy complete!"
