#!/bin/bash
# deploy.sh — Push IronBark code to the Raspberry Pi from Mac
PI="banebeetle@raspberrypi"
REMOTE="~/ironbark"

echo "Deploying IronBark to Pi..."
ssh $PI "mkdir -p $REMOTE"
scp config.py "${PI}:${REMOTE}/"
scp pi/pi_sender.py pi/motor_controller.py "${PI}:${REMOTE}/"

echo "Verifying..."
ssh $PI "ls -la $REMOTE/"
echo "Deploy complete!"
