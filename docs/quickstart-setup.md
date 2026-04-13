# Quickstart: IronBark Setup Guide

Get IronBark running from scratch. By the end of this guide, the robot dog will follow you around the room.

## Prerequisites

### Hardware
- SunFounder PiDog robot dog kit (assembled, servos calibrated)
- Raspberry Pi 5 (8 GB recommended) mounted on the PiDog
- USB UVC webcam mounted on the dog's head servo
- OV5647 CSI ribbon camera on the nose (optional, for EXPLORE mode)
- Mac with Apple Silicon (M1 or later)
- Both devices on the same network (WiFi or Tailscale)

### Software — Mac
- Python 3.10+
- Ollama (for VLM inference)
- Git

### Software — Pi
- Raspberry Pi OS (64-bit, Bookworm or later)
- Python 3.11+ (ships with Pi OS)
- picamera2 (`sudo apt install python3-picamera2`)

## Step 1: Clone and Set Up the Mac

```bash
git clone <your-repo-url> ~/Projects/ironbark
cd ~/Projects/ironbark

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install ultralytics insightface onnxruntime opencv-python pyzmq numpy requests
```

## Step 2: Install and Pull Ollama Models

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start the Ollama server
ollama serve &

# Pull the VLM model
ollama pull moondream
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

You should see `moondream` in the model list.

## Step 3: Configure Networking

```bash
cd ~/Projects/ironbark
cp .env.example .env
```

Edit `.env` with your network IPs:

```bash
PC_IP=<your-mac-ip>         # Mac's IP (or Tailscale IP)
PI_IP=<your-pi-ip>          # Pi's IP (or Tailscale IP)
PI_USER=<pi-username>       # SSH username on the Pi
PI_DEPLOY_PATH=ironbark     # Relative to ~/ on Pi (do NOT use ~/ironbark)
```

> **Warning:** Do not use `~/ironbark` for `PI_DEPLOY_PATH`. Bash expands `~` to the *local* home directory during variable assignment, so `deploy.sh` would create `/Users/you/ironbark` on the Pi instead of `/home/pi/ironbark`. Use the relative path `ironbark`.

### Tailscale (Recommended)

Tailscale provides stable IPs that work across any WiFi network:

```bash
# On Mac
brew install tailscale
sudo tailscale up

# On Pi
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Use the Tailscale IPs (starting with `100.`) in your `.env` file.

## Step 4: Deploy Code to the Pi

```bash
cd ~/Projects/ironbark
./deploy.sh
```

This copies `config.py`, `.env`, and all `pi/*.py` files to the Pi. Verify:

```bash
ssh <pi-user>@<pi-ip> 'ls ~/ironbark/'
```

You should see: `config.py`, `.env`, `pi_sender.py`, `motor_controller.py`, `remote_control.py`.

## Step 5: Enroll Your Face (Multi-Distance)

The dog needs to know what you look like from different distances. Enrollment captures your face at 3 distances from the ground-level camera so recognition works at operating range (not just close-up).

**Run enrollment each session** — future body ReID will also capture clothing appearance, which changes day to day.

Start the Pi webcam stream first (see Step 6, Terminal 2), then on the Mac:

```bash
cd ~/Projects/ironbark
.venv/bin/python pc/enroll_owner.py
```

1. Place the camera at its mounted position (on the dog's head, ~6 inches off the ground).
2. Press **SPACE**. The tool shows: **"Stage 1/3: Move to CLOSE (2 ft)"**.
3. Stand ~2 ft from the camera. Press **SPACE** to begin capture.
4. **Turn your head only** left and right (body stays facing camera). 10 samples are captured automatically.
5. The tool advances: **"Stage 2/3: Move to MEDIUM (4 ft)"**. Step back and press **SPACE**.
6. Again, turn your head only left and right. 10 samples captured.
7. The tool advances: **"Stage 3/3: Move to FAR (6-8 ft)"**. Step back and press **SPACE**.
8. At this distance, **slight body turns (~15 deg)** are fine in addition to head turns.
9. After all 30 samples, the gallery is saved to `data/owner_gallery.npy`.

> **Why multi-distance?** The camera is at ground level looking up. A face captured at 2 ft looks very different from one at 6 ft (angle, resolution, scale). Storing all embeddings as a gallery — rather than averaging into one — means the system finds the best match regardless of your current distance.

## Step 6: Start the System

Open four terminal windows.

### Terminal 1 — Pi: Motor Controller (requires sudo for speaker)

```bash
ssh -t <pi-user>@<pi-ip> 'cd ~/ironbark && sudo python3 motor_controller.py'
```

Wait for `[Motor] Ready. Waiting for commands...` before proceeding.

### Terminal 2 — Pi: USB Webcam Stream

```bash
ssh <pi-user>@<pi-ip> 'cd ~/ironbark && python3 -u pi_sender.py --source usb --port 50505'
```

You should see `[Camera] USB webcam opened at 640x480` and `[Sender] Streaming at <=30fps`.

### Terminal 3 — Pi: Ribbon Camera Stream (optional)

```bash
ssh <pi-user>@<pi-ip> 'cd ~/ironbark && python3 -u pi_sender.py --source picamera --port 50506'
```

Skip this if you don't have a ribbon camera. The system falls back to the webcam for EXPLORE queries.

### Terminal 4 — Mac: Follower (brain)

```bash
cd ~/Projects/ironbark
.venv/bin/python pc/follower.py
```

An OpenCV window opens showing the webcam feed with YOLO bounding boxes, face recognition labels, and behavior mode.

## What You Should See

| Event | Dog Behavior | Log Output |
|-------|-------------|------------|
| Owner visible | Dog walks toward you | `[FOLLOW] area=0.120 dist=65cm off_x=30` |
| Owner close (area > 0.40) | Dog stops, barks, sits | `[Follower] Arrival! area=0.420 mode=ACTIVE` |
| Owner leaves frame | Head sweeps left-right | `[Follower] FOLLOW -> SEARCH` |
| 5 seconds without owner | Dog moves toward doorways | `[Follower] SEARCH -> EXPLORE` |
| Owner returns | Dog re-acquires and follows | `[Follower] EXPLORE -> FOLLOW` |

The VLM logs behavior mode changes:

```
[VLM] Situation: MODE=ACTIVE (243ms) | yolo=1 | Person standing in a living room
[VLM] Situation: MODE=GENTLE (251ms) | yolo=1 | Person sitting on a couch
```

## Shutdown

1. Press `q` in the OpenCV window or Ctrl+C in the Mac terminal.
2. Stop camera streams: Ctrl+C in Terminals 2/3 (or `run_cameras.sh stop`).
3. Ctrl+C in Terminal 1 (motor controller lies down and releases GPIO).

## Troubleshooting

### Dog doesn't bark (visual bark works but no audio)

`motor_controller.py` must run with `sudo`. The PiDog `speak()` call requires root access to the audio hardware. Without it, the call silently fails.

### "GPIO busy" on motor_controller startup

A previous motor_controller process was killed without cleanup. The built-in orphan cleanup runs automatically, but if it persists:

```bash
ssh <pi-user>@<pi-ip> 'sudo pkill -9 -f pidog; sleep 1'
```

Then restart motor_controller.

### No frames received on Mac

Check that `PC_IP` in `.env` matches the Mac's actual IP. If using Tailscale, verify both nodes are online: `tailscale status`.

### VLM hallucinating (wheelchair, hospital, etc.)

Verify `config.py` has `VLM_MODEL = "moondream"`. If using `llava:7b`, hallucinations are expected on empty rooms with white furniture.

### YOLO not using GPU

Check MPS availability:

```python
import torch
print(torch.backends.mps.is_available())  # Should be True on Apple Silicon
```

If False, ensure you have a recent PyTorch version (`pip install --upgrade torch`).
