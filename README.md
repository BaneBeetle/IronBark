# IronBark

> An autonomous follow-me robot dog powered by distributed vision-language inference.

I moved to NYC for school and couldn't bring my dogs with me. So I built one.

IronBark is a [SunFounder PiDog](https://github.com/sunfounder/pidog) that
recognizes its owner's face, follows them around the room, greets them on
arrival, and reads the room with a vision-language model to decide *how* it
should behave. The Raspberry Pi runs the body. A MacBook runs the brain.
They talk over ZeroMQ on the local network (or any network, via Tailscale).

---

## Architecture

```
┌──────────────────────────┐                  ┌──────────────────────────┐
│  Raspberry Pi 5          │                  │  MacBook (Apple Silicon) │
│  ──────────────          │                  │  ──────────────────────  │
│  USB / ribbon camera     │ ─ JPEG @ 30fps ─→│  YOLO11n person detect   │
│  PiDog actuators         │   ZMQ :50505     │  InsightFace ArcFace     │
│  Ultrasonic sensor       │                  │  LLaVA 7B (situation)    │
│  Head-yaw bbox tracker   │ ← JSON cmds ─────│  Behavior FSM            │
│  Speaker + RGB strip     │   ZMQ :5556      │                          │
└──────────────────────────┘                  └──────────────────────────┘
        (the body)                                    (the brain)
```

The Pi captures and JPEG-encodes frames, ships them to the Mac over a ZMQ
PUSH socket, and receives back a stream of JSON commands at 10Hz. The Mac
runs YOLO + ArcFace at every frame (the *fast path*), and a vision-language
model in a background thread at ~0.4Hz (the *slow path*). End-to-end
latency from camera capture to motor execution is **~60-75ms**.

---

## Features

### Follow-me with robust face recognition
- **YOLO11n** person detection on Apple Silicon (MPS backend)
- **InsightFace ArcFace** owner recognition
- **Multi-shot enrollment**: capture 25 face samples over ~5 seconds with pose
  variety, L2-normalize each, average, and re-normalize. Vastly more reliable
  than single-frame enrollment.
- **IoU-based temporal smoothing** of face matches across frames eliminates
  per-frame flicker between owner and stranger labels.

### Situation-aware behavior modes
- The vision-language model reads the scene every ~2.5 seconds and selects
  a behavior mode that adjusts how the dog acts:

  | Mode    | Speed | Arrival | Bark           | Idle pose |
  |---------|-------|---------|----------------|-----------|
  | ACTIVE  | 98    | 0.40    | Loud (80)      | stand     |
  | PLAYFUL | 98    | 0.45    | Loud (80)      | stand     |
  | GENTLE  | 60    | 0.30    | Quiet (40)     | sit       |
  | SOCIAL  | 70    | 0.35    | Off            | stand     |
  | CALM    | 50    | 0.25    | Off            | lie       |

- Mode changes use **2-consecutive-same hysteresis** so the dog doesn't
  flicker between modes when the VLM disagrees with itself.

### Semantic exploration
- When the owner is lost, the VLM directs the dog toward doorways and open
  spaces instead of doing a blind head sweep. The dog plays a "thinking"
  animation (purple breath + slow tail wag) while the VLM is processing.

### Camera-based arrival + head-only bark
- Bbox area ratio acts as a distance proxy — no separate depth sensor.
- When the dog arrives, it re-centers its body to face the owner, looks up,
  and barks. The bark is **head-only** (no leg movement) and uses a hold
  guard to prevent it from lurching mid-bark when the head pitch shifts the
  bbox area ratio.

### Reliability
- GPIO zombie cleanup before PiDog init (handles Pi restart after `kill -9`)
- Graceful SIGTERM + atexit shutdown
- ZMQ `CONFLATE=1` on all SUB sockets — always reads the latest message,
  no stale-frame buildup
- Pi-side head tracker: head yaw/pitch computed locally from bbox in ~5ms
  instead of round-tripping through the Mac (~50ms)

### Portable networking
- All network IPs are loaded from a local `.env` file (gitignored). Works
  out of the box with [Tailscale](https://tailscale.com/) — same code, same
  IPs, regardless of whether you're at home, at school, or on a phone hotspot.

---

## Hardware

- **SunFounder PiDog kit** (the body)
- **Raspberry Pi 5** (4GB or 8GB)
- **microSD** (32GB+)
- **USB webcam** (I use a Toucan; the included Pi ribbon camera also works)
- **MacBook with Apple Silicon** (the brain — anything with MPS)

## Software

| Component                            | Where     |
|--------------------------------------|-----------|
| Python 3.11+                         | Mac & Pi  |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | Mac |
| [InsightFace](https://github.com/deepinsight/insightface) + ONNXRuntime | Mac |
| [Ollama](https://ollama.ai/) + `llava:7b` | Mac |
| OpenCV, NumPy, pyzmq                 | Mac & Pi  |
| picamera2                            | Pi        |
| [pidog (SunFounder SDK)](https://github.com/sunfounder/pidog) | Pi |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/BaneBeetle/IronBark.git
cd IronBark
```

### 2. Configure your network

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
PC_IP=<your-mac-ip>      # Tailscale IP from `tailscale ip -4`, or LAN IP
PI_IP=<your-pi-ip>       # Same — get from the Pi
PI_USER=<pi-username>    # SSH username for the Pi
PI_DEPLOY_PATH=~/ironbark
```

### 3. Install dependencies

**Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics insightface onnxruntime opencv-python pyzmq numpy
brew install ollama
ollama serve &        # leave this running in a separate terminal
ollama pull llava:7b
```

**Pi** (after `ssh PI_USER@PI_IP`):

```bash
pip3 install pyzmq numpy opencv-python-headless
# picamera2 and pidog are installed by the SunFounder kit setup
```

### 4. Deploy code to the Pi

```bash
./deploy.sh
```

This SCPs `config.py`, `.env`, and the `pi/*.py` files to `~/ironbark/` on
the Pi. The Pi reads its own copy of `.env` to know where the Mac is.

### 5. Enroll your face

On the Pi, start the camera stream:

```bash
cd ~/ironbark
python3 pi_sender.py --device 8       # --device 8 forces Toucan USB; omit for ribbon cam
```

On the Mac:

```bash
python pc/enroll_owner.py
```

Stand in frame, press SPACE, and slowly turn your head left → center →
right → up → down over ~5 seconds. The script captures 25 samples, averages
the L2-normalized embeddings, and saves the result to
`data/owner_embedding.npy`.

### 6. Run the system

On the Pi (in addition to `pi_sender.py`):

```bash
sudo python3 motor_controller.py
```

On the Mac:

```bash
python pc/follower.py
```

The dog should boot, find you, and start following.

---

## Manual control (teleoperation)

```bash
# pi/motor_controller.py already handles teleop on port 5557 — no extra Pi process needed

# On Mac:
python pc/remote_control.py
```

WASD to drive, space to bark, `q` to quit. Curses TUI shows live ultrasonic
distance and battery telemetry from the Pi.

---

## Project layout

```
IronBark/
├── config.py                  # Loads .env, defines all constants
├── .env.example               # Network template (copy to .env)
├── deploy.sh                  # SCP code + .env to Pi
├── pi/                        # Code that runs on the Raspberry Pi
│   ├── pi_sender.py             # Camera capture → ZMQ stream
│   ├── motor_controller.py      # Unified motor controller (follow + teleop)
│   ├── remote_control.py        # Legacy Pi-side teleop
│   └── read_distance.py         # Ultrasonic debug script
└── pc/                        # Code that runs on the Mac
    ├── follower.py              # Behavior FSM (the brain)
    ├── perception_pipeline.py   # YOLO + ArcFace + VLM orchestration
    ├── enroll_owner.py          # Multi-shot face enrollment
    ├── bbox_calibrate.py        # Calibration: bbox area ratio at distance
    ├── remote_control.py        # WASD curses teleop
    └── perception/
        ├── yolo_detector.py     # YOLO11 wrapper
        ├── face_recognizer.py   # InsightFace ArcFace
        └── vlm_reasoner.py      # LLaVA situation + explore queries
```

---

## Status

- [x] Phase 1 — Video streaming (Pi → Mac, ZMQ + JPEG)
- [x] Phase 2 — Perception pipeline (YOLO + ArcFace + VLM)
- [x] Phase 3 — Follow-me behavior FSM
- [x] Phase 4 — Pi-side head tracking + VLA integration
- [x] Phase 5 — Reliability polish (GPIO cleanup, bark fixes, head-only bark)
- [x] Phase 6 — Situation-aware behavior + semantic exploration *(implemented, hardware testing in progress)*
- [x] Multi-shot face enrollment + IoU temporal smoothing
- [x] Portable networking via Tailscale + `.env` config
- [ ] Demo video

---

## Acknowledgements

Hardware: [SunFounder PiDog](https://github.com/sunfounder/pidog).
Perception: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics),
[InsightFace](https://github.com/deepinsight/insightface),
and [Ollama](https://ollama.ai/) for local VLM inference.
