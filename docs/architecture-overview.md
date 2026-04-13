# Architecture Overview

IronBark is an autonomous VLA (Vision-Language-Action) robot dog built on the SunFounder PiDog platform. It recognizes its owner by face, follows them through rooms, and explores autonomously when the owner disappears.

## System Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5 (Body)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ USB Webcam   │  │ Ribbon Cam   │  │ motor_controller  │  │
│  │ (head-mount) │  │ (nose-mount) │  │ PiDog SDK         │  │
│  │ pi_sender    │  │ pi_sender    │  │ 12 servos + head  │  │
│  │ :50505 PUSH  │  │ :50506 PUSH  │  │ speaker + RGB     │  │
│  └──────┬───────┘  └──────┬───────┘  │ ultrasonic sensor │  │
│         │                 │          └───┬──────┬────────┘  │
│         │                 │              │      │           │
└─────────│─────────────────│──────────────│──────│───────────┘
          │ JPEG @30fps     │ JPEG @30fps  │SUB   │PUB
          │ ZMQ PUSH        │ ZMQ PUSH     │5556  │5558
          │                 │              │5557  │
          ▼                 ▼              ▼      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Mac — Apple Silicon (Brain)                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   follower.py                           │ │
│  │  ZMQ PULL :50505 (webcam) + :50506 (ribbon cam)        │ │
│  │  ZMQ PUB  :5556  (follow-me commands)                  │ │
│  │  ZMQ SUB  :5558  (telemetry receiver thread)           │ │
│  │                                                         │ │
│  │  ┌──────────────────────────────────────────────┐       │ │
│  │  │         perception_pipeline.py               │       │ │
│  │  │                                              │       │ │
│  │  │  Main thread (10 Hz):                        │       │ │
│  │  │    YOLO11n (MPS) ──▶ ArcFace gallery (CPU)   │       │ │
│  │  │    ──▶ IoU tracker ──▶ temporal smoothing     │       │ │
│  │  │                                              │       │ │
│  │  │  VLM worker thread (async):                  │       │ │
│  │  │    Moondream via Ollama (:11434)              │       │ │
│  │  │    situation_query() every 2.5s (FOLLOW)     │       │ │
│  │  │    explore_query()  on-demand   (EXPLORE)    │       │ │
│  │  └──────────────────────────────────────────────┘       │ │
│  │                                                         │ │
│  │  State Machine: IDLE ──▶ FOLLOW ──▶ SEARCH ──▶ EXPLORE  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────┐                                            │
│  │ Ollama       │  Moondream VLM — ~245ms per query          │
│  │ :11434       │  YOLO-grounded prompting (anti-hallucinate) │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

## Why Two Nodes

The Raspberry Pi 5 cannot run YOLO, ArcFace, and a VLM at real-time speeds simultaneously. The Mac has an Apple Silicon GPU (MPS) that runs YOLO in ~30ms and Moondream in ~245ms. Splitting the system into a body (Pi) and brain (Mac) connected over ZMQ gives both real-time motor control and fast ML inference.

## Data Flow

### Fast Path (10 Hz, ~70ms end-to-end)

```
Pi webcam ──[JPEG]──▶ Mac YOLO ──▶ ArcFace gallery match ──▶ face smoothing
    ──▶ follower state machine ──▶ Command JSON ──▶ Pi motor_controller
```

This loop runs every ~100ms. YOLO finds people, ArcFace identifies the owner, and the follower sends a motor command. The dog tracks and follows the owner at 10 Hz.

### Slow Path (0.4 Hz, ~245ms per query)

```
webcam frame ──▶ VLM worker thread ──▶ Moondream (Ollama)
    ──▶ SituationResponse (behavior mode)
    ──▶ follower adjusts speed, bark volume, arrival distance
```

Every 2.5 seconds, the VLM reads the scene and sets a behavior mode (ACTIVE, GENTLE, CALM, PLAYFUL, SOCIAL). This modifies how the dog follows but never blocks the fast tracking loop.

### Explore Path (on-demand during EXPLORE state)

```
ribbon cam frame ──▶ VLM worker thread ──▶ Moondream (Ollama)
    ──▶ ExploreResponse (FORWARD / LEFT / RIGHT / BACK)
    ──▶ follower drives motors in that direction for ~2s
    ──▶ repeat
```

When the owner is lost, the VLM examines the nose-mounted ribbon camera (which sees the floor and doorways) and picks a direction to explore.

## Dual-Camera Architecture

| Camera | Mount Point | Servo-Controlled | ZMQ Port | Purpose |
|--------|------------|-----------------|----------|---------|
| USB UVC webcam | Head (pan/tilt servo) | Yes | 50505 | Owner tracking — faces, gestures, people detection |
| OV5647 ribbon cam | Nose (fixed mount) | No | 50506 | Navigation — doorways, obstacles, floor-level features |

During FOLLOW, the head pitches up to see the owner's face. This means the webcam sees the ceiling during SEARCH/EXPLORE. The ribbon camera always sees forward at floor level, making it ideal for VLM-guided navigation.

## Thread Model

| Thread | Location | Frequency | Work |
|--------|----------|-----------|------|
| Main loop | `follower.py` | 30 fps (frame-limited) | Receive frames, run perception, send commands |
| VLM worker | `perception_pipeline.py` | ~0.4 Hz (situation) or on-demand (explore) | Moondream inference via Ollama HTTP |
| Telemetry receiver | `follower.py` | ~5 Hz | Subscribe to Pi ultrasonic + battery data |
| Motor loop | `motor_controller.py` (Pi) | 20 Hz | Poll commands, read ultrasonic, execute gaits, publish telemetry |

## Networking

All communication uses Tailscale mesh VPN, which assigns stable IPs regardless of WiFi network:

| Node | Tailscale IP |
|------|-------------|
| Mac (brain) | `100.x.x.x` (your Tailscale IP) |
| Pi (body) | `100.y.y.y` (your Tailscale IP) |

IPs are stored in `.env` (gitignored) and loaded by `config.py` at import time. No hardcoded IPs in source code.

## File Layout

```
ironbark/
├── config.py                  # Shared constants (both nodes import this)
├── .env                       # Network IPs (gitignored)
├── .env.example               # Template for .env
├── deploy.sh                  # SCP code to Pi
├── data/
│   ├── owner_gallery.npy      # Multi-distance face embedding gallery (Nx512)
│   └── owner_embedding.npy    # Legacy single embedding (auto-migrated to gallery)
│
├── pc/                        # Mac-side code (brain)
│   ├── follower.py            # Phase 6 state machine + main loop
│   ├── perception_pipeline.py # YOLO + ArcFace + VLM orchestration
│   ├── enroll_owner.py        # Owner face enrollment UI
│   ├── remote_control.py      # WASD teleop client
│   ├── bbox_calibrate.py      # Arrival distance calibration
│   └── perception/
│       ├── __init__.py
│       ├── yolo_detector.py   # YOLOv11n person detection
│       ├── face_recognizer.py # ArcFace owner recognition
│       └── vlm_reasoner.py    # Moondream VLM queries
│
├── pi/                        # Pi-side code (body)
│   ├── motor_controller.py    # Unified motor controller
│   ├── pi_sender.py           # Camera streaming over ZMQ
│   ├── remote_control.py      # Legacy teleop receiver
│   ├── read_distance.py       # Ultrasonic sensor utility
│   └── run_cameras.sh         # Launch both camera streams
│
└── scratch_*.py               # Benchmark and test scripts
```

## Key Design Decisions

1. **YOLO on fast path, VLM on slow path.** YOLO at 10 Hz keeps tracking responsive. VLM at 0.4 Hz provides scene understanding without blocking.

2. **Pi-side head tracking.** The Mac sends the raw owner bounding box; the Pi computes head servo angles locally. This eliminates a 50ms network round-trip.

3. **YOLO-grounded VLM prompting.** YOLO's person count is injected into the VLM prompt. The VLM describes the scene but cannot invent objects YOLO didn't detect.

4. **Multi-distance gallery enrollment.** 30 face samples captured at 3 distances (2 ft, 4 ft, 6-8 ft) from the ground-level camera. Stored as an Nx512 gallery — not averaged — so max-of-gallery matching preserves distance-variant and angle-variant information. Enrollment should happen each session since body ReID (Phase B) will encode clothing appearance.

5. **ZMQ CONFLATE.** Video sockets keep only the latest frame. If the Mac falls behind, stale frames are discarded automatically.

6. **2-consecutive-same hysteresis.** Behavior mode changes require the VLM to agree with itself twice in a row. Prevents mode flickering.

7. **Dual cameras for dual purposes.** Head cam for faces (tilts up), nose cam for navigation (fixed, level with floor).
