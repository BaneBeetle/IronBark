# IronBark Technical Guide

A complete walkthrough of every technology and method used to build IronBark — an autonomous VLA (Vision-Language-Action) robot dog that recognizes its owner by face, follows them, and explores rooms using a vision-language model.

---

## 1. Hardware

### Robot Platform
- **SunFounder PiDog** — a quadruped robot dog kit with 12 servos (3 per leg), head pan/tilt, tail servo, RGB LED strip, speaker, and ultrasonic distance sensor
- Controlled via the `pidog` Python SDK, which provides gaits (walk, trot), poses (sit, stand, lie), head/tail control, RGB patterns, and `speak()` audio

### Compute Nodes
| Node | Hardware | Role |
|------|----------|------|
| **Raspberry Pi 5** | ARM Cortex-A76, 8GB RAM, CSI ribbon camera + USB webcam | Body — motor control, camera streaming, ultrasonic sensing, speaker |
| **Mac (Apple Silicon)** | M-series GPU (MPS) | Brain — YOLO detection, face recognition, VLM inference, behavior decisions |

The Pi cannot run YOLO or a VLM at real-time speeds. The Mac has a GPU. So the Pi streams video to the Mac, the Mac thinks, and sends motor commands back. This two-node architecture is the core design decision.

### Cameras
| Camera | Mount | Port | Purpose |
|--------|-------|------|---------|
| USB UVC webcam | Head (tilts with head servo) | 50505 | Owner tracking — faces, gestures, people |
| OV5647 ribbon camera | Nose (fixed, level with floor) | 50506 | Navigation — doorways, obstacles, room layout |

The head-mounted webcam pitches up to see faces during FOLLOW, which means it sees the ceiling during EXPLORE. The nose-mounted ribbon camera always sees forward at floor level, making it ideal for VLM-guided navigation.

### Networking
- **Tailscale mesh VPN** — assigns stable IPs (Pi: `100.91.138.116`, Mac: `100.100.73.118`) regardless of which WiFi network you're on. No manual IP configuration when moving between home, school, etc.

---

## 2. Communication — ZeroMQ (ZMQ)

All inter-node communication uses ZMQ, a lightweight message-passing library. No HTTP overhead, no serialization frameworks — just raw bytes over TCP sockets.

### Port Layout
| Port | Direction | Pattern | Content |
|------|-----------|---------|---------|
| 50505 | Pi -> Mac | PUSH/PULL | USB webcam JPEG stream |
| 50506 | Pi -> Mac | PUSH/PULL | Ribbon camera JPEG stream |
| 5556 | Mac -> Pi | PUB/SUB | Follow-me motor commands (JSON) |
| 5557 | Mac -> Pi | PUB/SUB | Teleop commands (JSON) |
| 5558 | Pi -> Mac | PUB/SUB | Telemetry — distance, battery (JSON) |

### Why PUSH/PULL for Video
- **PUSH/PULL** is one-to-one, no subscription filtering needed, and importantly supports `ZMQ_CONFLATE=1` — which tells ZMQ to keep only the latest message in the buffer and discard older ones. This means if the Mac is busy (e.g., running YOLO), it always processes the *freshest* frame when it's ready, not a stale one from 500ms ago.

### Why PUB/SUB for Commands
- **PUB/SUB** allows multiple subscribers. The motor controller can listen to both follow-me commands (port 5556) and teleop commands (port 5557) simultaneously with last-wins multiplexing.

### Video Frame Format
```
[8 bytes]  int64   — timestamp (microseconds since epoch)
[4 bytes]  uint32  — JPEG payload length
[N bytes]  bytes   — JPEG-compressed frame data
```
JPEG compression at quality 60-75 keeps bandwidth to ~7-8 Mbps per stream at 640x480 @ 30fps.

### Command Format (JSON)
```json
{
  "action": "forward",
  "speed": 80,
  "step_count": 2,
  "head_yaw": 0.0,
  "head_pitch": 15,
  "head_mode": "local",
  "owner_bbox": {"x": 280, "y": 120, "w": 80, "h": 200, "frame_w": 640, "frame_h": 480},
  "bark": false,
  "bark_volume": 80,
  "idle_pose": "stand",
  "thinking": false
}
```

---

## 3. Camera Streaming — `pi/pi_sender.py`

The Pi runs one `pi_sender.py` process per camera.

### USB Webcam
```bash
python3 -u pi_sender.py --source usb --port 50505
```
Uses OpenCV `VideoCapture` to open the USB device. Scans `/dev/video8` through `/dev/video20` (indices 0-7 are reserved for the ribbon camera's V4L2 nodes).

### Ribbon Camera (picamera2)
```bash
python3 -u pi_sender.py --source picamera --port 50506
```
Uses `picamera2` (Python bindings for libcamera) to access the OV5647 CSI sensor. This is the Linux camera stack for Raspberry Pi — not OpenCV's `VideoCapture`, which doesn't support CSI cameras.

### Frame Loop
1. Capture frame (640x480 BGR)
2. JPEG encode with configurable quality
3. Pack timestamp + length + JPEG bytes
4. Send via ZMQ PUSH socket to Mac IP

The `-u` flag forces unbuffered Python stdout so you see log output in real-time over SSH instead of in 4KB chunks.

---

## 4. Perception Pipeline — `pc/perception_pipeline.py`

This is the Mac-side brain. It processes each webcam frame through three ML models and returns a structured result.

### Architecture
```
Webcam frame (30 fps)
    |
    v
[YOLO11n] ─── person detections (bbox, confidence) ──> 30ms on MPS
    |
    v (for each person bbox)
[ArcFace] ─── face embedding ─── cosine sim vs owner ──> ~50ms per face
    |
    v (temporal smoothing across frames)
[IoU Tracker] ─── rolling confidence window ──> is_owner vote
    |
    v (every 2.5 seconds, async)
[Moondream VLM] ─── scene description + behavior mode ──> ~245ms
```

### Why Three Models, Not One
- **YOLO** is fast (30ms) and runs every frame. It finds *where* people are.
- **ArcFace** answers *who* they are. YOLO can't distinguish your owner from a stranger.
- **VLM** answers *what's happening* — is it calm, playful, social? Are there doorways to explore? This runs 10x slower so it's on a background thread at 2.5s intervals.

No single model does all three. The pipeline runs them at different frequencies matched to their latency.

### Thread Model
- **Main thread**: YOLO + ArcFace (fast path, every frame, 10Hz)
- **VLM worker thread**: situation or explore query (slow path, ~2.5s intervals)
- A `threading.Lock` protects the shared VLM result. The main thread reads the latest VLM result without blocking.

---

## 5. YOLO11n Object Detection — `pc/perception/yolo_detector.py`

### What It Does
Detects all people in a frame and returns their bounding boxes.

### Model
- **YOLOv11 nano** (`yolo11n.pt`) — smallest variant, optimized for speed
- Runs on Apple Metal Performance Shaders (MPS) automatically via Ultralytics
- Confidence threshold: 0.5 (ignore detections below 50%)
- Only detects class 0 (person) — all other YOLO classes are filtered out

### Output
For each person detected:
```python
PersonDetection(
    bbox=(x1, y1, x2, y2),     # pixel coordinates
    center=(cx, cy),             # bbox center
    area=w*h,                    # bbox area in pixels
    confidence=0.87,             # YOLO confidence
    face_crop=np.array(...)      # top 30% of bbox (for ArcFace)
)
```

### Why YOLO and Not Just Face Detection
YOLO gives you the full body bounding box. This is critical because:
1. **Distance estimation** — bbox area ratio (bbox area / frame area) is a proxy for distance. Big bbox = close. Used for approach speed and arrival detection.
2. **Steering** — bbox center X offset from frame center tells the dog which way to turn.
3. **Face extraction** — the top 30% of the person bbox is cropped for face recognition. This is more reliable than running a separate face detector because YOLO already localized the person.

---

## 6. Face Recognition — `pc/perception/face_recognizer.py`

### What It Does
Determines whether a detected person is the owner or a stranger.

### Model
- **InsightFace buffalo_l** — a face analysis suite that includes ArcFace (ResNet50 backbone)
- ArcFace produces a 512-dimensional embedding vector for each face
- Recognition is embedding similarity, not classification — it compares the live face to the enrolled owner embedding

### Enrollment Process (`pc/enroll_owner.py`)
1. User presses SPACE to start
2. System captures 25 face samples over 8 seconds
3. Rate-limited to 1 sample every 0.2s — forces the user to move their head, capturing pose variety
4. Minimum face size: 80px (rejects blurry/distant faces)
5. Each sample produces a 512-dim ArcFace embedding
6. All 25 embeddings are L2-normalized individually
7. Averaged into a single mean embedding
8. Mean is re-normalized to unit length
9. Saved to `data/owner_embedding.npy`

### Why Multi-Shot Enrollment
A single-shot enrollment (one photo) is brittle — if the enrollment photo is slightly left-facing, the system struggles with right-facing views. By averaging 25 samples across different head angles, the mean embedding becomes rotation-invariant. The L2 normalization before averaging ensures each sample contributes equally regardless of InsightFace's raw magnitude.

### Why Face-Only (Not Full Body)
Early designs used full-body appearance. The problem: the system overfits to clothes. Wear a different shirt and you're a stranger. ArcFace operates on face crops only — clothing is irrelevant.

### Recognition at Runtime
1. Extract top 30% of person bbox (face region)
2. Run InsightFace `get()` to detect face landmarks and compute embedding
3. Compute cosine similarity against `owner_embedding.npy`
4. If similarity > 0.45 → owner. Otherwise → stranger.

### Temporal Smoothing
Single-frame recognition flickers (frame N says owner, frame N+1 says stranger due to motion blur). To fix this:
1. **IoU tracking** — associate detections across frames by bounding box overlap (IoU > 0.3 = same person)
2. **Rolling confidence window** — per-tracked-person deque of length 5 (the last 5 frames' face confidence scores)
3. **Vote** — mean confidence over the window determines is_owner
4. Only add confidence when a face was actually detected that frame — a missed detection doesn't poison the buffer with 0.0

---

## 7. Vision-Language Model (VLM) — `pc/perception/vlm_reasoner.py`

### What It Does
Takes a camera frame and answers high-level questions about the scene in natural language. Two query types:
- **Situation query** — "What's happening? What behavior mode should the dog be in?"
- **Explore query** — "Which direction should the dog go to find the owner?"

### Model
- **Moondream** (1.6B parameters) — a compact VLM optimized for visual Q&A
- Served locally by **Ollama** (`http://localhost:11434`)
- ~245ms per query on Apple Silicon

### Why Moondream Over LLaVA-7B
We benchmarked three models:

| Model | Parameters | Latency | Hallucination |
|-------|-----------|---------|---------------|
| llava:7b | 7B | 1160ms | Invents people, wheelchairs, hospitals from white furniture |
| llava-llama3:8b | 8B | ~1200ms | Still hallucinates, no speedup |
| moondream | 1.6B | 245ms | Accurate, no hallucinations |

LLaVA-7B was confidently describing "a man in a wheelchair in a hospital setting" when looking at an empty room with white furniture. This isn't a minor error — it drives the behavior state machine into wrong modes. Moondream is 4.7x faster and doesn't hallucinate.

### YOLO-Grounded Prompting
Even after swapping to Moondream, we added a safety layer. The situation prompt template includes a `{yolo_context}` placeholder that gets filled with the actual YOLO detection count:

```
CRITICAL: YOLO detected 0 people in this frame. Do NOT invent or hallucinate
people that are not there. If YOLO says 0 people, there are 0 people.
```

or:

```
CRITICAL: YOLO detected exactly 2 person(s) in this frame.
Your description must be consistent with this count.
```

This grounds the VLM's language output in the object detector's spatial output. The VLM can describe the *scene*, but it cannot invent *objects* that YOLO didn't find. This is the key technique that eliminated hallucinations.

### Situation Query
- Input: webcam frame + YOLO detection count
- Output: behavior mode (ACTIVE/GENTLE/CALM/PLAYFUL/SOCIAL) + scene description
- Frequency: every 2.5 seconds
- The behavior mode controls follow speed, arrival distance, bark volume, and idle pose

### Explore Query
- Input: ribbon camera frame (or webcam fallback)
- Output: direction (FORWARD/LEFT/RIGHT/BACK) + reasoning
- Frequency: as fast as the VLM responds (~245ms) during EXPLORE state
- The reasoning is logged but the direction alone drives the motors

---

## 8. Behavior State Machine — `pc/follower.py`

### States
```
IDLE ──(owner detected)──> FOLLOW
FOLLOW ──(owner lost 2s)──> SEARCH
SEARCH ──(owner lost 5s)──> EXPLORE
EXPLORE ──(owner lost 35s)──> IDLE
Any ──(owner detected)──> FOLLOW
```

### FOLLOW State
The dog actively tracks and approaches the owner:
1. **Steering**: bbox center X offset from frame center → turn left/right. Dead zone of 100px prevents jitter.
2. **Approach**: if bbox area ratio < arrival threshold, move forward at behavior-mode speed
3. **Arrival**: bbox area ratio > 0.40 → stop, bark, sit (greeting sequence)
4. **Head tracking**: sends owner bbox to Pi, which computes yaw/pitch locally (eliminates 50ms network round-trip)

Safety hierarchy in FOLLOW:
1. Ultrasonic < 35cm → redirect or stop (highest priority)
2. Bark hold → sustain greeting for ~5 seconds
3. Arrival detection → stop + bark
4. Body centering → turn toward owner
5. Forward cruise (lowest priority)

### SEARCH State
Owner disappeared less than 5 seconds ago. The dog stands still and sweeps its head:
- Yaw: sinusoidal sweep ±35 degrees over 4 seconds
- Pitch: tilted down to -10 degrees (to see the floor/room, not the ceiling)
- Body: stopped
- VLM: situation queries continue (might detect owner returning)

### EXPLORE State
Owner gone for 5-30 seconds. The dog navigates the room using VLM-guided decisions:
1. Dog enters "thinking" mode (purple RGB, tail wag)
2. VLM receives the ribbon camera frame and returns a direction
3. Dog executes the move (forward, turn left/right, or 180-degree turn for BACK)
4. Repeat until owner found or 30s timeout
5. Head pitch stays at -10 degrees (looking forward-down for navigation)

### IDLE State
Owner gone for 35+ seconds. Dog returns to a resting pose and waits.

### Behavior Modes (VLM-Driven)
The VLM's situation query returns one of five modes that modify follow behavior:

| Mode | Speed | Arrival Ratio | Bark | Idle Pose |
|------|-------|--------------|------|-----------|
| ACTIVE | 80 | 0.40 | loud (80) | stand |
| GENTLE | 50 | 0.35 | soft (40) | sit |
| CALM | 40 | 0.30 | soft (30) | lie |
| PLAYFUL | 90 | 0.45 | loud (90) | stand |
| SOCIAL | 60 | 0.35 | medium (60) | stand |

Mode switching uses **2-consecutive-same hysteresis** — the VLM must return the same mode twice in a row before the dog switches. At 2.5s query intervals, this means minimum 5 seconds to change modes, preventing flickering.

---

## 9. Motor Control — `pi/motor_controller.py`

### What It Does
Receives commands from the Mac over ZMQ and translates them into PiDog SDK calls (servo positions, gaits, RGB, speaker).

### Loop (20 Hz)
1. Read ultrasonic distance
2. Poll both command sockets (follow-me on 5556, teleop on 5557) — last message wins
3. Safety check: if distance < 15cm and action is forward → STOP, red RGB, warning bark
4. Head tracking: if head_mode is "local", compute yaw/pitch from owner_bbox; if "remote", use Mac's angles
5. Execute motor action
6. Publish telemetry at 5Hz

### LocalHeadTracker
Instead of the Mac computing head angles and sending them over the network (adding ~50ms latency), the Mac sends the raw owner bounding box and the Pi computes head angles locally:
- **Yaw**: horizontal offset of bbox center from frame center → proportional yaw angle, clamped ±45 degrees
- **Pitch**: bbox area ratio compared to target ratio → pitch up (far away) or down (close), clamped 5-25 degrees
- **Exponential smoothing** (factor 0.7): prevents jerky head movements, converges in ~250ms
- **10-degree dead zone**: ignores micro-movements

### Why sudo
The PiDog speaker (`dog.speak()`) requires root access to the audio hardware. Without `sudo`, the speak call silently fails — the dog visually barks (head up, tail wag, RGB flash) but produces no sound.

### Danger Handling
- Forward into obstacle (< 15cm): motor blocked, red RGB, warning bark, dog backs up
- Redirect turn allowed through: if a turn command comes while danger is active, it's allowed (to steer away)
- Clear: resume blue breathing light

---

## 10. Deployment — `deploy.sh` and `.env`

### .env File
```bash
PC_IP=100.100.73.118      # Mac Tailscale IP
PI_IP=100.91.138.116      # Pi Tailscale IP
PI_USER=banebeetle
PI_DEPLOY_PATH=ironbark   # relative to ~/ on Pi (NOT ~/ironbark — tilde expansion bug)
```

### deploy.sh
Copies the following to Pi via `scp`:
- `config.py` (shared constants)
- `.env` (network config)
- `pi/pi_sender.py`
- `pi/motor_controller.py`
- `pi/remote_control.py`
- `pi/run_cameras.sh`

Note: deploy.sh flattens the `pi/` directory — files land in `~/ironbark/` on the Pi, not `~/ironbark/pi/`. This is why `run_cameras.sh` uses `cd "$(dirname "$0")"` instead of `cd "../"`.

### Run Order
1. **Pi Terminal 1**: `ssh -t pi@IP 'cd ~/ironbark && sudo python3 motor_controller.py'`
2. **Pi Terminal 2**: `ssh pi@IP 'cd ~/ironbark && python3 -u pi_sender.py --source usb --port 50505'`
3. **Pi Terminal 3**: `ssh pi@IP 'cd ~/ironbark && python3 -u pi_sender.py --source picamera --port 50506'`
4. **Mac**: `cd ~/Projects/ironbark && .venv/bin/python pc/follower.py`

Or use `run_cameras.sh start` on the Pi to launch both pi_senders in the background.

---

## 11. End-to-End Latency Budget

| Stage | Time | Runs Every |
|-------|------|-----------|
| Camera capture + JPEG encode | ~16ms | 33ms (30fps) |
| ZMQ transmit (Pi -> Mac) | <5ms | 33ms |
| YOLO detection | ~30ms | 100ms (10Hz) |
| ArcFace face recognition | ~50ms per face | 100ms |
| Temporal smoothing | <1ms | 100ms |
| Command transmit (Mac -> Pi) | <5ms | 100ms |
| Motor execution | ~20ms | 50ms (20Hz) |
| **Total fast path** | **~60-75ms** | **10Hz** |
| VLM situation query | ~245ms | 2500ms |
| VLM explore query | ~245ms | on-demand |

The fast path (YOLO + face + motor) runs at 10Hz with ~70ms latency. The VLM runs asynchronously on a background thread and its results are consumed when available — it never blocks the tracking loop.

---

## 12. Key Design Decisions

1. **Two-node architecture** — Pi can't run ML models at real-time speeds. Mac has the GPU. Splitting compute across nodes and connecting with ZMQ is the core enabling decision.

2. **YOLO on the fast path, VLM on the slow path** — YOLO at 10Hz keeps tracking responsive. VLM at 0.4Hz (every 2.5s) provides high-level scene understanding without blocking.

3. **Pi-side head tracking** — sending raw bounding boxes instead of computed angles eliminates a network round-trip. The Pi computes head angles locally with exponential smoothing.

4. **YOLO-grounded VLM prompting** — injecting the object detector's count into the VLM prompt prevents hallucinations. The VLM describes the scene but cannot invent objects YOLO didn't find.

5. **Multi-shot face enrollment** — 25 samples averaged and re-normalized creates a rotation-invariant owner embedding. Single-shot enrollment is too brittle.

6. **Face-only recognition** — ArcFace on face crops, not full-body appearance. The dog recognizes you regardless of what you're wearing.

7. **Dual cameras with different purposes** — head cam (tilts up for faces) vs nose cam (fixed, level for navigation). Discovered through testing that a single head-mounted camera can't serve both purposes.

8. **ZMQ CONFLATE** — always process the freshest frame. If the Mac falls behind, stale frames are discarded automatically.

9. **2-consecutive-same hysteresis** — behavior mode changes require the VLM to agree with itself twice. Prevents mode flickering.

10. **Tailscale for networking** — stable IPs across any network. No IP configuration when moving between locations.
