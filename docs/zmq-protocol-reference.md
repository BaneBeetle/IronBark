# ZeroMQ Protocol Reference

All inter-node communication in IronBark uses ZeroMQ (pyzmq). No HTTP, no REST, no serialization frameworks — just raw bytes and JSON over TCP.

## Port Map

| Port | Direction | ZMQ Pattern | Content | Rate |
|------|-----------|-------------|---------|------|
| 50505 | Pi -> Mac | PUSH/PULL | USB webcam JPEG stream | 30 fps |
| 50506 | Pi -> Mac | PUSH/PULL | Ribbon camera JPEG stream | 30 fps |
| 5556 | Mac -> Pi | PUB/SUB | Follow-me motor commands | 10 Hz |
| 5557 | Mac -> Pi | PUB/SUB | Teleop (WASD) commands | On keypress |
| 5558 | Pi -> Mac | PUB/SUB | Telemetry (ultrasonic, battery) | 5 Hz |

## Video Stream Protocol (Ports 50505, 50506)

### Pattern: PUSH/PULL

The Pi runs a PUSH socket that connects to the Mac. The Mac binds a PULL socket. PUSH/PULL is used because:

- Point-to-point, no subscription filtering overhead.
- Supports `ZMQ_CONFLATE=1`, which keeps only the latest message in the receive buffer.
- When the Mac is busy running YOLO, stale frames are discarded automatically. The next `recv()` always returns the freshest frame.

### Socket Configuration

**Pi (sender):**
```python
sock = ctx.socket(zmq.PUSH)
sock.connect(f"tcp://{PC_IP}:{port}")
```

**Mac (receiver):**
```python
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.CONFLATE, 1)    # Keep only latest frame
sock.setsockopt(zmq.RCVHWM, 2)     # Max 2 frames in buffer
sock.setsockopt(zmq.RCVTIMEO, 1000) # 1s timeout for initial connect
sock.bind(f"tcp://*:{port}")
```

### Message Format

```
┌──────────────┬────────────────┬──────────────────┐
│ 8 bytes      │ 4 bytes        │ N bytes          │
│ int64 (LE)   │ uint32 (LE)    │ raw bytes        │
│ timestamp_us │ jpeg_length    │ jpeg_payload     │
└──────────────┴────────────────┴──────────────────┘
```

| Field | Type | Byte Order | Description |
|-------|------|-----------|-------------|
| `timestamp_us` | `int64` | Little-endian | Microseconds since epoch (`time.time_ns() // 1000`) |
| `jpeg_length` | `uint32` | Little-endian | Length of the JPEG payload in bytes |
| `jpeg_payload` | `bytes` | — | JPEG-compressed frame (640x480, quality 60-75) |

**Encoding (Pi):**
```python
import struct, time, cv2

timestamp_us = int(time.time() * 1_000_000)
_, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
jpeg_bytes = jpeg_bytes.tobytes()
header = struct.pack("<qI", timestamp_us, len(jpeg_bytes))
sock.send(header + jpeg_bytes)
```

**Decoding (Mac):**
```python
import struct, cv2, numpy as np

HEADER_SIZE = struct.calcsize("<qI")  # 12 bytes
raw = sock.recv()

timestamp_us, payload_len = struct.unpack_from("<qI", raw, 0)
jpeg_data = raw[HEADER_SIZE : HEADER_SIZE + payload_len]
frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
```

### Bandwidth

At 640x480, JPEG quality 60, 30 fps:

| Stream | Per-Frame | Bandwidth |
|--------|-----------|-----------|
| USB webcam (port 50505) | ~32 KB | ~7.7 Mbps |
| Ribbon cam (port 50506) | ~24 KB | ~5.8 Mbps |
| **Both streams** | | **~13.5 Mbps** |

WiFi must sustain ~15 Mbps for reliable dual-camera operation. If bandwidth is constrained, reduce resolution, FPS, or JPEG quality in `config.py`.

## Command Protocol (Ports 5556, 5557)

### Pattern: PUB/SUB

The Mac publishes commands. The Pi subscribes. PUB/SUB is used because:

- The motor controller multiplexes two command sources (follow-me on 5556, teleop on 5557) with a poller.
- `ZMQ_CONFLATE=1` ensures only the latest command is processed — no command backlog.

### Socket Configuration

**Mac (publisher) — follow-me commands:**
```python
sock = ctx.socket(zmq.PUB)
sock.bind(f"tcp://*:{5556}")
```

**Pi (subscriber) — connects to Mac:**
```python
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1)
sock.connect(f"tcp://{PC_IP}:{5556}")
sock.subscribe(b"")  # Subscribe to all messages
```

### Message Format

UTF-8 encoded JSON. Sent with `sock.send(json.dumps(cmd).encode("utf-8"))`.

```json
{
  "action":      "forward",
  "speed":       80,
  "step_count":  2,
  "head_yaw":    0.0,
  "head_pitch":  15,
  "head_mode":   "local",
  "owner_bbox":  {"x": 280, "y": 120, "w": 80, "h": 200, "frame_w": 640, "frame_h": 480},
  "bark":        false,
  "bark_volume": 80,
  "idle_pose":   "stand",
  "thinking":    false
}
```

### Field Reference

| Field | Type | Required | Values | Description |
|-------|------|----------|--------|-------------|
| `action` | string | Yes | `forward`, `backward`, `turn_left`, `turn_right`, `stop` | Motor action to execute |
| `speed` | int | Yes | 0-100 | Gait speed (higher = faster leg movement) |
| `step_count` | int | Yes | 1-10 | Number of gait cycles to execute |
| `head_yaw` | float | No | -45.0 to 45.0 | Head horizontal angle (negative = left). Ignored in `local` head mode. |
| `head_pitch` | float | No | -10.0 to 35.0 | Head vertical angle (negative = down, positive = up). Ignored in `local` mode. |
| `head_mode` | string | Yes | `local`, `remote` | Who controls head tracking. `local`: Pi computes from `owner_bbox`. `remote`: Mac sends explicit angles. |
| `owner_bbox` | object | No | See below | Owner bounding box for Pi-side head tracking. Only meaningful in `local` mode. |
| `bark` | bool | No | true/false | Trigger bark sequence (head up, tail wag, speaker, RGB flash) |
| `bark_volume` | int | No | 0-99 | Speaker volume for bark. Set by behavior mode (ACTIVE=80, GENTLE=40). |
| `idle_pose` | string | No | `stand`, `sit`, `lie` | Pose after bark hold completes. Set by behavior mode. |
| `thinking` | bool | No | true/false | Trigger thinking animation (purple RGB, tail wag). Used during EXPLORE when waiting for VLM. |

### owner_bbox Object

```json
{
  "x": 280,        // Top-left X of bounding box (pixels)
  "y": 120,        // Top-left Y of bounding box (pixels)
  "w": 80,         // Width of bounding box (pixels)
  "h": 200,        // Height of bounding box (pixels)
  "frame_w": 640,  // Frame width (for center calculation)
  "frame_h": 480   // Frame height
}
```

The Pi's `LocalHeadTracker` uses this to compute head yaw (horizontal offset from frame center) and pitch (area ratio as distance proxy).

### Head Mode Semantics

| Mode | Who Computes Head Angles | When Used |
|------|--------------------------|-----------|
| `local` | Pi (`LocalHeadTracker`) from `owner_bbox` | FOLLOW state — smooth, low-latency tracking (~5ms vs ~55ms) |
| `remote` | Mac sends `head_yaw`/`head_pitch` directly | SEARCH (sinusoidal sweep), EXPLORE (gentle oscillation), BARK (head up) |

## Telemetry Protocol (Port 5558)

### Pattern: PUB/SUB

The Pi publishes telemetry. The Mac subscribes. Published at 5 Hz (every 200ms).

### Socket Configuration

**Pi (publisher):**
```python
sock = ctx.socket(zmq.PUB)
sock.bind(f"tcp://*:{5558}")
```

**Mac (subscriber):**
```python
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1)
sock.setsockopt(zmq.RCVTIMEO, 200)
sock.connect(f"tcp://{PI_IP}:{5558}")
sock.subscribe(b"")
```

### Message Format

UTF-8 JSON string, sent with `sock.send_string(json.dumps(telem))`.

```json
{
  "distance_cm": 45.5,
  "battery_v":   7.8,
  "action":      "forward",
  "danger":      false,
  "source":      "follow",
  "timestamp":   1700000000.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `distance_cm` | float | Ultrasonic sensor reading in centimeters. -1 if sensor unavailable. |
| `battery_v` | float | Battery voltage. Typical range: 6.5V (low) to 8.4V (full). |
| `action` | string | Current motor action being executed |
| `danger` | bool | True if ultrasonic distance < `DANGER_DISTANCE` (15 cm) |
| `source` | string | Command source: `"follow"` or `"teleop"` |
| `timestamp` | float | Unix timestamp (seconds) |

## Command Multiplexing

The Pi's `motor_controller.py` uses a `zmq.Poller` to read from both command sockets:

```python
poller = zmq.Poller()
poller.register(follow_sock, zmq.POLLIN)   # port 5556
poller.register(teleop_sock, zmq.POLLIN)   # port 5557

socks = dict(poller.poll(50))  # 50ms timeout
# Process both; last-wins if both have data
```

When both follower and teleop send commands simultaneously, the last socket processed in the loop wins. In practice, you run either follow-me or teleop, not both.

## Connection Topology Summary

```
Pi PUSH ──connect──▶ Mac PULL (bind)     Video: 50505, 50506
Mac PUB  ──bind────▶ Pi SUB  (connect)   Follow-me: 5556
Pi SUB   ──bind────▶ Mac PUB (connect)   Teleop: 5557
Pi PUB   ──bind────▶ Mac SUB (connect)   Telemetry: 5558
```

The general rule: the "server" side (the one that's always running) binds. The "client" side connects. Video senders connect to the Mac (brain is always up). Command publishers bind (Mac is the authority). Telemetry publisher binds on the Pi (Pi owns sensor data).
