# API Reference: Vision Pipeline

The vision pipeline runs on the Mac and processes webcam frames through three ML models: YOLO (object detection), ArcFace (face recognition), and Moondream (scene understanding via VLM).

## Overview

```
pc/perception_pipeline.py    Orchestrates all three models, manages threads
pc/perception/yolo_detector.py    YOLOv11n person detection
pc/perception/face_recognizer.py  ArcFace face recognition + enrollment
pc/perception/vlm_reasoner.py     Moondream VLM situation + explore queries
```

---

## PerceptionPipeline

**Module:** `pc/perception_pipeline.py`

Orchestrates YOLO detection, face recognition, temporal smoothing, and asynchronous VLM queries into a single `process_frame()` call.

### `class PerceptionPipeline`

```python
pipeline = PerceptionPipeline(config)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `config` | module or dict | Configuration source. If a module, attributes are read via `vars()`. Expects keys matching `config.py` constants. |

**Initialization behavior:**
- Creates `YOLODetector` (auto-detects MPS/CUDA/CPU)
- Creates `FaceRecognizer` (loads `owner_embedding.npy` if it exists)
- Creates `VLMReasoner` (connects to Ollama)
- Prepares VLM worker thread (not started until `start()` is called)

---

### `start()`

Start the background VLM worker thread.

```python
pipeline.start()
```

Must be called before `process_frame()`. The VLM thread runs as a daemon and processes situation/explore queries asynchronously.

---

### `stop()`

Stop the VLM worker thread and clean up.

```python
pipeline.stop()
```

Sends a poison pill to the VLM queue and joins the thread with a 5-second timeout.

---

### `set_vlm_query_type(query_type)`

Switch the VLM between situation and explore query modes.

```python
pipeline.set_vlm_query_type("explore")
```

**Parameters:**

| Name | Type | Values | Description |
|------|------|--------|-------------|
| `query_type` | str | `"situation"`, `"explore"` | Which VLM query to run on the next frame. `situation` uses the webcam; `explore` uses the ribbon cam (with webcam fallback). |

Called by the follower state machine in `_transition()` when switching between FOLLOW/SEARCH and EXPLORE states.

---

### `process_frame(frame, nav_frame=None)`

Run the full perception stack on one frame. This is the main entry point called every loop iteration.

```python
result = pipeline.process_frame(frame, nav_frame=latest_nav_frame)
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `frame` | `np.ndarray` | Yes | BGR webcam frame (640x480). Used for YOLO detection, face recognition, and situation VLM queries. |
| `nav_frame` | `np.ndarray` or None | No | BGR ribbon camera frame. Used for explore VLM queries. Falls back to `frame` if None. |

**Returns:** `PerceptionResult`

**Processing steps (in order):**
1. YOLO detection on `frame` -> list of `PersonDetection`
2. InsightFace face detection on full `frame` (finds all faces)
3. For each YOLO person bbox, match the nearest InsightFace face by geometric overlap
4. Compute cosine similarity of each matched face against `owner_embedding.npy`
5. Apply temporal smoothing (IoU tracking + rolling confidence window)
6. Queue a VLM query if interval elapsed (situation) or on-demand (explore)
7. Read latest VLM results from the worker thread (non-blocking lock)
8. Return assembled `PerceptionResult`

**Latency:** ~60-75ms for steps 1-5 (fast path). VLM runs asynchronously and does not block.

---

### `draw_overlay(frame, result)`

Draw bounding boxes, face labels, mode, and stats onto a frame for display.

```python
display = pipeline.draw_overlay(frame, result)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `frame` | `np.ndarray` | Original BGR frame to annotate |
| `result` | `PerceptionResult` | Output from `process_frame()` |

**Returns:** `np.ndarray` — Annotated copy of the frame (original is not modified).

**Annotations drawn:**
- Green box + "OWNER 87%" for recognized owner
- Red box + "Stranger 32%" for non-owner
- Yellow box + "Person 95%" when no owner embedding is enrolled
- Behavior mode and scene description (top of frame)
- Explore direction and reasoning (when in EXPLORE state)
- FPS, frame ID, person count (bottom of frame)

---

## Data Classes

### `PerceptionResult`

```python
@dataclass
class PerceptionResult:
    detections:         List[PersonDetection]     # YOLO person detections
    face_matches:       List[FaceMatch]           # Parallel to detections
    vlm_response:       Optional[VLMResponse]     # Legacy Phase 4 (kept for compat)
    vlm_seq:            int                       # Legacy sequence counter
    situation_response: Optional[SituationResponse]  # Latest behavior mode
    situation_seq:      int                       # Increments on each new situation
    explore_response:   Optional[ExploreResponse]    # Latest explore direction
    explore_seq:        int                       # Increments on each new explore
    frame_id:           int                       # Monotonic frame counter
    latency_ms:         float                     # Fast-path processing time
```

`situation_response` and `explore_response` persist across frames — they hold the latest VLM result until a newer one arrives. Use `situation_seq` and `explore_seq` to detect when a new result is available.

---

## YOLODetector

**Module:** `pc/perception/yolo_detector.py`

Detects people in camera frames using YOLOv11 nano.

### `class YOLODetector`

```python
detector = YOLODetector(model_path="yolo11n.pt", conf_threshold=0.5, device=None)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_path` | str | `"yolo11n.pt"` | Path to YOLO model weights. Downloaded automatically on first run. |
| `conf_threshold` | float | `0.5` | Minimum detection confidence. Detections below this are discarded. |
| `device` | str or None | None | Force device (`"mps"`, `"cuda"`, `"cpu"`). Auto-detected if None: MPS > CUDA > CPU. |

---

### `detect(frame)`

Run YOLO inference on a single frame.

```python
detections = detector.detect(frame)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `frame` | `np.ndarray` | BGR image (any resolution, 640x480 typical) |

**Returns:** `List[PersonDetection]` — One entry per detected person. Empty list if no people found.

**Notes:**
- Only detects class 0 (person). All other YOLO classes are filtered out.
- Runs with `verbose=False` to suppress Ultralytics logging.
- Each detection includes a face crop (top 30% of person bbox) for downstream face recognition.
- Latency: ~30ms on Apple Silicon MPS.

---

### `PersonDetection`

```python
@dataclass
class PersonDetection:
    bbox:       Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates
    confidence: float                       # YOLO confidence 0.0-1.0
    center:     Tuple[int, int]            # (cx, cy) bbox center
    area:       int                         # bbox area in pixels (w * h)
    face_crop:  Optional[np.ndarray]       # Top 30% of bbox, or None if too small
```

`face_crop` is None when the cropped region is smaller than 20x20 pixels (`MIN_FACE_CROP_SIZE`).

---

## FaceRecognizer

**Module:** `pc/perception/face_recognizer.py`

Identifies whether a detected person is the registered owner using ArcFace embeddings.

### `class FaceRecognizer`

```python
recognizer = FaceRecognizer(
    embedding_path="data/owner_embedding.npy",
    threshold=0.6,
    model_pack="buffalo_l"
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `embedding_path` | str | `"data/owner_embedding.npy"` | Path to the enrolled owner's 512-dim embedding file |
| `threshold` | float | `0.6` | Cosine similarity cutoff. Values above this are classified as owner. Overridden by `config.FACE_THRESHOLD` (0.45) at runtime. |
| `model_pack` | str | `"buffalo_l"` | InsightFace model pack. `buffalo_l` includes ArcFace ResNet50. |

**Notes:**
- Uses CPU inference (`ctx_id=-1`). InsightFace doesn't support MPS natively, but CPU is fast enough on Apple Silicon.
- If `owner_embedding.npy` doesn't exist, recognition always returns `is_owner=False`.

---

### `recognize(face_crop)`

Compare a face crop against the enrolled owner.

```python
match = recognizer.recognize(face_crop)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `face_crop` | `np.ndarray` | BGR face image (any size). Typically the top 30% of a YOLO person bbox. |

**Returns:** `FaceMatch`

**Processing:**
1. Run InsightFace face detection + embedding extraction on the crop.
2. If no face found, return `FaceMatch(is_owner=False, confidence=0.0)`.
3. Compute cosine similarity between the detected face embedding and `owner_embedding`.
4. If similarity >= threshold, `is_owner=True`.

---

### `enroll_owner(embeddings)`

Create the owner embedding from multiple face samples.

```python
final_embedding = recognizer.enroll_owner(embeddings_list)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `embeddings` | List[np.ndarray] | List of 512-dim face embeddings (typically 25 samples) |

**Returns:** `np.ndarray` — The final 512-dim owner embedding (unit length).

**Processing:**
1. L2-normalize each embedding individually.
2. Compute the element-wise mean across all normalized embeddings.
3. Re-normalize the mean to unit length.
4. Save to `owner_embedding.npy`.

**Why multi-shot:** A single embedding is brittle — it's biased toward whatever head angle the enrollment photo captured. Averaging 25 samples across different head poses creates a rotation-invariant embedding.

---

### `FaceMatch`

```python
@dataclass
class FaceMatch:
    is_owner:   bool                     # True if similarity >= threshold
    confidence: float                    # Cosine similarity (0.0 - 1.0)
    embedding:  Optional[np.ndarray]     # 512-dim embedding, or None if no face detected
```

---

## Temporal Smoothing

**Method:** `PerceptionPipeline._smooth_face_matches()`

Single-frame face recognition flickers (motion blur, partial occlusion). Temporal smoothing stabilizes the `is_owner` decision across frames.

### How It Works

1. **IoU tracking:** Associate each detection with a track from the previous frame using bounding box IoU (threshold: 0.3).
2. **Rolling window:** Each track maintains a deque of the last 5 face confidence values.
3. **Selective update:** Only add a confidence value when InsightFace actually detected a face (embedding is not None). A missed face in one frame doesn't pollute the buffer with 0.0.
4. **Vote:** The mean confidence over the window determines `is_owner`.

**Configuration:**

| Constant | Default | Description |
|----------|---------|-------------|
| `FACE_SMOOTH_WINDOW` | 5 | Rolling deque length |
| `FACE_TRACK_IOU` | 0.3 | Minimum IoU to associate detection with existing track |

---

## VLMReasoner

**Module:** `pc/perception/vlm_reasoner.py`

Queries a VLM (Moondream via Ollama) to understand the scene and guide behavior.

### `class VLMReasoner`

```python
vlm = VLMReasoner(model="moondream", host="http://localhost:11434", timeout_s=60)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | str | `"llava:7b"` | Ollama model name. Overridden by `config.VLM_MODEL` ("moondream"). |
| `host` | str | `"http://localhost:11434"` | Ollama server URL |
| `timeout_s` | int | `60` | HTTP request timeout in seconds |

---

### `situation_query(frame, detection_count=None, jpeg_quality=75)`

Classify the scene into a behavior mode.

```python
response = vlm.situation_query(frame, detection_count=2)
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `frame` | `np.ndarray` | Yes | BGR webcam frame |
| `detection_count` | int or None | No | Number of people YOLO detected. Injected into prompt to prevent hallucinations. |
| `jpeg_quality` | int | No | JPEG compression for the Ollama API (default 75) |

**Returns:** `SituationResponse`

**Prompt structure:**
The prompt template (`SITUATION_PROMPT_TEMPLATE`) includes a `{yolo_context}` placeholder filled with:
- `detection_count=0`: "CRITICAL: YOLO detected 0 people. Do NOT describe any person."
- `detection_count=N`: "CRITICAL: YOLO detected exactly N person(s). Describe only those."
- `detection_count=None`: No grounding context (not recommended).

This YOLO-grounded prompting is the primary defense against VLM hallucinations.

**Expected VLM output:**
```
SCENE: Person standing in a living room with white furniture
MODE: ACTIVE
```

**Latency:** ~245ms with Moondream on Apple Silicon.

---

### `explore_query(frame, jpeg_quality=75)`

Ask the VLM which direction to explore.

```python
response = vlm.explore_query(ribbon_cam_frame)
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `frame` | `np.ndarray` | Yes | BGR frame from the ribbon camera (or webcam fallback) |
| `jpeg_quality` | int | No | JPEG compression (default 75) |

**Returns:** `ExploreResponse`

**Expected VLM output:**
```
SCENE: Hallway with a doorway on the left leading to another room
DIRECTION: LEFT
```

**Latency:** ~245ms with Moondream.

---

### `health_check()`

Check if Ollama is reachable.

```python
is_alive = vlm.health_check()  # Returns bool
```

---

### `list_models()`

List all models available in Ollama.

```python
models = vlm.list_models()  # Returns List[str]
```

---

### Response Data Classes

#### `SituationResponse`

```python
@dataclass
class SituationResponse:
    mode:        str      # "ACTIVE" | "GENTLE" | "CALM" | "PLAYFUL" | "SOCIAL"
    description: str      # Scene description from VLM
    latency_ms:  float    # Query time in milliseconds
    model:       str      # Model name (e.g., "moondream")
```

#### `ExploreResponse`

```python
@dataclass
class ExploreResponse:
    direction:  str      # "FORWARD" | "LEFT" | "RIGHT" | "BACK"
    reasoning:  str      # Why the VLM chose this direction
    latency_ms: float    # Query time in milliseconds
    model:      str      # Model name
```

---

## Latency Budget

| Component | Typical Latency | Frequency | Thread |
|-----------|----------------|-----------|--------|
| YOLO11n detection | ~30ms | Every frame (10 Hz) | Main |
| InsightFace face detection | ~40ms | Every frame with detections | Main |
| ArcFace cosine similarity | <1ms | Per face | Main |
| Temporal smoothing (IoU + window) | <1ms | Every frame | Main |
| **Fast path total** | **~60-75ms** | **10 Hz** | **Main** |
| Moondream situation query | ~245ms | Every 2.5s | VLM worker |
| Moondream explore query | ~245ms | On-demand (EXPLORE state) | VLM worker |

The fast path (YOLO + face) runs synchronously on the main thread and never waits for the VLM. The VLM runs on a background thread and its results are read via a lock-protected shared variable whenever available.
