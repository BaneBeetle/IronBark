# =============================================================================
# IronBark Configuration
# Shared constants for Pi sender and PC receiver.
# Import this module anywhere with: import config
#
# Network IPs are loaded from a local `.env` file (gitignored). Copy
# `.env.example` → `.env` and fill in the values for your setup.
# =============================================================================

import os
from pathlib import Path

# ---- .env loader (no python-dotenv dependency) -----------------------------
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip()
            # Strip optional surrounding quotes
            if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ("'", '"'):
                _v = _v[1:-1]
            os.environ.setdefault(_k, _v)

# ---- Network ----------------------------------------------------------------
# Set PC_IP and PI_IP in your local .env file. Defaults to localhost so a
# missing .env fails loudly during connect rather than silently misrouting.
PC_IP    = os.environ.get("PC_IP", "127.0.0.1")    # Mac (brain)
PI_IP    = os.environ.get("PI_IP", "127.0.0.1")    # Raspberry Pi (dog)
ZMQ_PORT   = 50505              # ZeroMQ PUSH/PULL port for video stream
MJPEG_PORT = 9000               # MJPEG HTTP stream port (fallback)

# ---- Camera -----------------------------------------------------------------
CAMERA_RESOLUTION = (640, 480)
JPEG_QUALITY      = 75
TARGET_FPS        = 30

# =============================================================================
# Phase 2 - Perception Stack
# =============================================================================

YOLO_MODEL          = "yolo11n.pt"
YOLO_CONF_THRESHOLD = 0.5

VLM_MODEL = "llava:7b"
VLM_HOST  = "http://localhost:11434"

FACE_THRESHOLD       = 0.45     # cosine sim cutoff (was 0.35; bumped after multi-shot enrollment)
FACE_CROP_RATIO      = 0.30
OWNER_EMBEDDING_PATH = "data/owner_embedding.npy"

# ── Face recognition smoothing & enrollment ──────────────────────
FACE_SMOOTH_WINDOW   = 5        # rolling window of confidences for is_owner decision
FACE_TRACK_IOU       = 0.3      # IoU threshold for cross-frame detection matching
ENROLL_NUM_SAMPLES   = 25       # face crops to capture during enrollment
ENROLL_TIMEOUT_S     = 8.0      # max seconds to gather all samples
ENROLL_MIN_FACE_PX   = 80       # min face bbox side (pixels) to accept a sample
ENROLL_SAMPLE_INTERVAL_S = 0.2  # min seconds between captured samples (forces pose variety)

# =============================================================================
# Phase 3 - Follow-Me Behavior
# =============================================================================

CMD_PORT = 5556

FRAME_CENTER_X     = 320
STEERING_DEAD_ZONE = 100
TARGET_AREA_RATIO  = 0.12
BODY_TURN_THRESHOLD = 120     # pixels from center before body turns (generous margin)
MIN_AREA_RATIO     = 0.02

ULTRASONIC_STOP_CM  = 35
ULTRASONIC_SLOW_CM  = 60

BARK_HEAD_PITCH     = 35        # Look all the way up during arrival bark

SEARCH_TIMEOUT_S   = 5.0
EXPLORE_TIMEOUT_S  = 30.0

HEAD_DEFAULT_PITCH = 15

# ── Timing ──────────────────────────────────────────────────────────
CMD_INTERVAL        = 0.1       # seconds between Mac→Pi commands (was 0.3)
FOLLOW_GAIT_PAUSE_S = 0.05     # pause between follow-me gaits (was 0.2)

# ── Arrival behavior — bbox area ratio thresholds ───────────────────
# The dog uses bbox size as a distance proxy: big bbox = close, small = far.
# Same concept drones use — no ultrasonic needed, camera-based.
#
# CALIBRATION: Run follower.py and watch the "[FOLLOW] area=X.XXX" log.
#   Stand at "arrived" distance (~3 feet) → note the ratio → set ARRIVED
ARRIVAL_ARRIVED_RATIO = 0.40    # bbox > 40% → arrived! stop and bark

# =============================================================================
# Phase 4 - VLA (Vision-Language-Action) Integration
# =============================================================================

VLM_ACTION_CONFIDENCE_THRESHOLD = 0.5   # Min confidence to act on VLM action
VLM_TURN_STEP_COUNT = 4                 # step_count for VLM turns (more decisive)
VLM_TURN_SPEED = 90                     # Speed for VLM turns (slightly < max)
VLM_ACTION_TIMEOUT_S = 8.0              # Expire stale VLM action after this long

# =============================================================================
# Remote Control (Teleoperation)
# =============================================================================

REMOTE_CMD_PORT   = 5557        # Mac→Pi: WASD commands
REMOTE_TELEM_PORT = 5558        # Pi→Mac: telemetry (distance, battery, state)
DANGER_DISTANCE   = 15          # cm — ultrasonic auto-stop threshold

# =============================================================================
# Phase 6 — Situation-Aware Behavior + Semantic Exploration
# =============================================================================

# ── Part A: Situation-Aware Behavior ──────────────────────────────
# VLM reads the scene every few seconds and sets a behavior mode that
# modifies how the dog follows: speed, arrival distance, bark, idle pose.
VLM_SITUATION_INTERVAL_S = 2.5      # seconds between situation queries
VLM_SITUATION_TIMEOUT_S  = 8.0      # discard stale mode after this

BEHAVIOR_MODES = {
    "ACTIVE":  {"speed": 98, "arrival_ratio": 0.40, "bark_enabled": True,  "bark_volume": 80, "idle_pose": "stand"},
    "GENTLE":  {"speed": 60, "arrival_ratio": 0.30, "bark_enabled": True,  "bark_volume": 40, "idle_pose": "sit"},
    "CALM":    {"speed": 50, "arrival_ratio": 0.25, "bark_enabled": False, "bark_volume": 0,  "idle_pose": "lie"},
    "PLAYFUL": {"speed": 98, "arrival_ratio": 0.45, "bark_enabled": True,  "bark_volume": 80, "idle_pose": "stand"},
    "SOCIAL":  {"speed": 70, "arrival_ratio": 0.35, "bark_enabled": False, "bark_volume": 0,  "idle_pose": "stand"},
}

BEHAVIOR_DEFAULT_MODE = "ACTIVE"

# ── Part B: Semantic Exploration ──────────────────────────────────
# VLM guides navigation when owner is lost (EXPLORE state).
VLM_EXPLORE_STEP_COUNT    = 3       # steps per VLM-directed movement
VLM_EXPLORE_SPEED         = 70      # cautious explore speed
VLM_EXPLORE_BACK_STEPS    = 8       # step_count for 180° turn (BACK direction)
VLM_EXPLORE_DIRECTION_TIMEOUT_S = 5.0  # expire stale explore direction
