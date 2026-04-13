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

YOLO_MODEL          = "yolo11s.pt"
YOLO_CONF_THRESHOLD = 0.5

# VLM model — moondream is ~5-9x faster than llava:7b and dramatically
# better at grounding (doesn't hallucinate people/wheelchairs/hospitals
# from furniture). Benchmarked at ~245ms per query vs llava:7b's ~1200ms,
# which lets 2-consecutive-same mode hysteresis commit in ~500ms instead
# of 12-15 seconds. See scratch_benchmark_models.py for numbers.
VLM_MODEL = "moondream"
VLM_HOST  = "http://localhost:11434"

FACE_THRESHOLD       = 0.55     # cosine sim cutoff — raised to reduce false positives on roommates
FACE_CROP_RATIO      = 0.30
OWNER_EMBEDDING_PATH = "data/owner_embedding.npy"   # legacy single-embedding (auto-migrated)
OWNER_GALLERY_PATH   = "data/owner_gallery.npy"     # Nx512 gallery of distance-variant embeddings

# ── Body ReID (OSNet-AIN) ────────────────────────────────────────
OWNER_BODY_GALLERY_PATH = "data/owner_body_gallery.npy"  # Nx512 body appearance gallery
REID_THRESHOLD       = 0.50     # cosine sim cutoff for body ReID — raised to reduce false positives
REID_FACE_FULL_PX    = 80       # face >= this: trust face heavily (close range)
REID_FACE_NONE_PX    = 40       # face <  this: trust body almost entirely

# ── Face recognition smoothing & enrollment ──────────────────────
FACE_SMOOTH_WINDOW   = 5        # rolling window of confidences for is_owner decision
FACE_TRACK_IOU       = 0.3      # IoU threshold for cross-frame detection matching

# Multi-distance enrollment: capture at 3 distances from ground-level camera.
# Each stage collects ENROLL_SAMPLES_PER_STAGE embeddings.
ENROLL_DISTANCES     = ["CLOSE (2 ft)", "MEDIUM (3.5 ft)", "FAR (5 ft)"]
ENROLL_SAMPLES_PER_STAGE = 10   # samples per distance stage (30 total)
ENROLL_TIMEOUT_S     = 12.0     # max seconds per stage
ENROLL_MIN_FACE_PX   = 40       # min face bbox side — lowered for far-distance captures
ENROLL_SAMPLE_INTERVAL_S = 0.25 # min seconds between captured samples (forces pose variety)

# =============================================================================
# Phase 3 - Follow-Me Behavior
# =============================================================================

CMD_PORT = 5556

FRAME_CENTER_X     = 320
STEERING_DEAD_ZONE = 100
TARGET_AREA_RATIO  = 0.12
BODY_TURN_THRESHOLD = 80      # pixels from center before body turns (tighter = less drift)
BODY_TURN_STEP_COUNT = 8      # gait cycles per turn
FORWARD_STEP_COUNT = 8        # gait cycles per forward (was 2; reduces lurch-on-restart drift)
MIN_AREA_RATIO     = 0.02

ULTRASONIC_STOP_CM  = 35
ULTRASONIC_SLOW_CM  = 60

# ── Phase 7: RPLidar C1 (360° obstacle avoidance) ─────────────────
# Replaces the ultrasonic head-sweep scan with continuous 360° LiDAR.
# The LiDAR is body-mounted and head-independent — no more blind spots.
ZMQ_LIDAR_PORT            = 50507           # Pi → Mac: LiDAR scan stream
USE_LIDAR                 = True            # enable LiDAR obstacle avoidance
LIDAR_STALE_S             = 1.0             # ignore scans older than this (seconds)

# Arc definitions (degrees, clockwise: 0=forward, 90=right, 180=behind, 270=left)
LIDAR_FORWARD_ARC_HALF    = 30              # forward = 330° to 30° (60° wide)
LIDAR_OBSTACLE_CM         = 50              # trigger avoidance below this (cm) — react early
LIDAR_DANGER_CM           = 15              # hard stop below this (cm)

# Obstacle avoidance behavior
LIDAR_TURN_STEP_COUNT     = 8               # gait cycles for obstacle turn
LIDAR_TURN_HOLD_S         = 1.5             # seconds to hold turn before re-checking
LIDAR_BACKUP_SPEED        = 80              # speed when backing away
LIDAR_BACKUP_STEP_COUNT   = 6              # gait cycles for backup

# Legacy ultrasonic (kept as fallback if LiDAR unavailable)

# ── 3-phase maneuver (Mac-side obstacle avoidance) ────────────────
MANEUVER_CLEARING_DURATION_S = 3.5   # seconds to walk forward past obstacle edge
MANEUVER_MAX_RETRIES         = 2     # max clearing re-turns before aborting
NAV_FRAME_MAX_AGE_S          = 2.0   # discard ribbon cam frame older than this

BARK_HEAD_PITCH     = 35        # Look all the way up during arrival bark

SEARCH_TIMEOUT_S   = 5.0
EXPLORE_TIMEOUT_S  = 30.0

HEAD_DEFAULT_PITCH = 15

# ── Timing ──────────────────────────────────────────────────────────
CMD_INTERVAL        = 0.5       # seconds between Mac→Pi commands — methodical, not spamming
FOLLOW_GAIT_PAUSE_S = 0.05     # pause between follow-me gaits (was 0.2)

# ── Arrival behavior — bbox area ratio thresholds ───────────────────
# The dog uses bbox size as a distance proxy: big bbox = close, small = far.
# Same concept drones use — no ultrasonic needed, camera-based.
#
# CALIBRATION: Run follower.py and watch the "[FOLLOW] area=X.XXX" log.
#   Stand at "arrived" distance (~3 feet) → note the ratio → set ARRIVED
ARRIVAL_ARRIVED_RATIO = 0.55    # bbox > 55% → arrived! (~1.5 feet instead of ~3 feet)

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
    "ACTIVE":  {"speed": 98, "arrival_ratio": 0.55, "bark_enabled": True,  "bark_volume": 80, "idle_pose": "stand"},
    "GENTLE":  {"speed": 60, "arrival_ratio": 0.45, "bark_enabled": True,  "bark_volume": 40, "idle_pose": "sit"},
    "CALM":    {"speed": 50, "arrival_ratio": 0.40, "bark_enabled": False, "bark_volume": 0,  "idle_pose": "lie"},
    "PLAYFUL": {"speed": 98, "arrival_ratio": 0.60, "bark_enabled": True,  "bark_volume": 80, "idle_pose": "stand"},
    "SOCIAL":  {"speed": 70, "arrival_ratio": 0.50, "bark_enabled": False, "bark_volume": 0,  "idle_pose": "stand"},
}

BEHAVIOR_DEFAULT_MODE = "ACTIVE"

# ── Part B: Semantic Exploration ──────────────────────────────────
# VLM guides navigation when owner is lost (EXPLORE state).
VLM_EXPLORE_STEP_COUNT    = 3       # steps per VLM-directed movement
VLM_EXPLORE_SPEED         = 90      # explore speed (was 70, too slow)
VLM_EXPLORE_BACK_STEPS    = 8       # step_count for 180° turn (BACK direction)
VLM_EXPLORE_DIRECTION_TIMEOUT_S = 5.0  # expire stale explore direction

# Head pitch during SEARCH/EXPLORE (negative = looking down toward the floor).
# During FOLLOW the head is pitched up to track the owner's face, which is
# terrible for navigating — searching for doorways by looking at the ceiling
# is useless. When no owner is visible, pitch the head forward-and-down so
# the VLM can actually see floor, thresholds, and low obstacles.
EXPLORE_HEAD_PITCH        = -10

# ── Dual-camera setup (ribbon cam for navigation) ─────────────────
# The PiDog ships with a ribbon camera on the nose (forward-facing, level
# with the floor). We use it as a dedicated "navigation eye" during
# SEARCH/EXPLORE so the VLM reasons about doorways and obstacles instead
# of whatever angle the webcam's head servo happens to be at.
#
# Wire-up:
#   - Pi runs TWO pi_sender processes: webcam → ZMQ_PORT, ribbon → ZMQ_NAV_PORT
#   - Mac follower.py binds both ports, pulls from each, passes nav frame
#     to pipeline.process_frame(frame, nav_frame=...)
#   - If the ribbon cam stream is unavailable, process_frame silently
#     falls back to the webcam for explore queries — no failure mode.
ZMQ_NAV_PORT              = 50506   # ribbon camera stream (nose, forward)
USE_RIBBON_CAM            = True    # opt-in flag for dual-camera mode
