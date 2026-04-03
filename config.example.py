# =============================================================================
# IronBark Configuration — EXAMPLE
# Copy this file to config.py and fill in your values.
# config.py is gitignored so your IPs stay private.
# =============================================================================

# ---- Network ----------------------------------------------------------------
PC_IP    = "CHANGE_ME"          # IP of the Mac (brain) — run `ifconfig` to find it
PI_IP    = "CHANGE_ME"          # IP of the Raspberry Pi (dog)
ZMQ_PORT   = 50505              # ZeroMQ PUSH/PULL port for video stream
MJPEG_PORT = 9000               # MJPEG HTTP stream port (fallback)

# ---- Camera -----------------------------------------------------------------
CAMERA_RESOLUTION = (1280, 720)
JPEG_QUALITY      = 75
TARGET_FPS        = 30

# =============================================================================
# Phase 2 - Perception Stack
# =============================================================================

YOLO_MODEL          = "yolo11n.pt"
YOLO_CONF_THRESHOLD = 0.5

VLM_MODEL = "llava:7b"
VLM_HOST  = "http://localhost:11434"

FACE_THRESHOLD       = 0.35
FACE_CROP_RATIO      = 0.30
OWNER_EMBEDDING_PATH = "data/owner_embedding.npy"

# =============================================================================
# Phase 3 - Follow-Me Behavior
# =============================================================================

CMD_PORT = 5556

FRAME_CENTER_X     = 640
STEERING_DEAD_ZONE = 100
TARGET_AREA_RATIO  = 0.12
MIN_AREA_RATIO     = 0.02

ULTRASONIC_STOP_CM  = 25
ULTRASONIC_SLOW_CM  = 50

SEARCH_TIMEOUT_S   = 5.0
EXPLORE_TIMEOUT_S  = 30.0

HEAD_DEFAULT_PITCH = -30

# =============================================================================
# Remote Control (Teleoperation)
# =============================================================================

REMOTE_CMD_PORT   = 5557        # Mac→Pi: WASD commands
REMOTE_TELEM_PORT = 5558        # Pi→Mac: telemetry (distance, battery, state)
DANGER_DISTANCE   = 15          # cm — ultrasonic auto-stop threshold

