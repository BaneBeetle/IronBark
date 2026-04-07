"""
pi_sender.py — Project IronBark Phase 1
=========================================
Runs on the Raspberry Pi 5 (on the PiDog).

Captures frames from the PiDog's camera using picamera2, JPEG-compresses
them, stamps each frame with a microsecond timestamp, and blasts them over
the network to the PC via a ZeroMQ PUSH socket.

The PUSH/PULL pattern is used here because:
  - It's simple and reliable for point-to-point streaming
  - ZeroMQ handles queuing automatically
  - Phase 2 may introduce PUB/SUB if multiple consumers (YOLO, VLM) are needed

Usage:
    python3 pi_sender.py [--pc-ip IP] [--port PORT] [--fps FPS] [--quality Q]

Dependencies (install on Pi):
    pip3 install pyzmq numpy
    # picamera2 should already be on Pi OS — if not: sudo apt install python3-picamera2
"""

import argparse
import time
import struct
import sys

import numpy as np
import zmq

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not found. Install with: pip3 install opencv-python-headless")
    sys.exit(1)

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


# ─── Configuration Defaults ───────────────────────────────────────────────────
# Pull from shared config if available, otherwise fallback.
# Can still be overridden via command-line args (see argparse below).
try:
    import config as _cfg
    DEFAULT_PC_IP = _cfg.PC_IP
    DEFAULT_PORT  = _cfg.ZMQ_PORT
except ImportError:
    # Fallback if config.py / .env are missing on the Pi.
    # Override at runtime with `--pc-ip <ip>`.
    DEFAULT_PC_IP = "127.0.0.1"
    DEFAULT_PORT  = 50505
DEFAULT_WIDTH   = 640
DEFAULT_HEIGHT  = 480
DEFAULT_FPS     = 30
DEFAULT_QUALITY = 60      # JPEG quality 0–100; 60 is fast + good enough for YOLO

# How often to print stats to the terminal (in seconds)
STATS_INTERVAL = 2.0


def build_message(jpeg_bytes: bytes, timestamp_us: int) -> bytes:
    """
    Pack a frame into a binary message.

    Message format (little-endian):
        [8 bytes] int64  — timestamp in microseconds since epoch
        [4 bytes] uint32 — length of JPEG payload
        [N bytes] bytes  — JPEG payload

    Using a simple binary header avoids JSON overhead on every frame.
    Struct format '<qI' = little-endian int64 + uint32.
    """
    header = struct.pack("<qI", timestamp_us, len(jpeg_bytes))
    return header + jpeg_bytes


def setup_camera(width: int, height: int, source: str = "auto",
                  device: int = None):
    """
    Initialize camera for video capture.

    Supports three modes:
      - "usb"      : USB webcam via OpenCV (better quality, easy to angle)
      - "picamera" : Pi ribbon camera via picamera2
      - "auto"     : Try USB webcam first, fall back to picamera2

    Returns:
        (camera_object, camera_type) where camera_type is "usb" or "picamera"
    """
    if source in ("auto", "usb"):
        # If --device flag was passed, skip scanning and use that index directly
        if device is not None:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"\n*** FORCED /dev/video{device} ***")
                print(f"[Camera] Opened at {actual_w}x{actual_h}")
                return cap, "usb"
            else:
                print(f"ERROR: /dev/video{device} could not be opened")
                sys.exit(1)

        # Find USB webcam by scanning /dev/video devices.
        # The ribbon camera claims /dev/video0-7 via CSI, so we need to
        # find the actual USB device. We use v4l2 to identify USB cameras,
        # or brute-force try indices 8-20 (USB cams are usually higher).
        usb_indices = []
        try:
            import subprocess
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5
            )
            # Parse output: USB devices have "usb" in their path
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if "usb" in line.lower():
                    # Next lines contain /dev/videoN paths
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "/dev/video" in lines[j]:
                            idx = int(lines[j].strip().replace("/dev/video", ""))
                            usb_indices.append(idx)
                            break  # only need the first video node per device
        except Exception:
            # Fallback: try indices 8-20 (USB cams on Pi with ribbon cam)
            usb_indices = list(range(8, 21))

        if not usb_indices:
            usb_indices = list(range(8, 21))

        for idx in usb_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"\n*** USB WEBCAM DETECTED at /dev/video{idx} — USING WEBCAM ***")
                    print(f"[Camera] USB webcam opened at {actual_w}x{actual_h}")
                    return cap, "usb"
                cap.release()

    if source in ("auto", "picamera"):
        if not HAS_PICAMERA2:
            print("ERROR: No USB webcam found and picamera2 not available.")
            sys.exit(1)

        cam = Picamera2()
        sensor_modes = cam.sensor_modes
        # Prefer 1296x972 for wide FOV, fall back to largest available
        raw_mode = None
        for mode in sensor_modes:
            if mode["size"] == (1296, 972):
                raw_mode = mode
                break
        if raw_mode is None:
            raw_mode = max(sensor_modes, key=lambda m: m["size"][0] * m["size"][1])

        config = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            raw={"size": raw_mode["size"]},
            buffer_count=2,
        )
        cam.configure(config)
        cam.start()
        time.sleep(1.0)
        print(f"\n*** USING RIBBON CAMERA (picamera2) ***")
        print(f"[Camera] picamera2 started at {width}x{height}")
        return cam, "picamera"

    print("ERROR: No camera found.")
    sys.exit(1)


def setup_zmq_socket(pc_ip: str, port: int) -> zmq.Socket:
    """
    Create and connect a ZeroMQ PUSH socket.

    PUSH connects TO the PC's PULL socket. This means the Pi initiates
    the connection — the PC just listens. This is intentional: if the PC
    restarts, the Pi will reconnect automatically.

    ZeroMQ's PUSH socket will queue frames in memory if the receiver is
    slow. We set a low HWM (high-water mark) so it drops old frames
    instead of buffering indefinitely, keeping latency down.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)

    # High-water mark: max number of messages to queue before dropping.
    # At 30fps, 10 messages = ~333ms of buffering max.
    sock.setsockopt(zmq.SNDHWM, 10)

    # Don't linger on close — drop unsent messages immediately on shutdown
    sock.setsockopt(zmq.LINGER, 0)

    endpoint = f"tcp://{pc_ip}:{port}"
    sock.connect(endpoint)
    print(f"[ZMQ] PUSH socket connected → {endpoint}")
    return sock


def jpeg_encode(frame_rgb: np.ndarray, quality: int) -> bytes:
    """
    Encode an RGB888 numpy array to JPEG bytes using OpenCV.

    OpenCV uses BGR internally, so we flip the channel order before encoding.
    JPEG is lossy but much smaller than raw RGB — critical for WiFi bandwidth.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def run_sender(pc_ip: str, port: int, width: int, height: int,
               target_fps: int, quality: int, source: str = "auto",
               device: int = None) -> None:
    """
    Main capture-and-send loop.

    Captures frames at up to `target_fps`, JPEG-encodes them, stamps them
    with a timestamp, and sends them over ZeroMQ. Prints periodic stats.
    """
    cam, cam_type = setup_camera(width, height, source, device)
    sock = setup_zmq_socket(pc_ip, port)

    frame_interval = 1.0 / target_fps   # seconds between frames

    # Stats tracking
    frames_sent     = 0
    bytes_sent      = 0
    raw_bytes_total = 0
    stats_t0        = time.monotonic()
    last_frame_t    = time.monotonic()

    print(f"[Sender] Streaming at ≤{target_fps}fps, JPEG quality={quality}")
    print(f"[Sender] Press Ctrl+C to stop\n")

    try:
        while True:
            loop_start = time.monotonic()

            # ── Capture ──────────────────────────────────────────────────────
            if cam_type == "usb":
                ret, frame_bgr = cam.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cam.capture_array()  # Returns HxWx3 numpy array (RGB888)

            # ── Timestamp ────────────────────────────────────────────────────
            # Microseconds since epoch — used by the receiver to calculate latency.
            # time.time_ns() // 1000 gives microseconds and avoids float precision issues.
            timestamp_us = time.time_ns() // 1000

            # ── Encode ───────────────────────────────────────────────────────
            jpeg_bytes = jpeg_encode(frame_rgb, quality)

            # ── Pack & Send ──────────────────────────────────────────────────
            message = build_message(jpeg_bytes, timestamp_us)

            try:
                # NOBLOCK: if the SNDHWM queue is full, raise zmq.Again instead
                # of blocking. We'd rather drop a frame than fall behind.
                sock.send(message, zmq.NOBLOCK)
            except zmq.Again:
                # Queue full — receiver is slow. Drop the frame.
                pass

            # ── Stats ────────────────────────────────────────────────────────
            frames_sent     += 1
            bytes_sent      += len(message)
            raw_bytes_total += frame_rgb.nbytes

            now = time.monotonic()
            if now - stats_t0 >= STATS_INTERVAL:
                elapsed   = now - stats_t0
                fps       = frames_sent / elapsed
                mbps      = (bytes_sent * 8) / elapsed / 1e6
                ratio     = raw_bytes_total / bytes_sent if bytes_sent > 0 else 0
                avg_kb    = (bytes_sent / frames_sent / 1024) if frames_sent > 0 else 0

                print(
                    f"[Stats] FPS: {fps:.1f} | "
                    f"Bandwidth: {mbps:.2f} Mbps | "
                    f"Compression: {ratio:.1f}x | "
                    f"Avg frame: {avg_kb:.1f} KB"
                )

                # Reset accumulators
                frames_sent     = 0
                bytes_sent      = 0
                raw_bytes_total = 0
                stats_t0        = now

            # ── FPS Cap ──────────────────────────────────────────────────────
            # Sleep just long enough to hit the target FPS.
            elapsed = time.monotonic() - loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Sender] Shutting down...")
    finally:
        if cam_type == "usb":
            cam.release()
        else:
            cam.stop()
        sock.close()
        print("[Sender] Camera and socket closed. Bye!")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IronBark Pi ZeroMQ Frame Sender")
    parser.add_argument("--pc-ip",   default=DEFAULT_PC_IP,   help="PC IP address")
    parser.add_argument("--port",    default=DEFAULT_PORT,     type=int, help="ZeroMQ port")
    parser.add_argument("--width",   default=DEFAULT_WIDTH,    type=int, help="Capture width")
    parser.add_argument("--height",  default=DEFAULT_HEIGHT,   type=int, help="Capture height")
    parser.add_argument("--fps",     default=DEFAULT_FPS,      type=int, help="Max FPS cap")
    parser.add_argument("--quality", default=DEFAULT_QUALITY,  type=int, help="JPEG quality (0-100)")
    parser.add_argument("--source",  default="auto", choices=["auto", "usb", "picamera"], help="Camera source")
    parser.add_argument("--device",  default=None, type=int, help="Force specific /dev/videoN index (e.g. --device 8)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sender(
        pc_ip   = args.pc_ip,
        port    = args.port,
        width   = args.width,
        height  = args.height,
        target_fps = args.fps,
        quality = args.quality,
        source  = args.source,
        device  = args.device,
    )
