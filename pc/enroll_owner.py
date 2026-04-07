"""
enroll_owner.py — IronBark Owner Enrollment (Multi-shot)

Captures N face samples over a few seconds (insightface embeddings),
L2-normalizes them, averages them, and saves a single robust embedding.
This is dramatically more reliable than a single-frame capture.

Usage:
    python pc/enroll_owner.py
    SPACE = start capture
    ESC   = quit
"""

import sys
import time
import struct
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from perception import YOLODetector, FaceRecognizer


# ── Capture state machine ───────────────────────────────────────
STATE_IDLE      = "idle"
STATE_CAPTURING = "capturing"
STATE_DONE      = "done"
STATE_FAILED    = "failed"


def main():
    print("=" * 60)
    print("  IronBark — Multi-shot Owner Enrollment")
    print("=" * 60)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    print(f"[enroll_owner] ZMQ PULL bound on tcp://*:{config.ZMQ_PORT}")

    detector = YOLODetector(model_path=config.YOLO_MODEL, conf_threshold=0.4)
    recognizer = FaceRecognizer(
        embedding_path=config.OWNER_EMBEDDING_PATH,
        threshold=config.FACE_THRESHOLD,
    )

    window_name = "IronBark — Owner Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    header_size = struct.calcsize("<qI")

    target_samples = config.ENROLL_NUM_SAMPLES
    timeout_s      = config.ENROLL_TIMEOUT_S
    min_face_px    = config.ENROLL_MIN_FACE_PX
    sample_gap_s   = config.ENROLL_SAMPLE_INTERVAL_S

    state         = STATE_IDLE
    collected     = []
    capture_start = 0.0
    last_sample_t = 0.0
    status_msg    = ""
    status_until  = 0.0

    while True:
        try:
            raw_msg = sock.recv()
            if len(raw_msg) < header_size:
                continue
            ts, pl = struct.unpack_from("<qI", raw_msg, 0)
            jpeg = raw_msg[header_size:header_size + pl]
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
        except zmq.Again:
            blank = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for Pi stream...", (200, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.imshow(window_name, blank)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        detections = detector.detect(frame)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # ── Capture state machine ──────────────────────────────
        if state == STATE_CAPTURING:
            elapsed = time.time() - capture_start

            # Try to capture a sample (rate-limited so poses get diversity)
            if (time.time() - last_sample_t) >= sample_gap_s:
                faces = recognizer.app.get(frame)
                valid_face = None
                for f in faces:
                    fx1, fy1, fx2, fy2 = f.bbox.astype(int)
                    fw, fh = fx2 - fx1, fy2 - fy1
                    if fw >= min_face_px and fh >= min_face_px:
                        valid_face = f
                        break  # take first valid face

                if valid_face is not None:
                    collected.append(valid_face.embedding)
                    last_sample_t = time.time()
                    # Highlight the captured face
                    fx1, fy1, fx2, fy2 = valid_face.bbox.astype(int)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)

            # Check completion conditions
            if len(collected) >= target_samples:
                try:
                    recognizer.enroll_owner(collected)
                    state = STATE_DONE
                    status_msg = f"ENROLLED! {len(collected)} samples averaged"
                    status_until = time.time() + 2.5
                    print(f"[enroll_owner] Saved averaged embedding from {len(collected)} samples")
                except Exception as e:
                    state = STATE_FAILED
                    status_msg = f"Save failed: {e}"
                    status_until = time.time() + 3.0
                    print(f"[enroll_owner] Save failed: {e}")
                collected = []
            elif elapsed > timeout_s:
                state = STATE_FAILED
                status_msg = f"Timed out: only {len(collected)}/{target_samples} samples"
                status_until = time.time() + 3.0
                print(f"[enroll_owner] Timeout after {elapsed:.1f}s with {len(collected)} samples")
                collected = []

        # Auto-return to IDLE after status display
        if state in (STATE_DONE, STATE_FAILED) and time.time() > status_until:
            state = STATE_IDLE
            status_msg = ""

        # ── Draw HUD ───────────────────────────────────────────
        h, w = frame.shape[:2]
        if state == STATE_IDLE:
            cv2.putText(frame, "SPACE = Start enrollment | ESC = Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        elif state == STATE_CAPTURING:
            elapsed = time.time() - capture_start
            cv2.putText(frame, f"CAPTURING {len(collected)}/{target_samples}  |  {elapsed:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Slowly turn your head left and right",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Progress bar
            bar_w = w - 20
            filled = int(bar_w * len(collected) / target_samples)
            cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 10), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, h - 30), (10 + filled, h - 10), (0, 200, 0), -1)
        elif state == STATE_DONE:
            cv2.putText(frame, status_msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif state == STATE_FAILED:
            cv2.putText(frame, status_msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32 and state == STATE_IDLE:
            if not detections:
                status_msg = "No person detected — stand in frame first"
                state = STATE_FAILED
                status_until = time.time() + 2.0
                continue
            state = STATE_CAPTURING
            collected = []
            capture_start = time.time()
            last_sample_t = 0.0
            print(f"[enroll_owner] Starting capture, target={target_samples} samples")

    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
