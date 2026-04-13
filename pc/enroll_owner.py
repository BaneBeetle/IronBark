"""
enroll_owner.py — IronBark Owner Enrollment (Multi-Distance Gallery)

Captures face embeddings at 3 distances (close/medium/far) from the
ground-level camera. Stores the full gallery (Nx512) — no averaging.
Max-of-gallery matching preserves distance-variant information that
a single mean embedding destroys.

Also sends a head-pitch command to the dog so the camera looks up
slightly (matching follower.py FOLLOW state), which is the angle
the dog will actually use during operation.

Usage:
    python pc/enroll_owner.py
    SPACE = start / advance to next stage
    ESC   = quit
"""

import sys
import time
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from perception import YOLODetector, FaceRecognizer, ReIDRecognizer


# ── Capture state machine ───────────────────────────────────────
STATE_IDLE      = "idle"
STATE_READY     = "ready"       # showing distance instruction, waiting for SPACE
STATE_CAPTURING = "capturing"
STATE_DONE      = "done"
STATE_FAILED    = "failed"

# ── HUD styling ─────────────────────────────────────────────────
GREEN  = (0, 255, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
GRAY   = (180, 180, 180)
DARK   = (40, 40, 40)
BLACK  = (0, 0, 0)

# Per-stage instructions (close/medium are head-only, far allows body turns)
STAGE_INSTRUCTIONS = [
    "Turn HEAD only L/R, body facing camera",
    "Turn HEAD only L/R, body facing camera",
    "Head L/R + slight body turns (~15 deg)",
]


def _draw_text_with_bg(frame, text, pos, scale, color, thickness=2, pad=6):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), BLACK, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(frame, x, y, w, h, progress, color, label=None):
    """Draw a labelled progress bar with dark background."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    filled = int(w * min(progress, 1.0))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)
    if label:
        cv2.putText(frame, label, (x + 4, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)


def _send_head_pitch(cmd_sock, pitch):
    """Send a head-pitch command to the dog (matches follower.py FollowCommand format)."""
    cmd = {
        "action": "stop", "speed": 0, "step_count": 0,
        "head_yaw": 0.0, "head_pitch": pitch,
        "head_mode": "remote", "bark": False, "bark_volume": 0,
    }
    cmd_sock.send(json.dumps(cmd).encode("utf-8"))


def main():
    print("=" * 60)
    print("  IronBark — Multi-Distance Owner Enrollment")
    print("=" * 60)

    ctx = zmq.Context()

    # Video stream from Pi
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://{config.PC_IP}:{config.ZMQ_PORT}")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    print(f"[enroll_owner] ZMQ PULL bound on {config.PC_IP}:{config.ZMQ_PORT}")

    # Command socket to set head pitch (same as follower.py)
    cmd_sock = ctx.socket(zmq.PUB)
    cmd_sock.bind(f"tcp://{config.PC_IP}:{config.CMD_PORT}")
    print(f"[enroll_owner] Command PUB bound on {config.PC_IP}:{config.CMD_PORT}")

    # Let PUB socket connect before sending
    time.sleep(0.5)

    # Pitch head up to match FOLLOW state — this is the angle
    # the camera will be at during actual operation.
    head_pitch = getattr(config, "HEAD_DEFAULT_PITCH", 15)
    _send_head_pitch(cmd_sock, head_pitch)
    print(f"[enroll_owner] Head pitch set to {head_pitch} (matching FOLLOW state)")

    detector = YOLODetector(model_path=config.YOLO_MODEL, conf_threshold=0.4)
    recognizer = FaceRecognizer(
        embedding_path=config.OWNER_EMBEDDING_PATH,
        gallery_path=config.OWNER_GALLERY_PATH,
        threshold=config.FACE_THRESHOLD,
    )
    reid = ReIDRecognizer(
        gallery_path=config.OWNER_BODY_GALLERY_PATH,
        threshold=config.REID_THRESHOLD,
    )

    window_name = "IronBark — Owner Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    header_size = struct.calcsize("<qI")

    distances       = config.ENROLL_DISTANCES
    samples_per     = config.ENROLL_SAMPLES_PER_STAGE
    timeout_s       = config.ENROLL_TIMEOUT_S
    min_face_px     = config.ENROLL_MIN_FACE_PX
    sample_gap_s    = config.ENROLL_SAMPLE_INTERVAL_S
    total_target    = len(distances) * samples_per

    # YOLO-guided crop: instead of running insightface on the full 640x480
    # frame (where the face is a tiny speck at distance), we crop the YOLO
    # person bbox with padding, then upscale so the face fills the input.
    # This solves both the size problem AND gives RetinaFace a cleaner input
    # with less distracting background.
    CROP_PAD        = 0.3    # 30% padding around YOLO bbox
    CROP_MIN_SIDE   = 640    # upscale crop so shortest side >= this

    state           = STATE_IDLE
    stage_idx       = 0
    all_embeddings  = []          # face embeddings across all stages
    body_crops      = []          # body crops across all stages (for ReID)
    stage_collected = []          # face embeddings, current stage only
    capture_start   = 0.0
    last_sample_t   = 0.0
    status_msg      = ""
    status_until    = 0.0
    ready_entered   = 0.0         # timestamp when READY state was entered

    # Countdown before each stage auto-starts (seconds).
    # First stage is shorter (you're already standing there).
    # Later stages give you time to walk back.
    COUNTDOWN_FIRST_S = 3.0
    COUNTDOWN_NEXT_S  = 6.0

    # Resend head pitch periodically so the dog doesn't drift
    last_pitch_t    = time.time()
    PITCH_RESEND_S  = 3.0

    while True:
        # ── Receive frame ─────────────────────────────────────
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
            _draw_text_with_bg(blank, "Waiting for Pi stream...", (200, 270), 1.0, YELLOW)
            cv2.imshow(window_name, blank)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        # Keep head pitched up
        now = time.time()
        if now - last_pitch_t >= PITCH_RESEND_S:
            _send_head_pitch(cmd_sock, head_pitch)
            last_pitch_t = now

        detections = detector.detect(frame)
        h, w = frame.shape[:2]

        # Draw person bboxes
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)

        # ── State machine ─────────────────────────────────────
        if state == STATE_READY:
            # Auto-advance: countdown then start capturing (no SPACE needed)
            countdown_s = COUNTDOWN_FIRST_S if stage_idx == 0 else COUNTDOWN_NEXT_S
            elapsed_ready = time.time() - ready_entered
            if elapsed_ready >= countdown_s:
                state = STATE_CAPTURING
                stage_collected = []
                capture_start = time.time()
                last_sample_t = 0.0
                dist_label = distances[stage_idx]
                print(f"[enroll_owner] Auto-starting stage {stage_idx + 1}: {dist_label}")

        if state == STATE_CAPTURING:
            elapsed = time.time() - capture_start
            dist_label = distances[stage_idx]

            # Try to capture a sample (rate-limited).
            # Strategy: crop the largest YOLO person detection, pad + upscale
            # it, and run insightface on that crop. This makes the face fill
            # a much larger portion of the detector input vs running on the
            # full frame where a 4ft-away face is a ~40px speck.
            if (time.time() - last_sample_t) >= sample_gap_s and detections:
                # Pick the largest person detection (most likely the owner).
                best_det = max(detections, key=lambda d: d.area)
                bw_raw = best_det.bbox[2] - best_det.bbox[0]
                bh_raw = best_det.bbox[3] - best_det.bbox[1]
                bx1, by1, bx2, by2 = best_det.bbox
                bw, bh = bx2 - bx1, by2 - by1

                # Pad the crop generously so the face isn't clipped
                pad_x = int(bw * CROP_PAD)
                pad_y = int(bh * CROP_PAD)
                cx1 = max(0, bx1 - pad_x)
                cy1 = max(0, by1 - pad_y)
                cx2 = min(w, bx2 + pad_x)
                cy2 = min(h, by2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]

                # Upscale so the face is large enough for RetinaFace
                crop_h, crop_w = crop.shape[:2]
                scale = max(1.0, CROP_MIN_SIDE / min(crop_w, crop_h))
                if scale > 1.0:
                    crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale,
                                      interpolation=cv2.INTER_LINEAR)

                faces = recognizer.app.get(crop)
                if not faces:
                    print(f"  [detect] YOLO person {bw_raw}x{bh_raw}px, "
                          f"crop {crop_w}x{crop_h} -> {int(crop_w*scale)}x{int(crop_h*scale)}, "
                          f"insightface: 0 faces")
                valid_face = None
                for f in faces:
                    fx1, fy1, fx2, fy2 = f.bbox.astype(int)
                    # Size check in original-frame pixels
                    fw = (fx2 - fx1) / scale
                    fh = (fy2 - fy1) / scale
                    if fw >= min_face_px and fh >= min_face_px:
                        valid_face = f
                        break

                # Always capture body crop (works at any distance, no face needed).
                # Use the raw YOLO bbox from the original frame — no padding,
                # since ReID models expect tight person crops.
                body_crop = frame[max(0,by1):min(h,by2), max(0,bx1):min(w,bx2)].copy()
                if body_crop.size > 0:
                    body_crops.append(body_crop)

                if valid_face is not None:
                    stage_collected.append(valid_face.embedding)
                    last_sample_t = time.time()
                    # Map face bbox back to original frame coords for display
                    fb = valid_face.bbox / scale
                    fx1 = int(fb[0]) + cx1
                    fy1 = int(fb[1]) + cy1
                    fx2 = int(fb[2]) + cx1
                    fy2 = int(fb[3]) + cy1
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), GREEN, 3)
                else:
                    # No face but body still captured — show a blue bbox
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 150, 0), 2)
                    last_sample_t = time.time()  # still rate-limit attempts

            # Stage complete?
            if len(stage_collected) >= samples_per:
                all_embeddings.extend(stage_collected)
                stage_collected = []
                stage_idx += 1
                print(f"[enroll_owner] Stage {stage_idx}/{len(distances)} done, "
                      f"total embeddings: {len(all_embeddings)}")

                if stage_idx >= len(distances):
                    # All stages done — save face + body galleries
                    try:
                        recognizer.enroll_owner(all_embeddings)
                        reid.enroll_owner(body_crops)
                        n_face = len(all_embeddings)
                        n_body = len(body_crops)
                        state = STATE_DONE
                        status_msg = f"ENROLLED! {n_face} face + {n_body} body embeddings"
                        status_until = time.time() + 3.0
                        print(f"[enroll_owner] Saved: {n_face} face, {n_body} body")
                    except Exception as e:
                        state = STATE_FAILED
                        status_msg = f"Save failed: {e}"
                        status_until = time.time() + 3.0
                        print(f"[enroll_owner] Save failed: {e}")
                else:
                    # Advance to next distance stage
                    state = STATE_READY
                    ready_entered = time.time()
            elif elapsed > timeout_s:
                # Timeout — flush partial captures, advance to next stage.
                # A stage with 0 captures is fine (e.g., FAR where the face
                # angle is too extreme for RetinaFace). We save whatever
                # gallery we've built from earlier stages.
                got = len(stage_collected)
                if stage_collected:
                    all_embeddings.extend(stage_collected)
                stage_collected = []
                stage_idx += 1
                print(f"[enroll_owner] Stage {stage_idx}/{len(distances)} timed out "
                      f"({got} samples), total: {len(all_embeddings)}")

                if stage_idx >= len(distances):
                    # All stages attempted — save if we have enough
                    if len(all_embeddings) >= 5 or len(body_crops) >= 5:
                        try:
                            if all_embeddings:
                                recognizer.enroll_owner(all_embeddings)
                            if body_crops:
                                reid.enroll_owner(body_crops)
                            n_face = len(all_embeddings)
                            n_body = len(body_crops)
                            state = STATE_DONE
                            status_msg = f"ENROLLED! {n_face} face + {n_body} body"
                            status_until = time.time() + 3.0
                        except Exception as e:
                            state = STATE_FAILED
                            status_msg = f"Save failed: {e}"
                            status_until = time.time() + 3.0
                    else:
                        state = STATE_FAILED
                        status_msg = f"Too few samples: {len(all_embeddings)} (need 5+)"
                        status_until = time.time() + 3.0
                else:
                    state = STATE_READY
                    ready_entered = time.time()

        # Auto-return to IDLE after status display
        if state in (STATE_DONE, STATE_FAILED) and time.time() > status_until:
            state = STATE_IDLE
            stage_idx = 0
            all_embeddings = []
            body_crops = []
            status_msg = ""

        # ── Draw HUD ──────────────────────────────────────────
        if state == STATE_IDLE:
            _draw_text_with_bg(frame, "SPACE = Start enrollment   ESC = Quit",
                               (10, 35), 0.7, WHITE)
            _draw_text_with_bg(frame, "Camera should be on the dog (ground level, head pitched up)",
                               (10, 72), 0.55, GRAY)

        elif state == STATE_READY:
            dist_label = distances[stage_idx]
            instruction = STAGE_INSTRUCTIONS[min(stage_idx, len(STAGE_INSTRUCTIONS) - 1)]
            countdown_s = COUNTDOWN_FIRST_S if stage_idx == 0 else COUNTDOWN_NEXT_S
            remaining = max(0, countdown_s - (time.time() - ready_entered))

            _draw_text_with_bg(frame, f"Stage {stage_idx + 1}/{len(distances)}:  {dist_label}",
                               (10, 40), 0.85, YELLOW)
            _draw_text_with_bg(frame, instruction,
                               (10, 80), 0.6, WHITE)
            _draw_text_with_bg(frame, f"Starting in {remaining:.0f}s...  (SPACE to skip)",
                               (10, 115), 0.55, GRAY)

            # Overall progress
            overall_progress = len(all_embeddings) / total_target
            _draw_progress_bar(frame, 10, h - 30, w - 20, 18, overall_progress, YELLOW,
                               f"Gallery: {len(all_embeddings)}/{total_target}")

        elif state == STATE_CAPTURING:
            elapsed = time.time() - capture_start
            dist_label = distances[stage_idx]
            instruction = STAGE_INSTRUCTIONS[min(stage_idx, len(STAGE_INSTRUCTIONS) - 1)]
            count = len(stage_collected)

            _draw_text_with_bg(frame, f"CAPTURING  {dist_label}   {count}/{samples_per}   {elapsed:.0f}s",
                               (10, 40), 0.85, GREEN)
            _draw_text_with_bg(frame, instruction,
                               (10, 80), 0.6, WHITE)

            # Stage progress bar
            _draw_progress_bar(frame, 10, h - 58, w - 20, 18, count / samples_per, GREEN,
                               f"Stage {stage_idx + 1}")
            # Overall progress bar
            total_done = len(all_embeddings) + count
            _draw_progress_bar(frame, 10, h - 30, w - 20, 18, total_done / total_target, YELLOW,
                               f"Overall: {total_done}/{total_target}")

        elif state == STATE_DONE:
            _draw_text_with_bg(frame, status_msg, (10, 40), 0.8, GREEN)
        elif state == STATE_FAILED:
            _draw_text_with_bg(frame, status_msg, (10, 40), 0.8, RED)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:  # SPACE
            if state == STATE_IDLE:
                if not detections:
                    status_msg = "No person detected — stand in frame first"
                    state = STATE_FAILED
                    status_until = time.time() + 2.0
                    continue
                # Start first stage — countdown auto-starts capture
                state = STATE_READY
                ready_entered = time.time()
                stage_idx = 0
                all_embeddings = []
                body_crops = []
                print(f"[enroll_owner] Starting multi-distance enrollment: "
                      f"{len(distances)} stages x {samples_per} samples")
            elif state == STATE_READY:
                # SPACE can still skip the countdown if you're impatient
                state = STATE_CAPTURING
                stage_collected = []
                capture_start = time.time()
                last_sample_t = 0.0
                dist_label = distances[stage_idx]
                print(f"[enroll_owner] Capturing stage {stage_idx + 1}: {dist_label}")

    cv2.destroyAllWindows()
    cmd_sock.close()
    sock.close()


if __name__ == "__main__":
    main()
