"""
pc/bbox_calibrate.py — IronBark Bbox Area Ratio Calibration Tool

Receives the Pi camera stream, runs YOLO + ArcFace, and prints
the owner's bounding box area ratio every frame.

USE THIS TO CALIBRATE config.py thresholds:
  1. Start pi_sender on Pi:  python3 pi_sender.py
  2. Run this on Mac:        python pc/bbox_calibrate.py
  3. Stand at different distances from the dog
  4. Note the area ratio at each distance
  5. Update ARRIVAL_SLOW_RATIO, ARRIVAL_CLOSE_RATIO,
     ARRIVAL_ARRIVED_RATIO in config.py

The area ratio = (bbox_width * bbox_height) / (frame_width * frame_height)
Bigger number = closer to camera. Range is 0.0 to 1.0.
"""

import sys
import time
import struct
from pathlib import Path

import numpy as np
import zmq
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import config
from perception_pipeline import PerceptionPipeline


def decode_frame(raw):
    header_size = struct.calcsize("<qI")
    if len(raw) < header_size:
        return None
    timestamp, payload_len = struct.unpack_from("<qI", raw, 0)
    jpeg_data = raw[header_size:header_size + payload_len]
    return cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    print("=" * 60)
    print("  IronBark — Bbox Area Ratio Calibrator")
    print("=" * 60)
    print()
    print("Stand at different distances and note the area ratio.")
    print("Use these values to set ARRIVAL_*_RATIO in config.py.")
    print()

    # ZMQ receiver — same as follower.py
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    print(f"[Calibrate] PULL socket bound on port {config.ZMQ_PORT}")

    # Perception (YOLO + ArcFace, no VLM needed but it starts anyway)
    pipeline = PerceptionPipeline(config)
    pipeline.start()

    frame_w, frame_h = config.CAMERA_RESOLUTION
    frame_area = frame_w * frame_h

    window = "IronBark Bbox Calibrator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    print("[Calibrate] Waiting for Pi stream...")
    print()

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for Pi stream...",
                            (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (100, 100, 255), 2)
                cv2.imshow(window, waiting)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            frame = decode_frame(raw)
            if frame is None:
                continue

            result = pipeline.process_frame(frame)
            display = pipeline.draw_overlay(frame, result)

            # Find owner bbox and compute area ratio
            owner_found = False
            if result.detections and result.face_matches:
                for i, (det, match) in enumerate(
                        zip(result.detections, result.face_matches)):
                    x1, y1, x2, y2 = det.bbox
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    area_ratio = area / frame_area
                    label = "OWNER" if match.is_owner else "person"

                    # Print every detection's ratio
                    print(f"  [{label}] bbox=({x1},{y1},{x2},{y2}) "
                          f"w={w} h={h} area_ratio={area_ratio:.4f}")

                    if match.is_owner:
                        owner_found = True
                        # Big overlay text for owner
                        zone = "FAR"
                        color = (0, 255, 0)
                        if area_ratio > config.ARRIVAL_ARRIVED_RATIO:
                            zone = "ARRIVED"
                            color = (0, 255, 255)
                        elif area_ratio > config.ARRIVAL_CLOSE_RATIO:
                            zone = "CLOSE"
                            color = (0, 180, 255)
                        elif area_ratio > config.ARRIVAL_SLOW_RATIO:
                            zone = "SLOW"
                            color = (0, 255, 180)

                        ratio_text = f"AREA RATIO: {area_ratio:.4f}  [{zone}]"
                        cv2.putText(display, ratio_text, (10, 440),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Threshold reference
                        ref = (f"Thresholds: SLOW>{config.ARRIVAL_SLOW_RATIO} "
                               f"CLOSE>{config.ARRIVAL_CLOSE_RATIO} "
                               f"ARRIVED>{config.ARRIVAL_ARRIVED_RATIO}")
                        cv2.putText(display, ref, (10, 465),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (180, 180, 180), 1)

            if not owner_found and result.detections:
                # Person detected but not owner
                for det in result.detections:
                    x1, y1, x2, y2 = det.bbox
                    area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                    print(f"  [person] area_ratio={area_ratio:.4f} (not owner)")

            if not result.detections:
                cv2.putText(display, "No person detected", (10, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[Calibrate] Shutting down...")
    finally:
        pipeline.stop()
        sock.close()
        ctx.term()
        cv2.destroyAllWindows()
        print("[Calibrate] Done.")


if __name__ == "__main__":
    main()
