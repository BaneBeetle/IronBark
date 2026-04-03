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


def main():
    print("=" * 60)
    print("  IronBark — Owner Enrollment")
    print("=" * 60)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    print(f"[enroll_owner] ZMQ PULL bound on tcp://*:{config.ZMQ_PORT}")

    detector = YOLODetector(model_path=config.YOLO_MODEL, conf_threshold=0.4)
    recognizer = FaceRecognizer(embedding_path=config.OWNER_EMBEDDING_PATH, threshold=config.FACE_THRESHOLD)

    window_name = "IronBark — Owner Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    header_size = struct.calcsize("<qI")

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

        cv2.putText(frame, "SPACE = Enroll | ESC = Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            if not detections:
                print("No person detected!")
                continue
            try:
                embedding = recognizer.enroll_owner(frame)
                print(f"ENROLLED! Shape={embedding.shape}")
            except ValueError as e:
                print(f"Failed: {e}")

    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
