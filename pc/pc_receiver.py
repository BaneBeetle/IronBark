"""Simple ZMQ frame receiver for testing Pi→Mac video stream."""

import struct
import time
import sys
from collections import deque

import numpy as np
import zmq
import cv2

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config

HEADER_FORMAT = "<qI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.bind(f"tcp://{config.PC_IP}:{config.ZMQ_PORT}")
    print(f"[ZMQ] Listening on {config.PC_IP}:{config.ZMQ_PORT}")

    cv2.namedWindow("IronBark Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IronBark Feed", 1280, 720)

    frame_times = deque(maxlen=30)

    try:
        while True:
            try:
                msg = sock.recv()
            except zmq.Again:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for Pi...", (400, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)
                cv2.imshow("IronBark Feed", blank)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            # Drain queue
            while True:
                try:
                    msg = sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if len(msg) < HEADER_SIZE:
                continue
            ts, pl = struct.unpack(HEADER_FORMAT, msg[:HEADER_SIZE])
            jpeg = msg[HEADER_SIZE:]
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            now = time.monotonic()
            frame_times.append(now)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("IronBark Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
