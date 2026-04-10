"""
Standalone test for Phase 6 — runs without the full follower loop.

1. Pulls one real frame off the Pi ZMQ stream
2. Runs the perception pipeline (YOLO + face) on it
3. Calls situation_query + explore_query directly
4. Verifies parsing returns valid modes/directions
5. Reports timing

This validates everything Phase 6 needs except the actual motor commands.
Run with: .venv/bin/python scratch_test_phase6.py
"""
import sys
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pc"))
import config
from perception_pipeline import PerceptionPipeline
from perception.vlm_reasoner import VLMReasoner, VALID_MODES, VALID_DIRECTIONS


def grab_one_frame(timeout_s=10.0):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.bind(f"tcp://*:{config.ZMQ_PORT}")
    print(f"[test] Listening on tcp://*:{config.ZMQ_PORT} for one frame...")
    raw = sock.recv()
    sock.close()
    ctx.term()

    header_size = struct.calcsize("<qI")
    timestamp_us, payload_len = struct.unpack_from("<qI", raw, 0)
    jpeg = raw[header_size : header_size + payload_len]
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, timestamp_us


def main():
    print("=" * 60)
    print("Phase 6 standalone test")
    print("=" * 60)

    # ── 1. Get a real frame from the Pi ────────────────────────────
    t = time.perf_counter()
    frame, ts_us = grab_one_frame(timeout_s=10.0)
    grab_ms = (time.perf_counter() - t) * 1000
    if frame is None:
        print("FAIL: could not decode frame")
        return 1
    h, w = frame.shape[:2]
    print(f"[test] Got frame {w}x{h} in {grab_ms:.0f}ms")
    cv2.imwrite("/tmp/ironbark_test_frame.jpg", frame)
    print("[test] Saved frame to /tmp/ironbark_test_frame.jpg")

    # ── 2. Run perception pipeline (YOLO + face) ───────────────────
    print("\n--- Perception pipeline ---")
    pipe = PerceptionPipeline(config)
    pipe.start()
    time.sleep(0.5)  # let any background threads spin up

    t = time.perf_counter()
    result = pipe.process_frame(frame)
    process_ms = (time.perf_counter() - t) * 1000
    print(f"[test] process_frame() took {process_ms:.0f}ms")
    print(f"[test] detections: {len(result.detections) if hasattr(result, 'detections') else 'N/A'}")
    if hasattr(result, 'face_matches') and result.face_matches:
        for i, m in enumerate(result.face_matches):
            print(f"[test] face {i}: is_owner={m.is_owner} confidence={m.confidence:.3f}")
    else:
        print("[test] no face matches")

    pipe.stop()

    # ── 3. Test VLM situation_query directly ───────────────────────
    print("\n--- VLM situation_query ---")
    vlm = VLMReasoner(model=config.VLM_MODEL, host=config.VLM_HOST)
    if not vlm.health_check():
        print("FAIL: ollama not reachable")
        return 1

    t = time.perf_counter()
    sit = vlm.situation_query(frame)
    sit_ms = (time.perf_counter() - t) * 1000
    print(f"[test] situation_query took {sit_ms:.0f}ms (latency_ms field: {sit.latency_ms:.0f})")
    print(f"[test] mode: {sit.mode}")
    print(f"[test] description: {sit.description!r}")
    if sit.mode not in VALID_MODES:
        print(f"FAIL: returned mode '{sit.mode}' not in {VALID_MODES}")
        return 1
    print("[test] PASS — mode is valid")

    # ── 4. Test VLM explore_query directly ─────────────────────────
    print("\n--- VLM explore_query ---")
    t = time.perf_counter()
    exp = vlm.explore_query(frame)
    exp_ms = (time.perf_counter() - t) * 1000
    print(f"[test] explore_query took {exp_ms:.0f}ms (latency_ms field: {exp.latency_ms:.0f})")
    print(f"[test] direction: {exp.direction}")
    print(f"[test] reasoning: {exp.reasoning!r}")
    if exp.direction not in VALID_DIRECTIONS:
        print(f"FAIL: returned direction '{exp.direction}' not in {VALID_DIRECTIONS}")
        return 1
    print("[test] PASS — direction is valid")

    # ── 5. Run situation_query 3x to check consistency ─────────────
    print("\n--- situation_query consistency (3 runs on same frame) ---")
    modes = []
    for i in range(3):
        s = vlm.situation_query(frame)
        modes.append(s.mode)
        print(f"  run {i+1}: {s.mode}")
    if len(set(modes)) == 1:
        print(f"[test] PASS — all 3 runs returned {modes[0]} (deterministic at temp 0.1)")
    else:
        print(f"[test] WARN — runs returned {modes} (non-deterministic; hysteresis matters)")

    print("\n" + "=" * 60)
    print("Phase 6 standalone test complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
