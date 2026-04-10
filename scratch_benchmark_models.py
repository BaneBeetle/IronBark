"""
Benchmark multiple VLMs on the same frame.

Compares llava:7b, moondream, and llava-llama3:8b on:
  - Latency (wall clock)
  - Situation grounding (should say "empty" for empty room, not hallucinate people)
  - Explore direction parsing (should return valid FORWARD/LEFT/RIGHT/BACK)

Uses the saved test frame at /tmp/ironbark_test_frame.jpg.
Run with: .venv/bin/python scratch_benchmark_models.py
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pc"))
from perception.vlm_reasoner import VLMReasoner, VALID_MODES, VALID_DIRECTIONS


MODELS = ["llava:7b", "moondream", "llava-llama3:8b"]
TEST_FRAME_PATH = "/tmp/ironbark_test_frame.jpg"


def main():
    frame = cv2.imread(TEST_FRAME_PATH)
    if frame is None:
        print(f"FAIL: couldn't load {TEST_FRAME_PATH}")
        return 1
    h, w = frame.shape[:2]
    print(f"[bench] Frame: {w}x{h} (empty-room test frame)")
    print("[bench] Ground truth: YOLO detected 0 people in this frame.")
    print()

    results = {}

    for model in MODELS:
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)
        vlm = VLMReasoner(model=model, host="http://localhost:11434")

        # Warm up — first call has cold-start overhead
        print("[bench] Warming up...")
        try:
            _ = vlm.situation_query(frame, detection_count=0)
        except Exception as e:
            print(f"[bench] Warmup FAILED for {model}: {e}")
            continue

        # ── Situation (with YOLO grounding = 0) ────────────────
        print("\n--- situation_query (YOLO detection_count=0) ---")
        sit_latencies = []
        sit_modes = []
        sit_descriptions = []
        for i in range(3):
            t = time.perf_counter()
            s = vlm.situation_query(frame, detection_count=0)
            ms = (time.perf_counter() - t) * 1000
            sit_latencies.append(ms)
            sit_modes.append(s.mode)
            sit_descriptions.append(s.description)
            valid = "✓" if s.mode in VALID_MODES else "✗"
            print(f"  run {i+1} ({ms:.0f}ms) {valid} MODE={s.mode}")
            print(f"    SCENE: {s.description[:100]!r}")

        # ── Explore ────────────────────────────────────────────
        print("\n--- explore_query ---")
        exp_latencies = []
        exp_dirs = []
        for i in range(3):
            t = time.perf_counter()
            e = vlm.explore_query(frame)
            ms = (time.perf_counter() - t) * 1000
            exp_latencies.append(ms)
            exp_dirs.append(e.direction)
            valid = "✓" if e.direction in VALID_DIRECTIONS else "✗"
            print(f"  run {i+1} ({ms:.0f}ms) {valid} DIR={e.direction}")
            print(f"    REASONING: {e.reasoning[:100]!r}")

        # Judge hallucination: does description contain bad keywords?
        bad_words = ["wheelchair", "hospital", "man sitting", "woman sitting",
                     "person sitting", "medical", "patient", "nurse"]
        hallucinated = any(
            any(bw in d.lower() for bw in bad_words)
            for d in sit_descriptions
        )

        results[model] = {
            "sit_avg_ms": sum(sit_latencies) / len(sit_latencies),
            "sit_modes": sit_modes,
            "sit_unique_modes": len(set(sit_modes)),
            "sit_hallucinated": hallucinated,
            "exp_avg_ms": sum(exp_latencies) / len(exp_latencies),
            "exp_dirs": exp_dirs,
            "exp_valid": all(d in VALID_DIRECTIONS for d in exp_dirs),
        }

        print(f"\n[bench] {model} summary:")
        print(f"  situation avg: {results[model]['sit_avg_ms']:.0f}ms")
        print(f"  situation modes: {sit_modes} (unique={results[model]['sit_unique_modes']})")
        print(f"  hallucinated? {'YES' if hallucinated else 'no'}")
        print(f"  explore avg: {results[model]['exp_avg_ms']:.0f}ms")
        print(f"  explore dirs: {exp_dirs}")
        print()

    # ── Final comparison ───────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'model':<20} {'sit_ms':>10} {'exp_ms':>10} {'hallu':>8} {'consistent':>12}")
    for model in MODELS:
        if model not in results:
            print(f"{model:<20} {'FAILED':>10}")
            continue
        r = results[model]
        consistent = "yes" if r["sit_unique_modes"] == 1 else f"{r['sit_unique_modes']} modes"
        halluc = "YES" if r["sit_hallucinated"] else "no"
        print(f"{model:<20} {r['sit_avg_ms']:>9.0f}ms {r['exp_avg_ms']:>9.0f}ms "
              f"{halluc:>8} {consistent:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
