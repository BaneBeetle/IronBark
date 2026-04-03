from __future__ import annotations

import time
import threading
from queue import Queue, Empty, Full
from dataclasses import dataclass, field
from typing import List, Optional, Any

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from perception import YOLODetector, PersonDetection
from perception import VLMReasoner, VLMResponse
from perception import FaceRecognizer, FaceMatch


@dataclass
class PerceptionResult:
    detections:   List[PersonDetection]
    face_matches: List[FaceMatch]
    vlm_response: Optional[VLMResponse]
    frame_id:     int
    latency_ms:   float


class PerceptionPipeline:
    def __init__(self, config: Any):
        if hasattr(config, "__dict__") and not isinstance(config, dict):
            cfg = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        else:
            cfg = config

        print("[PerceptionPipeline] Initializing perception stack...")

        # NOTE: No hardcoded device — YOLODetector auto-detects MPS/CUDA/CPU
        self.yolo = YOLODetector(
            model_path=cfg.get("YOLO_MODEL", "yolo11n.pt"),
            conf_threshold=cfg.get("YOLO_CONF_THRESHOLD", 0.5),
        )

        self.face_recognizer = FaceRecognizer(
            embedding_path=cfg.get("OWNER_EMBEDDING_PATH", "data/owner_embedding.npy"),
            threshold=cfg.get("FACE_THRESHOLD", 0.6),
        )

        self.vlm = VLMReasoner(
            model=cfg.get("VLM_MODEL", "llava:7b"),
            host=cfg.get("VLM_HOST", "http://localhost:11434"),
        )

        self._vlm_queue: Queue = Queue(maxsize=1)
        self._latest_vlm_response: Optional[VLMResponse] = None
        self._vlm_lock = threading.Lock()
        self._vlm_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_id = 0
        print("[PerceptionPipeline] All components initialized.")

    def start(self):
        if self._running:
            return
        self._running = True
        self._vlm_thread = threading.Thread(target=self._vlm_worker, name="VLM-Worker", daemon=True)
        self._vlm_thread.start()
        print("[PerceptionPipeline] VLM background thread started.")

    def stop(self):
        self._running = False
        try:
            self._vlm_queue.put_nowait(None)
        except Full:
            pass
        if self._vlm_thread:
            self._vlm_thread.join(timeout=5.0)

    def _vlm_worker(self):
        print("[VLM-Worker] Background thread started, waiting for frames...")
        while self._running:
            try:
                item = self._vlm_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            frame, detections = item
            try:
                vlm_response = self.vlm.reason(frame, detections)
                with self._vlm_lock:
                    self._latest_vlm_response = vlm_response
                print(f"[VLM-Worker] {vlm_response.latency_ms:.0f}ms | {vlm_response.navigation_hint[:60]}...")
            except Exception as e:
                print(f"[VLM-Worker] Error: {e}")

    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        t_start = time.perf_counter()
        self._frame_id += 1

        detections = self.yolo.detect(frame)

        face_matches = []
        if detections:
            faces = self.face_recognizer.app.get(frame)
            for detection in detections:
                best_match = FaceMatch(is_owner=False, confidence=0.0, embedding=None)
                best_similarity = -1.0
                x1, y1, x2, y2 = detection.bbox
                pw, ph = x2 - x1, y2 - y1
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    face_cx, face_cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    margin_x, margin_y = int(pw * 0.1), int(ph * 0.1)
                    if (x1 - margin_x <= face_cx <= x2 + margin_x and
                        y1 - margin_y <= face_cy <= y2 + margin_y):
                        if self.face_recognizer.owner_embedding is not None:
                            similarity = self.face_recognizer._cosine_similarity(
                                face.embedding, self.face_recognizer.owner_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = FaceMatch(
                                    is_owner=similarity >= self.face_recognizer.threshold,
                                    confidence=float(similarity), embedding=face.embedding)
                        else:
                            best_match = FaceMatch(is_owner=False, confidence=0.0, embedding=face.embedding)
                            break
                face_matches.append(best_match)

        try:
            self._vlm_queue.put_nowait((frame.copy(), detections))
        except Full:
            pass

        with self._vlm_lock:
            vlm_response = self._latest_vlm_response

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return PerceptionResult(detections=detections, face_matches=face_matches,
                                vlm_response=vlm_response, frame_id=self._frame_id, latency_ms=latency_ms)

    def draw_overlay(self, frame, result):
        display = frame.copy()
        h, w = display.shape[:2]

        for detection, face_match in zip(result.detections, result.face_matches):
            x1, y1, x2, y2 = detection.bbox
            if self.face_recognizer.owner_embedding is None:
                color, label = (0, 255, 255), f"Person {detection.confidence:.0%}"
            elif face_match.is_owner:
                color, label = (0, 200, 0), f"OWNER {face_match.confidence:.0%}"
            else:
                color, label = (0, 0, 220), f"Stranger {face_match.confidence:.0%}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if result.vlm_response:
            hint = f"VLM: {result.vlm_response.navigation_hint}"
            hint = hint[:90] + "..." if len(hint) > 90 else hint
            cv2.putText(display, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        fps_approx = 1000.0 / max(result.latency_ms, 1.0)
        stats = f"Frame #{result.frame_id} | Fast path: {result.latency_ms:.1f}ms (~{fps_approx:.0f}fps) | Persons: {len(result.detections)}"
        cv2.putText(display, stats, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return display
