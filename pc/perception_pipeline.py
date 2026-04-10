from __future__ import annotations

import time
import threading
from collections import deque
from queue import Queue, Empty, Full
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from perception import YOLODetector, PersonDetection
from perception import VLMReasoner, VLMResponse, SituationResponse, ExploreResponse
from perception import FaceRecognizer, FaceMatch

import config as _cfg


@dataclass
class PerceptionResult:
    detections:         List[PersonDetection]
    face_matches:       List[FaceMatch]
    vlm_response:       Optional[VLMResponse] = None       # legacy, kept for compat
    vlm_seq:            int = 0                             # legacy
    situation_response: Optional[SituationResponse] = None
    situation_seq:      int = 0
    explore_response:   Optional[ExploreResponse] = None
    explore_seq:        int = 0
    frame_id:           int = 0
    latency_ms:         float = 0.0


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
        self._vlm_lock = threading.Lock()
        self._vlm_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_id = 0

        # Phase 6: VLM query routing
        self._vlm_query_type = "situation"  # "situation" or "explore"
        self._last_situation_time = 0.0

        # Phase 6A: Situation awareness
        self._latest_situation: Optional[SituationResponse] = None
        self._situation_seq = 0

        # Phase 6B: Semantic exploration
        self._latest_explore: Optional[ExploreResponse] = None
        self._explore_seq = 0

        # Face recognition temporal smoothing — IoU-tracked rolling buffers
        self._face_tracks: List[Dict[str, Any]] = []

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

    def set_vlm_query_type(self, query_type: str):
        """Switch VLM between 'situation' (FOLLOW) and 'explore' (EXPLORE)."""
        if query_type in ("situation", "explore"):
            self._vlm_query_type = query_type

    def _vlm_worker(self):
        """Background thread: routes frames to situation or explore VLM queries."""
        print("[VLM-Worker] Phase 6 background thread started.")
        while self._running:
            try:
                item = self._vlm_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            # New tuple shape: (frame, query_type, detection_count)
            # detection_count is the number of people YOLO saw in that frame,
            # used to ground the VLM and prevent hallucinations.
            frame, query_type, detection_count = item
            try:
                if query_type == "situation":
                    resp = self.vlm.situation_query(frame, detection_count=detection_count)
                    with self._vlm_lock:
                        self._latest_situation = resp
                        self._situation_seq += 1
                    print(f"[VLM] Situation: MODE={resp.mode} "
                          f"({resp.latency_ms:.0f}ms) | yolo={detection_count} | {resp.description[:60]}")
                elif query_type == "explore":
                    resp = self.vlm.explore_query(frame)
                    with self._vlm_lock:
                        self._latest_explore = resp
                        self._explore_seq += 1
                    print(f"[VLM] Explore: DIR={resp.direction} "
                          f"({resp.latency_ms:.0f}ms) | {resp.reasoning[:60]}")
            except Exception as e:
                print(f"[VLM-Worker] Error: {e}")

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _smooth_face_matches(self, detections, raw_matches):
        """
        Apply temporal smoothing to per-detection face matches.
        Associates detections across frames via IoU and maintains a rolling
        confidence window per track. Only frames where a face was actually
        detected (embedding is not None) update the buffer — this prevents
        single missed-face frames from polluting the smoothed result.
        """
        window     = getattr(_cfg, "FACE_SMOOTH_WINDOW", 5)
        iou_thresh = getattr(_cfg, "FACE_TRACK_IOU", 0.3)
        threshold  = self.face_recognizer.threshold

        new_tracks = []
        smoothed = []
        used_idxs = set()

        for det, fm in zip(detections, raw_matches):
            # Find best matching previous track by IoU (greedy 1-to-1)
            best_idx, best_iou = -1, 0.0
            for i, track in enumerate(self._face_tracks):
                if i in used_idxs:
                    continue
                iou = self._bbox_iou(det.bbox, track["bbox"])
                if iou > best_iou and iou >= iou_thresh:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                history = self._face_tracks[best_idx]["history"]
                used_idxs.add(best_idx)
            else:
                history = deque(maxlen=window)

            # Only feed the buffer when a face was actually matched this frame
            if fm.embedding is not None:
                history.append(fm.confidence)

            new_tracks.append({"bbox": det.bbox, "history": history})

            if len(history) > 0:
                mean_conf = sum(history) / len(history)
                smoothed.append(FaceMatch(
                    is_owner=mean_conf >= threshold,
                    confidence=mean_conf,
                    embedding=fm.embedding,
                ))
            else:
                # Brand-new track with no face yet — pass raw match through
                smoothed.append(fm)

        self._face_tracks = new_tracks
        return smoothed

    def process_frame(self, frame: np.ndarray,
                      nav_frame: Optional[np.ndarray] = None) -> PerceptionResult:
        """
        Run perception on the owner-tracking frame (`frame`) and optionally
        use a second frame (`nav_frame`) for EXPLORE-state VLM queries.

        Intended wiring:
          - `frame`     = webcam (head-mounted, pitched up for face tracking)
          - `nav_frame` = ribbon camera (nose-mounted, level with floor)

        During FOLLOW the webcam is used everywhere. During EXPLORE the
        ribbon cam (if available) is forwarded to the VLM so it can reason
        about doorways and floor-level features instead of the ceiling.
        """
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

        # Temporal smoothing: rolling-window vote across frames per IoU track
        if detections:
            face_matches = self._smooth_face_matches(detections, face_matches)
        else:
            self._face_tracks = []

        # Phase 6: Route VLM queries based on current mode.
        # Every queue item is (frame, query_type, detection_count).
        now = time.time()
        det_count = len(detections)
        if self._vlm_query_type == "situation":
            # Throttle situation queries to every VLM_SITUATION_INTERVAL_S.
            # Situation uses the webcam (owner tracking frame).
            interval = getattr(_cfg, "VLM_SITUATION_INTERVAL_S", 2.5)
            if now - self._last_situation_time >= interval:
                try:
                    self._vlm_queue.put_nowait((frame.copy(), "situation", det_count))
                    self._last_situation_time = now
                except Full:
                    pass
        elif self._vlm_query_type == "explore":
            # Explore: submit as fast as the VLM can handle.
            # Prefer the ribbon cam (nav_frame) since it's level with the
            # floor and sees doorways. Fall back to webcam if unavailable.
            explore_src = nav_frame if nav_frame is not None else frame
            try:
                self._vlm_queue.put_nowait((explore_src.copy(), "explore", det_count))
            except Full:
                pass

        with self._vlm_lock:
            situation_response = self._latest_situation
            situation_seq = self._situation_seq
            explore_response = self._latest_explore
            explore_seq = self._explore_seq

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return PerceptionResult(
            detections=detections, face_matches=face_matches,
            situation_response=situation_response, situation_seq=situation_seq,
            explore_response=explore_response, explore_seq=explore_seq,
            frame_id=self._frame_id, latency_ms=latency_ms)

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

        # Phase 6: Show behavior mode or explore direction
        if result.situation_response:
            sit = result.situation_response
            mode_text = f"Mode: {sit.mode}"
            scene_text = f"Scene: {sit.description[:70]}" if sit.description else ""
            cv2.putText(display, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            if scene_text:
                cv2.putText(display, scene_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if result.explore_response:
            exp = result.explore_response
            exp_text = f"Explore: {exp.direction} | {exp.reasoning[:50]}"
            cv2.putText(display, exp_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

        fps_approx = 1000.0 / max(result.latency_ms, 1.0)
        stats = f"Frame #{result.frame_id} | Fast path: {result.latency_ms:.1f}ms (~{fps_approx:.0f}fps) | Persons: {len(result.detections)}"
        cv2.putText(display, stats, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return display
