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
from perception import ReIDRecognizer

import config as _cfg


@dataclass
class OwnerMatch:
    """Fused face + body recognition result for one person detection."""
    is_owner:       bool
    confidence:     float          # fused score (0.0-1.0)
    face_score:     float = 0.0   # raw ArcFace gallery score
    body_score:     float = 0.0   # raw OSNet body ReID score
    face_weight:    float = 0.0   # how much face contributed to fusion
    body_weight:    float = 0.0   # how much body contributed to fusion
    face_px:        int   = 0     # face bbox size (px), 0 = no face detected


@dataclass
class PerceptionResult:
    detections:         List[PersonDetection]
    face_matches:       List[FaceMatch]           # raw face-only results (kept for compat)
    owner_matches:      List[OwnerMatch] = field(default_factory=list)  # fused face+body
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
            gallery_path=cfg.get("OWNER_GALLERY_PATH", "data/owner_gallery.npy"),
            threshold=cfg.get("FACE_THRESHOLD", 0.6),
        )

        self.reid = ReIDRecognizer(
            gallery_path=cfg.get("OWNER_BODY_GALLERY_PATH", "data/owner_body_gallery.npy"),
            threshold=cfg.get("REID_THRESHOLD", 0.40),
        )

        # Distance-adaptive fusion thresholds (face bbox size in pixels)
        self._face_full_px = cfg.get("REID_FACE_FULL_PX", 80)
        self._face_none_px = cfg.get("REID_FACE_NONE_PX", 40)

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

        # Face recognition temporal smoothing — keyed by ByteTrack ID
        self._face_tracks: Dict[int, deque] = {}

        # ── ByteTrack owner persistence ──────────────────────
        # Once a track ID is confirmed owner, it stays owner for up to
        # HOLDOVER_FRAMES frames even if recognition momentarily fails.
        # This prevents the dog from entering SEARCH on a single bad frame.
        self._owner_track_id: int = -1          # current owner's ByteTrack ID
        self._owner_track_age: int = 0          # frames since last positive recognition
        self._owner_holdover: int = 15          # keep owner label for this many frames

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

    def _fuse_scores(self, face_match: FaceMatch, body_score: float) -> 'OwnerMatch':
        """
        Distance-adaptive face+body fusion.

        Weights are a function of face bbox size (px):
          - face >= REID_FACE_FULL_PX (80): face_w=0.8, body_w=0.2
          - face <= REID_FACE_NONE_PX (40): face_w=0.0, body_w=1.0
          - between: linear interpolation
          - no face detected at all:        face_w=0.0, body_w=1.0

        This ensures ArcFace dominates when it's reliable (close range,
        big face) and body ReID takes over when face is too small or absent.
        """
        face_score = face_match.confidence
        face_detected = face_match.embedding is not None

        if not face_detected:
            # No face at all — body only
            face_px = 0
            face_w, body_w = 0.0, 1.0
        else:
            # Estimate face size from the embedding (we don't store bbox,
            # but we can infer from confidence — high confidence = big face).
            # For now, use a heuristic: if face_score > threshold, treat as
            # "good enough" face. The real face_px would come from insightface
            # bbox, which we'll thread through in a future pass.
            # Approximate: map confidence to a proxy face size.
            # A face with score >= 0.45 at close range is ~80px+.
            # A face with score ~0.30 at medium range is ~40-60px.
            if face_score >= self.face_recognizer.threshold:
                face_px = self._face_full_px  # treat high-confidence as close
            else:
                face_px = self._face_none_px  # treat low-confidence as far

            # Compute adaptive weights
            t = max(0.0, min(1.0,
                (face_px - self._face_none_px) /
                max(1, self._face_full_px - self._face_none_px)
            ))
            face_w = 0.8 * t          # 0.0 at far, 0.8 at close
            body_w = 1.0 - face_w     # 1.0 at far, 0.2 at close

        fused = face_w * face_score + body_w * body_score
        fused_threshold = (face_w * self.face_recognizer.threshold +
                           body_w * self.reid.threshold)

        return OwnerMatch(
            is_owner=fused >= fused_threshold,
            confidence=float(fused),
            face_score=float(face_score),
            body_score=float(body_score),
            face_weight=float(face_w),
            body_weight=float(body_w),
            face_px=face_px,
        )

    def _smooth_face_matches(self, detections, raw_matches):
        """
        Apply temporal smoothing to per-detection face matches.
        Uses ByteTrack track IDs (instead of hand-rolled IoU) to associate
        detections across frames. Maintains a rolling confidence window per
        track. Only frames where a face was actually detected (embedding is
        not None) update the buffer.
        """
        window    = getattr(_cfg, "FACE_SMOOTH_WINDOW", 5)
        threshold = self.face_recognizer.threshold

        smoothed = []
        active_ids = set()

        for det, fm in zip(detections, raw_matches):
            tid = det.track_id

            if tid >= 0 and tid in self._face_tracks:
                history = self._face_tracks[tid]
            else:
                history = deque(maxlen=window)

            # Only feed the buffer when a face was actually matched
            if fm.embedding is not None:
                history.append(fm.confidence)

            if tid >= 0:
                self._face_tracks[tid] = history
                active_ids.add(tid)

            if len(history) > 0:
                mean_conf = sum(history) / len(history)
                smoothed.append(FaceMatch(
                    is_owner=mean_conf >= threshold,
                    confidence=mean_conf,
                    embedding=fm.embedding,
                ))
            else:
                smoothed.append(fm)

        # Prune stale tracks (IDs not seen this frame)
        stale = [k for k in self._face_tracks if k not in active_ids]
        for k in stale:
            del self._face_tracks[k]

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
                        if self.face_recognizer.owner_gallery is not None:
                            similarity = self.face_recognizer.match_gallery(face.embedding)
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
            self._face_tracks.clear()

        # ── Face + body fusion ────────────────────────────────
        # Run body ReID on every detection and fuse with face scores.
        # Weights are distance-adaptive: face dominates when the face is
        # large (close range), body dominates when face is small/absent.
        owner_matches = []
        h_frame, w_frame = frame.shape[:2]
        for det, fm in zip(detections, face_matches):
            x1, y1, x2, y2 = det.bbox
            # Body ReID — always runs (works from behind, at any distance)
            body_crop = frame[max(0,y1):min(h_frame,y2), max(0,x1):min(w_frame,x2)]
            body_score = self.reid.recognize(body_crop)

            owner_matches.append(self._fuse_scores(fm, body_score))

        # ── ByteTrack owner persistence ──────────────────────
        # If a detection was confirmed owner this frame, lock its track ID.
        # If the owner track ID appears but recognition dipped, hold the
        # label for up to _owner_holdover frames before giving up.
        owner_found_this_frame = False
        for det, om in zip(detections, owner_matches):
            if om.is_owner and det.track_id >= 0:
                self._owner_track_id = det.track_id
                self._owner_track_age = 0
                owner_found_this_frame = True
                break

        if not owner_found_this_frame and self._owner_track_id >= 0:
            self._owner_track_age += 1
            if self._owner_track_age <= self._owner_holdover:
                # Try to find the cached owner track ID in this frame's
                # detections — if present, promote it to owner even though
                # recognition didn't fire this frame.
                for i, det in enumerate(detections):
                    if det.track_id == self._owner_track_id:
                        om = owner_matches[i]
                        if not om.is_owner:
                            owner_matches[i] = OwnerMatch(
                                is_owner=True,
                                confidence=om.confidence,
                                face_score=om.face_score,
                                body_score=om.body_score,
                                face_weight=om.face_weight,
                                body_weight=om.body_weight,
                                face_px=om.face_px,
                            )
                        owner_found_this_frame = True
                        break

            if self._owner_track_age > self._owner_holdover:
                # Track expired — release the lock
                self._owner_track_id = -1
                self._owner_track_age = 0

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
            owner_matches=owner_matches,
            situation_response=situation_response, situation_seq=situation_seq,
            explore_response=explore_response, explore_seq=explore_seq,
            frame_id=self._frame_id, latency_ms=latency_ms)

    def draw_overlay(self, frame, result):
        display = frame.copy()
        h, w = display.shape[:2]

        enrolled = (self.face_recognizer.owner_gallery is not None or
                    self.reid.owner_gallery is not None)

        for i, detection in enumerate(result.detections):
            x1, y1, x2, y2 = detection.bbox

            tid = detection.track_id
            tid_str = f"#{tid}" if tid >= 0 else ""

            if not enrolled:
                color, label = (0, 255, 255), f"Person{tid_str} {detection.confidence:.0%}"
            elif i < len(result.owner_matches):
                om = result.owner_matches[i]
                if om.is_owner:
                    detail = f"F:{om.face_score:.0%} B:{om.body_score:.0%}"
                    color, label = (0, 200, 0), f"OWNER{tid_str} {om.confidence:.0%} ({detail})"
                else:
                    color, label = (0, 0, 220), f"Stranger{tid_str} {om.confidence:.0%}"
            else:
                color, label = (0, 255, 255), f"Person{tid_str}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
