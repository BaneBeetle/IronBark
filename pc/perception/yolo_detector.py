from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")

import torch


@dataclass
class PersonDetection:
    bbox:       Tuple[int, int, int, int]
    confidence: float
    center:     Tuple[int, int] = (0, 0)
    area:       int = 0
    track_id:   int = -1                                    # ByteTrack ID (-1 = untracked)
    face_crop:  Optional[np.ndarray] = field(default=None, repr=False)


class YOLODetector:
    PERSON_CLASS_ID = 0
    FACE_CROP_FRACTION = 0.30
    MIN_FACE_CROP_SIZE = 20

    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.5, device=None):
        self.conf_threshold = conf_threshold
        # Auto-detect device: MPS (Apple Silicon) > CUDA > CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[YOLODetector] Loading model: {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self._tracking = False
        print(f"[YOLODetector] Model loaded. conf_threshold={conf_threshold}")

    def detect(self, frame: np.ndarray, track: bool = True) -> List[PersonDetection]:
        """
        Detect people in a frame. When track=True, uses ByteTrack to assign
        persistent track IDs across frames. Each person keeps their ID even
        through brief occlusions or detection drops.
        """
        if track:
            results = self.model.track(
                frame, verbose=False, conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID], device=self.device,
                persist=True, tracker="bytetrack.yaml",
            )
            self._tracking = True
        else:
            results = self.model(
                frame, verbose=False, conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID], device=self.device,
            )

        detections = []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        has_ids = result.boxes.id is not None

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            track_id = -1
            if has_ids:
                track_id = int(result.boxes.id[i].cpu().numpy())

            face_crop = self._extract_face_crop(frame, x1, y1, x2, y2)
            detections.append(PersonDetection(
                bbox=(x1, y1, x2, y2), confidence=confidence,
                center=(cx, cy), area=area, track_id=track_id,
                face_crop=face_crop,
            ))
        return detections

    def _extract_face_crop(self, frame, x1, y1, x2, y2):
        bbox_height = y2 - y1
        face_y2 = y1 + int(bbox_height * self.FACE_CROP_FRACTION)
        h, w = frame.shape[:2]
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, face_y2)
        if fx2 - fx1 < self.MIN_FACE_CROP_SIZE or fy2 - fy1 < self.MIN_FACE_CROP_SIZE:
            return None
        return frame[fy1:fy2, fx1:fx2].copy()
