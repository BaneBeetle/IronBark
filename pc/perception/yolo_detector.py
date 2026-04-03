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
        print(f"[YOLODetector] Model loaded. conf_threshold={conf_threshold}")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID], device=self.device,
        )
        detections = []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            face_crop = self._extract_face_crop(frame, x1, y1, x2, y2)
            detections.append(PersonDetection(
                bbox=(x1, y1, x2, y2), confidence=confidence,
                center=(cx, cy), area=area, face_crop=face_crop,
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
