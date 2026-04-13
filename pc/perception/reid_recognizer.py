"""
reid_recognizer.py — Body Re-Identification (OSNet-AIN)

Identifies the owner by body appearance (clothing, build, hair) when
the face is not visible. Uses OSNet-AIN-x1.0 from torchreid — the
Adaptive Instance Normalization variant is specifically robust to
domain shift (lighting, camera angle changes).

Gallery-based matching: stores Nx512 body embeddings captured during
enrollment at multiple distances. Matching uses max-of-gallery cosine
similarity (same strategy as FaceRecognizer).

Clothing caveat: body ReID encodes clothing color/texture. If the owner
changes clothes, re-enrollment is needed. Enrollment should happen each
session as a ~15s boot-up ritual.
"""

from __future__ import annotations

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, List


class ReIDRecognizer:
    """Body re-identification using OSNet-AIN-x1.0."""

    # Standard ReID input size (height x width) — all benchmarks use this
    INPUT_SIZE = (256, 128)

    # ImageNet normalization (same as torchreid internals)
    PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    PIXEL_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, gallery_path="data/owner_body_gallery.npy",
                 threshold=0.40, device=None):
        try:
            import torchreid
        except ImportError:
            raise ImportError(
                "torchreid not installed.\n"
                "Install with: pip install torchreid gdown tensorboard"
            )

        self.gallery_path = Path(gallery_path)
        self.threshold = threshold
        self.owner_gallery: Optional[np.ndarray] = None

        # Auto-detect device
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[ReIDRecognizer] Loading OSNet-AIN-x1.0 on {self.device}...")
        self.model = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=1000,  # unused for feature extraction
            loss="softmax",
            pretrained=True,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # Warmup pass to avoid cold-start latency spike
        with torch.no_grad():
            dummy = torch.randn(1, 3, *self.INPUT_SIZE).to(self.device)
            self.model(dummy)
            if self.device == "mps":
                torch.mps.synchronize()
        print("[ReIDRecognizer] OSNet-AIN loaded (~9ms/inference on MPS).")

        if self.gallery_path.exists():
            self.owner_gallery = np.load(str(self.gallery_path))
            print(f"[ReIDRecognizer] Loaded body gallery: "
                  f"{self.owner_gallery.shape[0]} embeddings from {self.gallery_path}")
        else:
            print(f"[ReIDRecognizer] No body gallery at {self.gallery_path}")

    def _preprocess(self, body_crop: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and convert a BGR body crop to a model-ready tensor."""
        # BGR -> RGB, resize to 256x128
        img = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]),
                         interpolation=cv2.INTER_LINEAR)

        # Normalize: float32 [0,1] -> ImageNet mean/std
        img = img.astype(np.float32) / 255.0
        img = (img - self.PIXEL_MEAN) / self.PIXEL_STD

        # HWC -> CHW -> batch dim
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def extract_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract a 512-d L2-normalized body embedding from a BGR crop."""
        if body_crop is None or body_crop.size == 0:
            return None

        tensor = self._preprocess(body_crop)
        with torch.no_grad():
            features = self.model(tensor)

        emb = features.cpu().numpy().flatten()
        n = np.linalg.norm(emb)
        return (emb / n).astype(np.float32) if n > 0 else emb.astype(np.float32)

    def enroll_owner(self, body_crops: List[np.ndarray]) -> np.ndarray:
        """
        Enroll the owner from multiple full-body crops.
        Stores the full gallery (Nx512) — same pattern as FaceRecognizer.
        """
        if not body_crops:
            raise ValueError("No body crops provided for enrollment")

        embeddings = []
        for crop in body_crops:
            emb = self.extract_embedding(crop)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid embeddings extracted from body crops")

        gallery = np.array(embeddings, dtype=np.float32)
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), gallery)
        self.owner_gallery = gallery
        print(f"[ReIDRecognizer] Owner body enrolled: "
              f"{gallery.shape[0]} embeddings -> {self.gallery_path}")
        return gallery

    def match_gallery(self, embedding: np.ndarray) -> float:
        """Max-of-gallery cosine similarity (same logic as FaceRecognizer)."""
        if self.owner_gallery is None:
            return 0.0
        n = np.linalg.norm(embedding)
        if n == 0:
            return 0.0
        query = embedding / n
        # Gallery is already L2-normalized
        sims = self.owner_gallery @ query
        return float(sims.max())

    def recognize(self, body_crop: np.ndarray) -> float:
        """
        End-to-end: extract embedding from crop and match against gallery.
        Returns cosine similarity score (0.0 if no gallery or bad crop).
        """
        if self.owner_gallery is None:
            return 0.0
        emb = self.extract_embedding(body_crop)
        if emb is None:
            return 0.0
        return self.match_gallery(emb)
