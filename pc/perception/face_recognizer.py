from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FaceMatch:
    is_owner:   bool
    confidence: float
    embedding:  Optional[np.ndarray] = field(default=None, repr=False)


class FaceRecognizer:
    def __init__(self, embedding_path="data/owner_embedding.npy",
                 gallery_path="data/owner_gallery.npy",
                 threshold=0.6, model_pack="buffalo_l"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed.\n"
                "Install with: pip install insightface onnxruntime"
            )

        self.embedding_path = Path(embedding_path)
        self.gallery_path = Path(gallery_path)
        self.threshold = threshold

        # Gallery: Nx512 matrix of owner face embeddings across distances/angles.
        # Matching uses max-of-gallery (nearest neighbor) instead of single mean.
        self.owner_gallery: Optional[np.ndarray] = None

        # Legacy single embedding — kept for backward compat / auto-migration
        self.owner_embedding: Optional[np.ndarray] = None

        print(f"[FaceRecognizer] Loading insightface ({model_pack})...")
        self.app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 = CPU. insightface doesn't support MPS natively.
        # CPU is fast enough on Apple Silicon M5 Pro.
        # det_thresh lowered from default 0.5 → 0.3 to catch small/angled
        # faces at operating distance (ground-level cam looking up at 4-8ft).
        # RetinaFace is still accurate at 0.3 — false positives don't appear
        # until ~0.1. The 0.5 default silently drops faces scoring 0.35-0.45
        # which is exactly what a 40-60px upward-angled face produces.
        self.app.prepare(ctx_id=-1, det_thresh=0.3, det_size=(640, 640))
        print("[FaceRecognizer] insightface loaded (det_thresh=0.3).")

        self._load_owner_data()

    def _load_owner_data(self):
        """Load gallery (preferred) or legacy single embedding, auto-migrating if needed."""
        if self.gallery_path.exists():
            self.owner_gallery = np.load(str(self.gallery_path))
            # Derive a single embedding for any code that still reads owner_embedding
            self.owner_embedding = self._gallery_mean(self.owner_gallery)
            print(f"[FaceRecognizer] Loaded owner gallery: {self.owner_gallery.shape[0]} embeddings from {self.gallery_path}")
        elif self.embedding_path.exists():
            # Auto-migrate legacy single embedding → 1-row gallery
            legacy = np.load(str(self.embedding_path))
            self.owner_embedding = legacy
            self.owner_gallery = legacy.reshape(1, -1)
            print(f"[FaceRecognizer] Migrated legacy embedding → 1-row gallery from {self.embedding_path}")
        else:
            print(f"[FaceRecognizer] No owner data found at {self.gallery_path}")
            print("  Run enroll_owner.py to create one.")

    @staticmethod
    def _gallery_mean(gallery: np.ndarray) -> np.ndarray:
        """L2-normalized mean of gallery embeddings."""
        mean = gallery.mean(axis=0)
        n = np.linalg.norm(mean)
        return (mean / n) if n > 0 else mean

    def _extract_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None
        faces = self.app.get(face_crop)
        if not faces:
            return None
        return faces[0].embedding

    def enroll_owner(self, embeddings):
        """
        Enroll the owner from N pre-computed insightface embeddings.
        Stores the full gallery (L2-normalized individually) — no averaging.
        Max-of-gallery matching preserves distance-variant information.
        """
        if not embeddings:
            raise ValueError("No embeddings provided for enrollment")

        normed = []
        for e in embeddings:
            n = np.linalg.norm(e)
            if n > 0:
                normed.append(e / n)

        if not normed:
            raise ValueError("All embeddings have zero norm")

        gallery = np.array(normed, dtype=np.float32)

        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), gallery)
        self.owner_gallery = gallery
        self.owner_embedding = self._gallery_mean(gallery)
        print(f"[FaceRecognizer] Owner enrolled: {gallery.shape[0]} embeddings → {self.gallery_path}")
        return gallery

    def match_gallery(self, embedding: np.ndarray) -> float:
        """
        Max-of-gallery cosine similarity.
        Returns the highest similarity between the query embedding and any
        gallery embedding. This preserves distance/angle-variant information
        that averaging destroys.
        """
        if self.owner_gallery is None:
            return 0.0
        # Normalize query
        n = np.linalg.norm(embedding)
        if n == 0:
            return 0.0
        query = embedding / n
        # Batch cosine sim: gallery is already L2-normalized
        sims = self.owner_gallery @ query
        return float(sims.max())

    def recognize(self, face_crop):
        if self.owner_gallery is None:
            return FaceMatch(is_owner=False, confidence=0.0, embedding=None)
        embedding = self._extract_embedding(face_crop)
        if embedding is None:
            return FaceMatch(is_owner=False, confidence=0.0, embedding=None)
        similarity = self.match_gallery(embedding)
        confidence = float(max(0.0, similarity))
        is_owner = confidence >= self.threshold
        return FaceMatch(is_owner=is_owner, confidence=confidence, embedding=embedding)

    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
