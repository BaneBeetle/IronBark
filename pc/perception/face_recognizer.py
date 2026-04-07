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
                 threshold=0.6, model_pack="buffalo_l"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed.\n"
                "Install with: pip install insightface onnxruntime"
            )

        self.embedding_path = Path(embedding_path)
        self.threshold = threshold
        self.owner_embedding: Optional[np.ndarray] = None

        print(f"[FaceRecognizer] Loading insightface ({model_pack})...")
        self.app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 = CPU. insightface doesn't support MPS natively.
        # CPU is fast enough on Apple Silicon M5 Pro.
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[FaceRecognizer] insightface loaded.")

        if self.embedding_path.exists():
            self.owner_embedding = np.load(str(self.embedding_path))
            print(f"[FaceRecognizer] Loaded owner embedding from {self.embedding_path}")
        else:
            print(f"[FaceRecognizer] No owner embedding found at {self.embedding_path}")
            print("  Run enroll_owner.py to create one.")

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
        L2-normalizes each, averages them, then re-normalizes the result.
        Multi-shot averaging is dramatically more robust than a single capture.
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

        mean = np.mean(normed, axis=0)
        n = np.linalg.norm(mean)
        final = (mean / n) if n > 0 else mean

        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.embedding_path), final)
        self.owner_embedding = final
        print(f"[FaceRecognizer] Owner enrolled from {len(normed)} samples → {self.embedding_path}")
        return final

    def recognize(self, face_crop):
        if self.owner_embedding is None:
            return FaceMatch(is_owner=False, confidence=0.0, embedding=None)
        embedding = self._extract_embedding(face_crop)
        if embedding is None:
            return FaceMatch(is_owner=False, confidence=0.0, embedding=None)
        similarity = self._cosine_similarity(embedding, self.owner_embedding)
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
