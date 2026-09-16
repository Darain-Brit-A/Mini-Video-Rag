"""Local sentence embeddings for chunks and questions."""

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Lazy-load the small MiniLM model so the app starts quickly."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Normalize vectors so FAISS inner product is cosine similarity."""
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
