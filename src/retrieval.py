"""FAISS vector storage and top-k semantic retrieval."""

import json
from pathlib import Path

import faiss
import numpy as np

from .embeddings import EmbeddingModel


class VectorStore:
    """A tiny in-process FAISS store suitable for one video."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, object]] = []

    def build(self, chunks: list[dict[str, object]]) -> None:
        vectors = self.embedding_model.encode([str(chunk["text"]) for chunk in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        if self.index is None or not self.chunks:
            raise RuntimeError("Process a video before asking a question.")
        question_vector = self.embedding_model.encode([question])
        scores, positions = self.index.search(question_vector, min(top_k, len(self.chunks)))
        results = []
        for score, position in zip(scores[0], positions[0]):
            result = dict(self.chunks[int(position)])
            result["similarity"] = round(float(score), 4)
            results.append(result)
        return results

    def save(self, index_path: Path, chunks_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty vector store.")
        faiss.write_index(self.index, str(index_path))
        chunks_path.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")

    def load(self, index_path: Path, chunks_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
