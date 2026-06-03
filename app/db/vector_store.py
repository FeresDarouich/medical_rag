from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict
    embedding: list[float]


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._index: faiss.Index | None = None
        self._dimension: int | None = None

    def add(self, chunk: StoredChunk) -> None:
        """
        Add a single chunk
        """
        if self._dimension is None:
            self._dimension = len(chunk.embedding)
        elif len(chunk.embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimention mismatch."
                f"Expected {self._dimension}, got {len(chunk.embedding)}"
            )
        self._chunks.append(chunk)
        embedding = np.array(
            [chunk.embedding],
            dtype= np.float32,
        )
        faiss.normalize_L2(embedding) # normalizing
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embedding)

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[tuple[StoredChunk, float]]:
        scored: list[tuple[StoredChunk, float]] = []
        for chunk in self._chunks:
            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "embedding": c.embedding,
            }
            for c in self._chunks
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        if not path.exists():
            return store
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            store.add(
                StoredChunk(
                    chunk_id=str(item["chunk_id"]),
                    text=str(item["text"]),
                    metadata=dict(item.get("metadata") or {}),
                    embedding=[float(x) for x in item["embedding"]],
                )
            )
        return store
