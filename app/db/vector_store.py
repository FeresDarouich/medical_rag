from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b, strict=False):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict
    embedding: list[float]


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def add(self, chunk: StoredChunk) -> None:
        self._chunks.append(chunk)

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
