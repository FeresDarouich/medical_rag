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

    def add_many(self, chunks: list[StoredChunk]) -> None:
        """
        add many chunks efficiently.
        """
        if not chunks:
            return
        if self._dimension is None:
            self._dimension = len(chunks[0].embedding)
        for chunk in chunks:
            if len(chunk.embedding) != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch. "
                    f"Expected {self._dimension}, got {len(chunk.embedding)}."
                )
        self._chunks.extend(chunks)

        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[tuple[StoredChunk, float]]:
        if self._index is None:
            return []
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self._dimension}, got {len(query_embedding)}."
            )
        query = np.array(
            [query_embedding],
            dtype=np.float32,
        )
        faiss.normalize_L2(query)
        scores, indices = self._index.search(
            query,
            min(k, len(self._chunks)),
        )
        results: list[tuple[StoredChunk, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0:
                continue
            results.append(
                (
                    self._chunks[int(idx)],
                    float(score),
                )
            )
        return results

    def save(self, directory: str | Path) -> None:
        """
        save metadata + FAISS index
        output:
            directory/
                chunks.json
                vectors.index
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        chunks_file = directory / "chunks.json"
        index_file = directory / "vectors.index"

        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "embedding": c.embedding,
            }
            for c in self._chunks
        ]
        chunks_file.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._index is not None:
            faiss.write_index(
                self._index,
                str(index_file),
            )

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
