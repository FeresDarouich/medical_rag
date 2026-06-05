from __future__ import annotations

import json
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
                "Embedding dimension mismatch. "
                f"Expected {self._dimension}, got {len(chunk.embedding)}."
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
    def load(
        cls,
        directory: str | Path,
    )-> "VectorStore":
        """
        load VectorStore from disk.
        """
        directory = Path(directory)

        chunks_file = directory / "chunks.json"
        index_file = directory / "vectors.index"

        store = cls()

        if not chunks_file.exists():
            return store
        
        raw = json.loads(chunks_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Invalid chunks.json format in {chunks_file}")

        store._chunks = [
            StoredChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                embedding=list(item["embedding"]),
            )
            for item in raw
        ]
        if store._chunks:
            store._dimension = len(store._chunks[0].embedding)
        if index_file.exists():
            store._index = faiss.read_index(
                str(index_file)
            )
        else:
            store.rebuild_index()
        return store

    def rebuild_index(self)-> None:
        """
        Rebuild FAISS index from stored chunks.
        Useful after manual modifications.
        """
        if not self._chunks:
            self._index = None
            return
        embeddings = np.array(
            [c.embedding for c in self._chunks],
            dtype = np.float32,
        )
        faiss.normalize_L2(embeddings)
        self._dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            self._dimension
        )
        index.add(embeddings)
        self._index = index

    def __len__(self) -> int:
        return len(self._chunks)
    
    def is_empty(self) -> bool:
        return len(self._chunks) == 0