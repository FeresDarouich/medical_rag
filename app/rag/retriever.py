from __future__ import annotations

from dataclasses import dataclass

from app.db.vector_store import StoredChunk, VectorStore
from app.services.embedding import SimpleHashEmbedding


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(self, *, store: VectorStore, embedder: SimpleHashEmbedding) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed(query)
        vector_hits = self._store.search(query_embedding, k=k)
        results: list[RetrievedChunk] = []
        for chunk, score in vector_hits:
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                )
            )
        return results

    def upsert_texts(self, texts: list[str], *, source: str = "local") -> None:
        for idx, text in enumerate(texts):
            chunk_id = f"{source}:{idx}"
            embedding = self._embedder.embed(text)
            self._store.add(StoredChunk(chunk_id=chunk_id, text=text, metadata={"source": source}, embedding=embedding))
