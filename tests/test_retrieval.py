from __future__ import annotations

from app.db.vector_store import StoredChunk, VectorStore
from app.rag.retriever import HybridRetriever
from app.services.embedding import SimpleHashEmbedding


def test_retriever_returns_relevant_chunk() -> None:
    store = VectorStore()
    embedder = SimpleHashEmbedding()

    store.add(
        StoredChunk(
            chunk_id="a",
            text="Aspirin is commonly used as an analgesic.",
            metadata={"source": "test"},
            embedding=embedder.embed("Aspirin is commonly used as an analgesic."),
        )
    )
    store.add(
        StoredChunk(
            chunk_id="b",
            text="Metformin is a medication used to treat type 2 diabetes.",
            metadata={"source": "test"},
            embedding=embedder.embed("Metformin is a medication used to treat type 2 diabetes."),
        )
    )

    retriever = HybridRetriever(store=store, embedder=embedder)
    hits = retriever.retrieve("What is metformin used for?", k=1)
    assert len(hits) == 1
    assert "metformin" in hits[0].text.lower()
