from __future__ import annotations

from app.db.vector_store import StoredChunk, VectorStore
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import PipelineDeps, RAGPipeline
from app.rag.retriever import HybridRetriever
from app.services.embedding import SimpleHashEmbedding


def test_pipeline_returns_answer_and_citations() -> None:
    store = VectorStore()
    embedder = SimpleHashEmbedding()
    store.add(
        StoredChunk(
            chunk_id="c1",
            text="Hypertension is high blood pressure.",
            metadata={"source": "test"},
            embedding=embedder.embed("Hypertension is high blood pressure."),
        )
    )
    deps = PipelineDeps(
        store=store,
        embedder=embedder,
        retriever=HybridRetriever(store=store, embedder=embedder),
        generator=AnswerGenerator(),
    )
    pipeline = RAGPipeline(deps)

    result = pipeline.run("What is hypertension?", top_k=3)
    assert result.answer
    assert len(result.citations) <= 3
