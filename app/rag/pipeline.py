from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from app.core.config import get_settings
from app.core.safety import redact_phi
from app.db.vector_store import VectorStore
from app.rag.generator import AnswerGenerator
from app.rag.postprocess import postprocess_answer
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.services.embedding import SimpleHashEmbedding
from app.services.language import detect_language, translate


class Citation(BaseModel):
    chunk_id: str
    score: float
    metadata: dict


class RAGResult(BaseModel):
    answer: str
    citations: list[Citation]
    language: str


@dataclass(frozen=True)
class PipelineDeps:
    store: VectorStore
    embedder: SimpleHashEmbedding
    retriever: HybridRetriever
    generator: AnswerGenerator

    @classmethod
    def default(cls) -> "PipelineDeps":
        settings = get_settings()
        store = VectorStore.load(settings.vector_store_path)
        embedder = SimpleHashEmbedding()
        retriever = HybridRetriever(store=store, embedder=embedder)
        generator = AnswerGenerator()
        return cls(store=store, embedder=embedder, retriever=retriever, generator=generator)


class RAGPipeline:
    def __init__(self, deps: PipelineDeps) -> None:
        self._deps = deps

    def run(self, question: str, *, top_k: int = 5) -> RAGResult:
        safe_question = redact_phi(question)
        lang = detect_language(safe_question)

        normalized_question = safe_question
        if lang != "en":
            normalized_question = translate(safe_question, target_language="en")

        chunks: list[RetrievedChunk] = self._deps.retriever.retrieve(normalized_question, k=top_k)
        generated = self._deps.generator.generate(normalized_question, chunks=chunks)
        answer = postprocess_answer(generated.answer)

        citations = [Citation(chunk_id=c.chunk_id, score=c.score, metadata=c.metadata) for c in chunks]
        return RAGResult(answer=answer, citations=citations, language=lang)
