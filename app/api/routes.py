from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.pipeline import PipelineDeps, RAGPipeline


router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    pipeline = RAGPipeline(PipelineDeps.default())
    result = pipeline.run(payload.question, top_k=payload.top_k)
    return AskResponse(answer=result.answer, citations=[c.model_dump() for c in result.citations])
