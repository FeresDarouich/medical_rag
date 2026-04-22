from __future__ import annotations

from dataclasses import dataclass

from app.core.prompts import build_rag_prompt
from app.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str


class AnswerGenerator:
    """Template generator.

    Replace `generate()` with real LLM calls (OpenAI/Azure/etc.).
    """

    def generate(self, question: str, *, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        context = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
        _prompt = build_rag_prompt(question, context)

        if not chunks:
            return GeneratedAnswer(
                answer=(
                    "I don't know based on the provided context. "
                    "Please provide relevant medical documents or references to answer this question."
                )
            )

        top = chunks[0]
        return GeneratedAnswer(answer=f"Based on the provided context, here is what I found: {top.text}")
