from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a medical information assistant. Use ONLY the provided context. "
    "If the context is insufficient, say you don't know and suggest what to provide." 
)


def build_rag_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
    )
