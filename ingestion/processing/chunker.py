from __future__ import annotations

from app.services.chunking import chunk_text


def chunk_documents(text: str, *, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
