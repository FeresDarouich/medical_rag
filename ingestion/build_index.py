from __future__ import annotations

import argparse
from pathlib import Path

from app.db.vector_store import StoredChunk, VectorStore
from app.services.embedding import SimpleHashEmbedding


def main() -> int:
    parser = argparse.ArgumentParser(description="Build vector index from processed chunks")
    parser.add_argument("--chunks", default="data/processed/chunks.txt", help="Chunks file")
    parser.add_argument("--out", default="data/index/vector_store.json", help="Output index JSON")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    raw = chunks_path.read_text(encoding="utf-8")
    texts = [c.strip() for c in raw.split("\n\n---\n\n") if c.strip()]

    embedder = SimpleHashEmbedding()
    store = VectorStore()
    for idx, text in enumerate(texts):
        store.add(
            StoredChunk(
                chunk_id=f"processed:{idx}",
                text=text,
                metadata={"source": "processed"},
                embedding=embedder.embed(text),
            )
        )

    store.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
