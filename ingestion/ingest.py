from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.processing.cleaner import clean_text
from ingestion.processing.chunker import chunk_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest raw documents -> processed chunks")
    parser.add_argument("--input", required=True, help="Path to raw text file")
    parser.add_argument("--output", default="data/processed/chunks.txt", help="Output chunks file")
    args = parser.parse_args()

    raw_path = Path(args.input)
    text = raw_path.read_text(encoding="utf-8")
    text = clean_text(text)
    chunks = chunk_documents(text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
