# medical-rag

Template scaffold for a medical RAG project:

- `app/`: FastAPI backend + RAG pipeline modules
- `ingestion/`: offline ingestion + index build scripts
- `data/`: raw/processed/index artifacts
- `tests/`: basic unit tests

## Quickstart

1) Create a venv and install deps:

```bash
pip install -r requirements.txt
```

2) (Optional) Build an index from a text file:

```bash
python -m ingestion.ingest --input data/raw/example.txt --output data/processed/chunks.txt
python -m ingestion.build_index --chunks data/processed/chunks.txt --out data/index/vector_store.json
```

3) Run the API:

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /ask` with JSON `{ "question": "...", "top_k": 5 }`

## Notes

This template uses a lightweight deterministic embedder (`SimpleHashEmbedding`) so the wiring and tests run without external services. Replace it with a real embedding model and LLM in `app/services/embedding.py` and `app/rag/generator.py`.
