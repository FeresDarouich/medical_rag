# medical-rag

Template scaffold for a medical RAG project:

- `app/`: FastAPI backend + RAG pipeline modules
- `ingestion/`: offline ingestion + index build scripts
- `data/`: raw/processed/index artifacts
- `tests/`: basic unit tests

## Quickstart

### Poetry (recommended)

1) Install dependencies:

```bash
poetry install
```

2) (Optional) Build an index from a text file:

```bash
poetry run python -m ingestion.ingest --input data/raw/example.txt --output data/processed/chunks.txt
poetry run python -m ingestion.build_index --chunks data/processed/chunks.txt --out data/index
```

3) Run the API:

```bash
poetry run uvicorn app.main:app --reload
```

### Makefile shortcuts (optional)

If you have `make` available (WSL/Git Bash/Linux/macOS):

```bash
make install
make test
make run
make ingest INPUT=data/raw/example.txt
make build-index
```

### pip (fallback)

1) Create a venv and install deps:

```bash
pip install -r requirements.txt
```

2) (Optional) Build an index from a text file:

```bash
python -m ingestion.ingest --input data/raw/example.txt --output data/processed/chunks.txt
python -m ingestion.build_index --chunks data/processed/chunks.txt --out data/index
```

3) Run the API:

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /ask` with JSON `{ "question": "...", "top_k": 5 }`
