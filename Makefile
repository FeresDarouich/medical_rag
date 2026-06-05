
.PHONY: help install update test run ingest build-index health

PYTHON ?= python
POETRY ?= poetry

INPUT ?= data/raw/example.txt
OUTPUT ?= data/processed/chunks.txt
CHUNKS ?= data/processed/chunks.txt
OUT ?= data/index

help:
	@echo "Targets:"
	@echo "  install       Install deps (poetry install)"
	@echo "  update        Update lockfile and deps (poetry update)"
	@echo "  test          Run tests (pytest)"
	@echo "  run           Run API (uvicorn)"
	@echo "  ingest        Build chunks from raw text (INPUT=... OUTPUT=...)"
	@echo "  build-index   Build FAISS index (CHUNKS=... OUT=...)"
	@echo "  health        Curl health endpoint"

install:
	$(POETRY) install

update:
	$(POETRY) update

test:
	$(POETRY) run $(PYTHON) -m pytest -q

run:
	$(POETRY) run uvicorn app.main:app --reload

ingest:
	$(POETRY) run $(PYTHON) -m ingestion.ingest --input $(INPUT) --output $(OUTPUT)

build-index:
	$(POETRY) run $(PYTHON) -m ingestion.build_index --chunks $(CHUNKS) --out $(OUT)

health:
	@curl -s http://127.0.0.1:8000/health || true
