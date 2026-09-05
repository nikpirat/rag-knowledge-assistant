# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system answering questions over
AWS Well-Architected Framework whitepapers — a public stand-in for
internal solution/process design documents. Built entirely on local,
self-hosted, zero-cost infrastructure: local LLM (Ollama), self-hosted
vector database (Qdrant), open-source embeddings.

## Status

Phase 0 complete: project scaffolding, tooling, and CI. No ingestion yet.

## Tech stack

- **Language/tooling:** Python 3.12, uv, ruff, mypy (strict), pytest
- **LLM serving:** Ollama (local)
- **Vector database:** Qdrant (self-hosted)
- **Embeddings:** open-source, local (model TBD in Phase 2)
- **Evaluation:** RAGAS
- **Corpus:** AWS Well-Architected Framework whitepapers (public, official PDFs)

Everything runs self-hosted/local — $0/month infrastructure cost.

## Local setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <your-repo-url>
cd rag-knowledge-assistant
make install
uv run pre-commit install
```

## Common commands

```bash
make check
make lint
make format
make typecheck
make test
```

## Project structure

```
src/rag_knowledge_assistant/
├── config/       # settings/configuration management
├── ingestion/    # PDF download, text extraction, chunking
├── embeddings/   # embedding model wrapper
├── retrieval/    # Qdrant indexing and query
├── generation/   # Ollama prompt construction and generation
├── serving/      # FastAPI app
└── evaluation/   # RAGAS evaluation harness
tests/
```
