"""Centralized, type-validated application settings.

Consolidates values that were previously scattered as per-module default
parameters (DEFAULT_MODEL, DEFAULT_BASE_URL constants in the embeddings/
generation modules) into one place — those module-level defaults still
exist and remain useful for standalone scripts, but the served API
explicitly sources its values from here, the same config-management
discipline established in Project 1.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    chunks_path: str = "data/processed/chunks.jsonl"
    qdrant_collection_name: str = "well_architected_chunks"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    # Upgraded from qwen3:8b after real evaluation data: 8b produced a
    # degenerate "list of questions instead of an answer" failure on 3 of
    # 12 real questions (25%), vs 0 of 12 at 14b — a genuine capacity
    # difference, confirmed independent of the ROCm/ollama backend bug
    # found during Phase 5 (both models were re-tested after that fix;
    # only 8b still showed the failure). generate_answer's own retry
    # safeguard (see generation/ollama_generator.py) is defense-in-depth
    # on top of this, not a substitute for it — no model size fully
    # eliminates the risk.
    generation_model: str = "qwen3:14b"
    judge_model: str = "qwen3:14b"
    retrieval_top_k: int = 5


settings = Settings()
