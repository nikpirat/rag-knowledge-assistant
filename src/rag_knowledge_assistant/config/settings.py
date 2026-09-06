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

    qdrant_path: str = "qdrant_storage"
    chunks_path: str = "data/processed/chunks.jsonl"
    qdrant_collection_name: str = "well_architected_chunks"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    generation_model: str = "qwen3:8b"
    retrieval_top_k: int = 5


settings = Settings()
