"""Embedding generation via a locally-running Ollama instance
(Qwen3-Embedding-0.6B).

Uses Ollama's current /api/embed endpoint (POST), not the deprecated
/api/embeddings - the request/response schema differs (plural
"embeddings" list, batched input), and many tutorials online still
reference the deprecated one.

Qwen3-Embedding uses ASYMMETRIC embedding: queries need an instruction
prefix ("Instruct: ...\\nQuery: {text}"), documents/passages do not.
Getting this backwards doesn't raise an error - it silently produces
measurably worse retrieval (the model's own documentation cites a 1-5%
quality difference from using task-tailored query instructions), which is
exactly the kind of bug that's easy to ship without checking the source
model's documentation directly.
"""

import logging
from typing import Any, cast

import httpx

from rag_knowledge_assistant.config.settings import settings

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = (
    "Instruct: Given a question about system architecture and best "
    "practices, retrieve relevant passages that answer the question\nQuery:"
)


def _call_ollama_embed(
    inputs: list[str], model: str, base_url: str, timeout: float
) -> list[list[float]]:
    """Low-level call to Ollama's /api/embed endpoint."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": inputs},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    embeddings = data.get("embeddings")
    if embeddings is None:
        raise ValueError(f"Ollama response missing 'embeddings' field: {data}")
    return cast(list[list[float]], embeddings)


def embed_documents(
    texts: list[str],
    model: str = settings.generation_model,
    base_url: str = settings.ollama_base_url,
    timeout: float = 60.0,
) -> list[list[float]]:
    """Embed document/passage text. No instruction prefix — Qwen3-Embedding's
    asymmetric design only requires the prefix on the query side.
    """
    if not texts:
        return []
    return _call_ollama_embed(texts, model=model, base_url=base_url, timeout=timeout)


def embed_query(
    text: str,
    model: str = settings.generation_model,
    base_url: str = settings.ollama_base_url,
    timeout: float = 60.0,
) -> list[float]:
    """Embed a search query, with the instruction prefix Qwen3-Embedding's
    own documentation recommends for retrieval quality.
    """
    prefixed = f"{QUERY_INSTRUCTION} {text}"
    embeddings = _call_ollama_embed([prefixed], model=model, base_url=base_url, timeout=timeout)
    return embeddings[0]
