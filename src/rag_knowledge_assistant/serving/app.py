"""FastAPI serving application for the RAG knowledge assistant.

Uses an app factory (create_app) with injectable dependencies:
tests inject fake embedding/search/generation
functions so no live Ollama or Qdrant is needed to run the test suite.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import cast

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag_knowledge_assistant.config.settings import settings
from rag_knowledge_assistant.embeddings.ollama_embedder import embed_query as real_embed_query
from rag_knowledge_assistant.generation.ollama_generator import GenerationResult
from rag_knowledge_assistant.generation.ollama_generator import (
    generate_answer as real_generate_answer,
)
from rag_knowledge_assistant.retrieval.store import SearchResult
from rag_knowledge_assistant.retrieval.store import search as real_search

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """A question to answer, grounded in the indexed corpus."""

    question: str = Field(..., min_length=1, description="The question to answer")


class SourceResponse(BaseModel):
    """A cited source, returned alongside the answer."""

    section_heading: str
    source_document: str
    page_number: int
    score: float


class AskResponse(BaseModel):
    """The generated answer plus only the sources actually cited."""

    answer: str
    sources: list[SourceResponse]


def _default_qdrant_client_builder() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def create_app(
    qdrant_client_builder: Callable[[], QdrantClient] = _default_qdrant_client_builder,
    embed_query_fn: Callable[..., list[float]] = real_embed_query,
    search_fn: Callable[..., list[SearchResult]] = real_search,
    generate_answer_fn: Callable[..., GenerationResult] = real_generate_answer,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        qdrant_client_builder: producer of the Qdrant client, built once
            at startup. Defaults to a real local-mode client via Settings;
            tests pass a builder pointed at a temp directory.
        embed_query_fn, search_fn, generate_answer_fn: default to the
            real Ollama/Qdrant-backed implementations; tests pass fakes
            so the test suite needs no live Ollama or populated Qdrant
            collection to pass.
            :param generate_answer_fn:
            :param search_fn:
            :param qdrant_client_builder:
            :param embed_query_fn:
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Connecting to Qdrant at %s...", settings.qdrant_url)
        app.state.qdrant_client = qdrant_client_builder()
        yield

    app = FastAPI(
        title="RAG Knowledge Assistant",
        description=(
            "Answers questions over AWS Well-Architected whitepapers, "
            "grounded with citations back to source."
        ),
        lifespan=lifespan,
    )

    def get_qdrant_client(request: Request) -> QdrantClient:
        return cast(QdrantClient, request.app.state.qdrant_client)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(
        payload: AskRequest,
        qdrant_client: QdrantClient = Depends(get_qdrant_client),  # noqa: B008
    ) -> AskResponse:
        query_vector = embed_query_fn(
            payload.question,
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        results = search_fn(
            qdrant_client,
            settings.qdrant_collection_name,
            query_vector,
            limit=settings.retrieval_top_k,
        )
        generation_result = generate_answer_fn(
            payload.question,
            results,
            model=settings.generation_model,
            base_url=settings.ollama_base_url,
        )

        return AskResponse(
            answer=generation_result.answer,
            sources=[
                SourceResponse(
                    section_heading=s.section_heading,
                    source_document=s.source_document,
                    page_number=s.page_number,
                    score=s.score,
                )
                for s in generation_result.cited_sources
            ],
        )

    return app


app = create_app()
