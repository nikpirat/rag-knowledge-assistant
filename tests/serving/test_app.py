"""Tests for the FastAPI serving application.

Uses fake embed_query/search/generate_answer functions — no live Ollama
or populated Qdrant collection needed, same "isolate from external
infrastructure" principle used throughout this project's test suites.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag_knowledge_assistant.generation.ollama_generator import GenerationResult
from rag_knowledge_assistant.retrieval.store import SearchResult
from rag_knowledge_assistant.serving.app import create_app


def _fake_embed_query(text: str, **kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


def _fake_search(
    client: QdrantClient, collection: str, vector: list[float], **kwargs: object
) -> list[SearchResult]:
    return [
        SearchResult(
            text="AWS Budgets lets you configure cost alerts.",
            section_heading="COST01-BP05",
            source_document="cost-optimization-pillar.pdf",
            page_number=8,
            score=0.85,
        )
    ]


def _fake_generate_answer(
    question: str, results: list[SearchResult], **kwargs: object
) -> GenerationResult:
    return GenerationResult(
        answer="You can use AWS Budgets to configure cost alerts [1].",
        cited_sources=[results[0]] if results else [],
    )


def _make_app(tmp_path: Path) -> FastAPI:
    return create_app(
        qdrant_client_builder=lambda: QdrantClient(path=str(tmp_path / "qdrant")),
        embed_query_fn=_fake_embed_query,
        search_fn=_fake_search,
        generate_answer_fn=_fake_generate_answer,
    )


class TestHealthEndpoint:
    def test_returns_ok(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAskEndpoint:
    def test_returns_answer_and_sources(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "How do I set cost alerts?"})

        assert response.status_code == 200
        body = response.json()
        assert "AWS Budgets" in body["answer"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["section_heading"] == "COST01-BP05"
        assert body["sources"][0]["source_document"] == "cost-optimization-pillar.pdf"

    def test_empty_question_rejected(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            response = client.post("/ask", json={"question": ""})

        assert response.status_code == 422

    def test_missing_question_field_rejected(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        with TestClient(app) as client:
            response = client.post("/ask", json={})

        assert response.status_code == 422
