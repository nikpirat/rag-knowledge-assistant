"""Tests for rag_knowledge_assistant.generation.ollama_generator.

Uses httpx.MockTransport rather than a real Ollama daemon - same
rationale as the embeddings tests: this verifies our own request
construction and response parsing (including citation extraction) against
Ollama's documented /api/chat contract, not Ollama's own behavior.
"""

import json

import httpx
import pytest

from rag_knowledge_assistant.generation.ollama_generator import generate_answer
from rag_knowledge_assistant.retrieval.store import SearchResult


def _make_result(text: str, heading: str, source: str) -> SearchResult:
    return SearchResult(
        text=text, section_heading=heading, source_document=source, page_number=1, score=0.9
    )


def _mock_chat_response(monkeypatch: pytest.MonkeyPatch, answer_content: str) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": answer_content},
                "done": True,
            },
        )

    mock_transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
    )
    return captured


class TestGenerateAnswer:
    def test_sends_stream_false_and_correct_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _mock_chat_response(monkeypatch, "Some answer [1].")
        results = [_make_result("Content.", "Heading", "doc.pdf")]

        generate_answer("Question?", results, model="qwen3:8b")

        assert captured["body"]["model"] == "qwen3:8b"
        assert captured["body"]["stream"] is False
        assert captured["body"]["think"] is False
        assert len(captured["body"]["messages"]) == 2

    def test_no_repeat_penalty_override_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _mock_chat_response(monkeypatch, "Answer [1].")
        results = [_make_result("Content.", "Heading", "doc.pdf")]

        generate_answer("Question?", results)

        assert "options" not in captured["body"]

    def test_repeat_penalty_is_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _mock_chat_response(monkeypatch, "Answer [1].")
        results = [_make_result("Content.", "Heading", "doc.pdf")]

        generate_answer("Question?", results, repeat_penalty=1.1)

        assert captured["body"]["options"]["repeat_penalty"] == 1.1

    def test_extracts_cited_sources_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_chat_response(monkeypatch, "The answer is X [1], also confirmed by [2].")
        results = [
            _make_result("First doc content.", "Heading A", "doc_a.pdf"),
            _make_result("Second doc content.", "Heading B", "doc_b.pdf"),
            _make_result("Unrelated content.", "Heading C", "doc_c.pdf"),
        ]

        result = generate_answer("Question?", results)

        assert len(result.cited_sources) == 2
        assert result.cited_sources[0].source_document == "doc_a.pdf"
        assert result.cited_sources[1].source_document == "doc_b.pdf"
        assert all(s.source_document != "doc_c.pdf" for s in result.cited_sources)

    def test_ignores_hallucinated_out_of_range_citation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real, observable failure mode: a small local model under a
        strict citation-format instruction can cite a source number that
        doesn't exist (e.g. [7] when only 2 sources were provided). This
        must not raise an IndexError — it should be silently dropped."""
        _mock_chat_response(monkeypatch, "The answer involves [1] and also [7].")
        results = [_make_result("Only content.", "Heading", "doc.pdf")]

        result = generate_answer("Question?", results)

        assert len(result.cited_sources) == 1
        assert result.cited_sources[0].source_document == "doc.pdf"

    def test_no_citations_in_answer_returns_empty_cited_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_chat_response(monkeypatch, "I don't know based on the provided sources.")
        results = [_make_result("Content.", "Heading", "doc.pdf")]

        result = generate_answer("Question?", results)

        assert result.cited_sources == []
        assert "don't know" in result.answer

    def test_raises_when_message_content_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "qwen3:8b", "done": True})

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        with pytest.raises(ValueError, match="missing 'message.content'"):
            generate_answer("Question?", [_make_result("x", "h", "d.pdf")])
