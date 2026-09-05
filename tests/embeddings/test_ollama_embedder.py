"""Tests for rag_knowledge_assistant.embeddings.ollama_embedder.

Uses httpx.MockTransport rather than a real Ollama daemon - this isolates
what we actually control and need to verify: that our client constructs
the request correctly (model, input, the query instruction prefix) and
parses the response correctly, against Ollama's documented /api/embed
contract. It cannot verify Ollama's own behavior, only ours.
"""

import json

import httpx
import pytest

from rag_knowledge_assistant.embeddings.ollama_embedder import (
    QUERY_INSTRUCTION,
    embed_documents,
    embed_query,
)


class TestEmbedDocuments:
    def test_sends_no_instruction_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_request: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"model": "qwen3-embedding:0.6b", "embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            )

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        result = embed_documents(["First document.", "Second document."])

        assert captured_request["body"]["input"] == ["First document.", "Second document."]
        assert "Instruct:" not in json.dumps(captured_request["body"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_empty_input_returns_empty_without_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Should not make an HTTP request for empty input")

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        assert embed_documents([]) == []

    def test_raises_when_response_missing_embeddings_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "qwen3-embedding:0.6b"})

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        with pytest.raises(ValueError, match="missing 'embeddings' field"):
            embed_documents(["some text"])


class TestEmbedQuery:
    def test_sends_instruction_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_request: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"model": "qwen3-embedding:0.6b", "embeddings": [[0.5, 0.6]]}
            )

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        result = embed_query("How do I optimize costs?")

        sent_input = captured_request["body"]["input"][0]
        assert QUERY_INSTRUCTION in sent_input
        assert "How do I optimize costs?" in sent_input
        assert result == [0.5, 0.6]
