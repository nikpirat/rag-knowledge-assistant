"""Tests for rag_knowledge_assistant.evaluation.metrics.

Uses httpx.MockTransport, same rationale as the embeddings/generation
tests: verifies our own request construction and response parsing against
Ollama's real API contracts, without needing a live Ollama daemon.
"""

import json
from itertools import count

import httpx
import pytest

from rag_knowledge_assistant.evaluation.metrics import (
    _cosine_similarity,
    answer_relevancy_score,
    faithfulness_score,
)


def _install_chat_responses(
    monkeypatch: pytest.MonkeyPatch, responses: list[str]
) -> list[dict[str, object]]:
    captured_requests: list[dict[str, object]] = []
    call_index = count()

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content))
        idx = next(call_index)
        content = responses[idx]
        return httpx.Response(
            200, json={"model": "qwen3:8b", "message": {"role": "assistant", "content": content}}
        )

    mock_transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
    )
    return captured_requests


class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_not_nan(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestFaithfulnessScore:
    def test_all_claims_supported_scores_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_chat_responses(
            monkeypatch,
            responses=[
                "The sky is blue.\nWater is wet.",
                "YES",
                "YES",
            ],
        )

        score = faithfulness_score("The sky is blue and water is wet.", ["Some context."])

        assert score == pytest.approx(1.0)

    def test_half_claims_supported_scores_half(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_chat_responses(
            monkeypatch,
            responses=[
                "Claim one.\nClaim two.",
                "YES",
                "NO",
            ],
        )

        score = faithfulness_score("Some answer.", ["Some context."])

        assert score == pytest.approx(0.5)

    def test_no_extractable_claims_but_legitimate_refusal_scores_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_chat_responses(monkeypatch, responses=["   "])  # blank -> no claims

        score = faithfulness_score("I don't know based on the provided sources.", ["Some context."])

        assert score == 1.0

    def test_no_extractable_claims_and_not_a_refusal_scores_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real failure mode found in Phase 5 evaluation: the model
        returned a list of unanswered questions instead of an answer, for
        3 of 12 real eval questions. This has no extractable factual
        claims (nothing to decompose), but it is NOT a legitimate refusal
        - it should score 0.0, not get a free pass."""
        _install_chat_responses(monkeypatch, responses=["   "])  # blank -> no claims

        degenerate_answer = (
            "What are the best practices for managing service quotas "
            "across multiple accounts and regions?\n"
            "How can service quotas impact the reliability of a workload?"
        )
        score = faithfulness_score(degenerate_answer, ["Some context."])

        assert score == 0.0

    def test_real_recurrence_from_production_still_caught(self) -> None:
        """This exact answer text was returned by qwen3:8b in a real
        evaluation re-run and STILL scored 1.0 under the first fix
        (checking for a refusal phrase when claims come back empty) -
        because the claim-splitter didn't return an empty list for it; it
        extracted each question line as if it were a claim, and the judge
        rated them "supported" since they're topically related. No mocked
        HTTP needed here: _looks_like_question_list runs before any judge
        call at all, so this must short-circuit without a network call.
        """
        real_degenerate_answer = (
            "How can I monitor operational health of a workload?  \n"
            "What tools are recommended for monitoring workload components?  \n"
            "What are the best practices for creating effective dashboards "
            "for operational visibility?"
        )

        score = faithfulness_score(real_degenerate_answer, ["Some context."])

        assert score == 0.0


class TestAnswerRelevancyScore:
    def test_computes_average_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed_call_index = count()
        vectors = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/chat"):
                return httpx.Response(
                    200,
                    json={
                        "model": "qwen3:8b",
                        "message": {"role": "assistant", "content": "Question A?\nQuestion B?"},
                    },
                )
            if request.url.path.endswith("/api/embed"):
                idx = next(embed_call_index)
                return httpx.Response(
                    200, json={"model": "qwen3-embedding:0.6b", "embeddings": [vectors[idx]]}
                )
            raise AssertionError(f"Unexpected request path: {request.url.path}")

        mock_transport = httpx.MockTransport(handler)
        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: original_client(transport=mock_transport, **kwargs)
        )

        score = answer_relevancy_score("Original question?", "Some answer.", n_questions=2)

        assert score == pytest.approx(0.5)

    def test_no_hypothetical_questions_scores_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_chat_responses(monkeypatch, responses=["   "])

        score = answer_relevancy_score("Question?", "Answer with no clear topic.")

        assert score == 0.0
