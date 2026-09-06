"""Tests for rag_knowledge_assistant.evaluation.run_eval.

Fakes embed_query, search, generate_answer, and both metric functions —
this test verifies our own orchestration (looping over questions,
assembling result rows, writing CSV correctly, isolating per-question
failures), not the real Ollama pipeline or metric computation, which are
tested separately.
"""

import csv
from pathlib import Path

import pytest

from rag_knowledge_assistant.evaluation import run_eval
from rag_knowledge_assistant.generation.ollama_generator import GenerationResult
from rag_knowledge_assistant.retrieval.store import SearchResult


def _fake_search_result() -> SearchResult:
    return SearchResult(
        text="Some retrieved content.",
        section_heading="Heading",
        source_document="doc.pdf",
        page_number=1,
        score=0.9,
    )


class TestRunEvaluation:
    def test_writes_one_row_per_question_with_correct_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_questions = ["Question one?", "Question two?"]
        monkeypatch.setattr(run_eval, "EVAL_QUESTIONS", fake_questions)
        monkeypatch.setattr(run_eval, "QdrantClient", lambda path: object())
        monkeypatch.setattr(run_eval, "embed_query", lambda *a, **k: [0.1, 0.2])
        monkeypatch.setattr(run_eval, "search", lambda *a, **k: [_fake_search_result()])
        monkeypatch.setattr(
            run_eval,
            "generate_answer",
            lambda question, results, **k: GenerationResult(
                answer=f"Answer to: {question}", cited_sources=results
            ),
        )
        monkeypatch.setattr(run_eval, "faithfulness_score", lambda *a, **k: 0.8)
        monkeypatch.setattr(run_eval, "answer_relevancy_score", lambda *a, **k: 0.7)

        output_path = tmp_path / "results.csv"
        results = run_eval.run_evaluation(output_path)

        assert len(results) == 2
        assert results[0]["question"] == "Question one?"
        assert results[0]["answer"] == "Answer to: Question one?"
        assert results[0]["faithfulness"] == 0.8
        assert results[0]["answer_relevancy"] == 0.7
        assert results[0]["num_sources_cited"] == 1

        assert output_path.exists()
        with output_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["question"] == "Question one?"

    def test_empty_question_set_writes_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_eval, "EVAL_QUESTIONS", [])
        monkeypatch.setattr(run_eval, "QdrantClient", lambda path: object())

        output_path = tmp_path / "results.csv"
        results = run_eval.run_evaluation(output_path)

        assert results == []
        assert not output_path.exists()

    def test_one_failing_question_does_not_stop_the_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real motivation for this module: local Ollama has
        demonstrated genuine intermittent unreliability during this
        project's development. One question's pipeline call failing (or
        timing out) must not prevent the other 11 questions in a real run
        from being evaluated and saved."""
        fake_questions = ["Good question one?", "Bad question?", "Good question two?"]
        monkeypatch.setattr(run_eval, "EVAL_QUESTIONS", fake_questions)
        monkeypatch.setattr(run_eval, "QdrantClient", lambda path: object())
        monkeypatch.setattr(run_eval, "embed_query", lambda *a, **k: [0.1, 0.2])
        monkeypatch.setattr(run_eval, "search", lambda *a, **k: [_fake_search_result()])

        def fake_generate_answer(question: str, results: list, **kwargs: object) -> object:
            if question == "Bad question?":
                raise TimeoutError("simulated Ollama hang")
            return GenerationResult(answer=f"Answer to: {question}", cited_sources=results)

        monkeypatch.setattr(run_eval, "generate_answer", fake_generate_answer)
        monkeypatch.setattr(run_eval, "faithfulness_score", lambda *a, **k: 0.9)
        monkeypatch.setattr(run_eval, "answer_relevancy_score", lambda *a, **k: 0.8)

        output_path = tmp_path / "results.csv"
        results = run_eval.run_evaluation(output_path)

        assert len(results) == 3

        good_1, bad, good_2 = results
        assert good_1["error"] == ""
        assert good_1["faithfulness"] == 0.9
        assert good_1["answer"] == "Answer to: Good question one?"

        assert bad["error"] == "simulated Ollama hang"
        assert bad["answer"] == ""
        assert bad["faithfulness"] == ""

        assert good_2["error"] == ""
        assert good_2["answer"] == "Answer to: Good question two?"

        with output_path.open() as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 3
        assert csv_rows[1]["error"] == "simulated Ollama hang"
