"""Run the evaluation harness: full RAG pipeline + Faithfulness/Answer
Relevancy metrics across the curated question set.

Expect this to take a while to run for real: each question triggers the
full RAG pipeline (embed + retrieve + generate) plus several additional
judge-model calls per metric (claim decomposition, per-claim
verification, hypothetical question generation) — roughly 8-12 Ollama
calls per question. Across 12 questions that's over a hundred calls; a
few minutes end to end is expected, not a sign of something wrong.

Each question is evaluated inside its own try/except. Local Ollama has
demonstrated real, intermittent unreliability during this project's
development (Phase 4's request-backlog incident, Phase 5's ROCm backend
bug) — sometimes a single question's pipeline call hangs or times out for
reasons outside this code's control. Without per-question fault
isolation, one bad question kills the entire run and you get zero
results instead of eleven good ones. This is a standard resilience
pattern for any system built on an external dependency that can't be made
perfectly reliable — isolate the failure, keep going, report it clearly
rather than crashing or silently ignoring it.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from rag_knowledge_assistant.config.settings import settings
from rag_knowledge_assistant.embeddings.ollama_embedder import embed_query
from rag_knowledge_assistant.evaluation.dataset import EVAL_QUESTIONS
from rag_knowledge_assistant.evaluation.metrics import answer_relevancy_score, faithfulness_score
from rag_knowledge_assistant.generation.ollama_generator import generate_answer
from rag_knowledge_assistant.retrieval.store import search

logger = logging.getLogger(__name__)


def _evaluate_one_question(
    question: str,
    client: QdrantClient,
) -> dict[str, Any]:
    """Run the full pipeline + both metrics for a single question.

    Raises whatever the underlying calls raise — fault isolation (catching
    this per-question, not per-run) is the caller's responsibility, so
    this function itself stays simple and testable in isolation.
    """
    query_vector = embed_query(
        question, model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    retrieved = search(
        client, settings.qdrant_collection_name, query_vector, limit=settings.retrieval_top_k
    )
    generation_result = generate_answer(
        question,
        retrieved,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
    )

    contexts = [r.text for r in retrieved]
    faithfulness = faithfulness_score(
        generation_result.answer,
        contexts,
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
    )
    relevancy = answer_relevancy_score(
        question,
        generation_result.answer,
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
        embedding_model=settings.embedding_model,
    )

    return {
        "question": question,
        "answer": generation_result.answer,
        "num_sources_cited": len(generation_result.cited_sources),
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "error": "",
    }


def run_evaluation(output_path: Path) -> list[dict[str, Any]]:
    """Run the full RAG pipeline + evaluation metrics for every question
    in the curated eval set. Writes a CSV report and returns the raw
    per-question results — including any that failed, marked via the
    'error' column rather than silently dropped, so a failure is visible
    in the report rather than just missing.
    """
    client = QdrantClient(url=settings.qdrant_url)
    results_rows: list[dict[str, Any]] = []

    for question in EVAL_QUESTIONS:
        logger.info("Evaluating: %s", question)

        try:
            row = _evaluate_one_question(question, client)
            results_rows.append(row)
            logger.info(
                "  faithfulness=%.2f, answer_relevancy=%.2f",
                row["faithfulness"],
                row["answer_relevancy"],
            )
        except Exception as exc:  # noqa: BLE001 — intentionally broad: isolate ANY
            # failure for this one question (timeout, connection error, our own
            # ValueError, etc.) so it can't take down the other 11 questions.
            logger.error("Failed to evaluate question %r: %s", question, exc)
            results_rows.append(
                {
                    "question": question,
                    "answer": "",
                    "num_sources_cited": "",
                    "faithfulness": "",
                    "answer_relevancy": "",
                    "error": str(exc),
                }
            )

    if results_rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)

    return results_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = run_evaluation(
        Path(f"reports/evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv")
    )

    if not results:
        print("No results produced.")
        return

    successful = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]

    print(f"\nEvaluated {len(successful)}/{len(results)} questions successfully")
    if failed:
        print(f"{len(failed)} question(s) FAILED (see 'error' column in the CSV):")
        for r in failed:
            print(f"  - {r['question']}: {r['error']}")

    if successful:
        mean_faithfulness = sum(r["faithfulness"] for r in successful) / len(successful)
        mean_relevancy = sum(r["answer_relevancy"] for r in successful) / len(successful)
        print(f"Mean faithfulness (successful only): {mean_faithfulness:.3f}")
        print(f"Mean answer relevancy (successful only): {mean_relevancy:.3f}")
    else:
        print("No successful evaluations — cannot compute mean scores.")

    print("Full results: reports/evaluation_results.csv")


if __name__ == "__main__":
    main()
