"""Hand-rolled RAG evaluation metrics: Faithfulness and Answer Relevancy.

Implements the same methodology as RAGAS's reference-free metrics,
without depending on the `ragas` package - which, as of this writing, has
a hard, unconditional import of `langchain_community.chat_models.vertexai`
that's broken by langchain-community's own removal of that submodule
during its official sunsetting. No combination of ragas/langchain-community/
langchain-ollama versions resolves cleanly (confirmed against both the
latest ragas release and an older 0.3.x release - both fail identically).
Reimplementing these two metrics directly against our own already-working
Ollama clients sidesteps that broken dependency chain entirely.

Faithfulness: decompose the answer into discrete factual claims, then ask
the judge model whether each claim is supported by the retrieved context.
Score = fraction of claims supported. An answer with no extractable
claims is scored based on WHY: a genuine refusal scores 1.0 (nothing to
hallucinate), a degenerate non-answer scores 0.0 (see _looks_like_refusal
and generation.ollama_generator.looks_like_degenerate_answer docstrings —
this distinction exists because real Phase 5 evaluation found the model
sometimes returns a list of unanswered questions instead of an answer for
weakly-retrieved queries, which the original "no claims → 1.0" logic
incorrectly scored as perfectly faithful).

Answer Relevancy: ask the judge model to generate hypothetical questions
the answer would appropriately address, embed them alongside the original
question (via our own embed_query), and average their cosine similarity.
A directly relevant answer should imply questions similar to the one
actually asked; an evasive or off-topic answer implies dissimilar ones.

Known limitation: embed_query applies a retrieval-oriented instruction
prefix (Phase 2) to every text it embeds, including the hypothetical
questions here, which aren't a retrieval-query use case. Since the same
transformation is applied uniformly to both sides of the comparison,
relative similarity remains meaningful, but this wasn't purpose-built for
question-to-question similarity - a reasonable reuse of existing,
already-tested code, not a perfectly tailored metric.
"""

import logging
import math
from typing import Any

import httpx

from rag_knowledge_assistant.config.settings import settings
from rag_knowledge_assistant.embeddings.ollama_embedder import embed_query
from rag_knowledge_assistant.generation.ollama_generator import looks_like_degenerate_answer

logger = logging.getLogger(__name__)

_REFUSAL_INDICATORS = (
    "don't know",
    "do not know",
    "does not contain",
    "doesn't contain",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "can't answer",
    "no information",
    "not provided in the",
    "not contained in the",
)


def _call_judge(
    prompt: str,
    model: str = settings.judge_model,
    base_url: str = settings.ollama_base_url,
    timeout: float = 300.0,
) -> str:
    """Low-level single-turn chat call to the judge model.

    Deliberately separate from generation/ollama_generator.py's
    generate_answer(): that function is specific to RAG-answer generation
    (source-numbered context, citation extraction) - this is a simpler,
    single-purpose helper for judge prompts that just need raw text back.
    Still needs think=False for the same reason established in Phase 3:
    Qwen3's thinking mode can leave message.content empty otherwise.
    """
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    message = data.get("message")
    if message is None or "content" not in message:
        raise ValueError(f"Ollama response missing 'message.content' field: {data}")
    return str(message["content"])


def _split_into_claims(answer: str, model: str, base_url: str) -> list[str]:
    """Decompose an answer into discrete, individually-checkable factual claims."""
    prompt = (
        "Break the following answer into a list of individual, standalone "
        "factual claims. Output ONLY the claims, one per line, with no "
        "numbering, bullets, or extra commentary.\n\n"
        f"Answer:\n{answer}"
    )
    response = _call_judge(prompt, model=model, base_url=base_url)
    return [line.strip("-* \t") for line in response.strip().split("\n") if line.strip()]


def _looks_like_refusal(answer: str) -> bool:
    """Deterministic keyword heuristic, not another judge call - keeps
    per-question call count down given Ollama's demonstrated fragility
    under load (Phase 4). Documented limitation: a heuristic can miss
    unusual refusal phrasings or, more rarely, false-positive on a
    non-refusal answer that happens to contain one of these phrases.
    """
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_INDICATORS)


def _is_claim_supported(claim: str, context: str, model: str, base_url: str) -> bool:
    """Ask the judge whether a single claim is supported by the given context."""
    prompt = (
        f"Context:\n{context}\n\n"
        f"Claim: {claim}\n\n"
        "Does the context support this claim? Answer with exactly one "
        "word: YES or NO."
    )
    response = _call_judge(prompt, model=model, base_url=base_url)
    return response.strip().upper().startswith("YES")


def faithfulness_score(
    answer: str,
    contexts: list[str],
    model: str = settings.judge_model,
    base_url: str = settings.ollama_base_url,
) -> float:
    """Fraction of the answer's factual claims that are supported by the
    retrieved context. 1.0 = fully grounded, 0.0 = nothing supported.

    An answer with no extractable claims is scored based on WHY it has
    none: a legitimate refusal ("the sources don't contain enough
    information") scores 1.0 - there's nothing to hallucinate, and
    refusing appropriately is the correct behavior our system prompt asks
    for. A degenerate non-answer with no extractable claims for any OTHER
    reason scores 0.0 - see the real failure mode found in Phase 5
    evaluation (module docstring), where the model returned a list of
    unanswered questions instead of an answer or a proper refusal.

    A question-list is checked FIRST and rejected directly (see
    looks_like_degenerate_answer docstring in generation/ollama_generator.py):
    re-running the same evaluation showed the claim-splitter doesn't
    reliably return zero claims for one, so waiting for an empty claims
    list to trigger the refusal check was insufficient in practice, not
    just in theory.
    """
    if looks_like_degenerate_answer(answer):
        logger.warning(
            "Answer looks like a list of questions, not an answer — scoring 0.0: %r", answer
        )
        return 0.0

    combined_context = "\n\n".join(contexts)
    claims = _split_into_claims(answer, model, base_url)

    if not claims:
        if _looks_like_refusal(answer):
            return 1.0
        logger.warning(
            "No extractable claims and answer doesn't look like a refusal "
            "- likely a degenerate non-answer, not a safe refusal: %r",
            answer,
        )
        return 0.0

    supported = sum(
        1 for claim in claims if _is_claim_supported(claim, combined_context, model, base_url)
    )
    return supported / len(claims)


def _generate_hypothetical_questions(
    answer: str, model: str, base_url: str, n: int = 3
) -> list[str]:
    """Ask the judge what questions this answer would appropriately address."""
    prompt = (
        f"Generate {n} different questions that the following answer would "
        "be an appropriate, direct response to. Output ONLY the questions, "
        "one per line, no numbering.\n\n"
        f"Answer:\n{answer}"
    )
    response = _call_judge(prompt, model=model, base_url=base_url)
    questions = [line.strip("-* \t") for line in response.strip().split("\n") if line.strip()]
    return questions[:n]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def answer_relevancy_score(
    question: str,
    answer: str,
    model: str = settings.judge_model,
    base_url: str = settings.ollama_base_url,
    embedding_model: str = settings.embedding_model,
    n_questions: int = 3,
) -> float:
    """Average cosine similarity between the original question and several
    hypothetical questions the judge model believes the answer addresses.

    Returns 0.0 if the judge could not generate any hypothetical
    questions (e.g. a refusal-style answer with no clear topic).
    """
    hypothetical_questions = _generate_hypothetical_questions(answer, model, base_url, n_questions)
    if not hypothetical_questions:
        return 0.0

    question_embedding = embed_query(question, model=embedding_model, base_url=base_url)
    similarities = [
        _cosine_similarity(
            question_embedding, embed_query(hq, model=embedding_model, base_url=base_url)
        )
        for hq in hypothetical_questions
    ]
    return sum(similarities) / len(similarities)
