"""Answer generation via a locally-running Ollama instance (Qwen3 8B).

Uses Ollama's /api/chat endpoint (confirmed against the official
ollama/ollama repository docs), not a raw-prompt /api/generate call —
chat-tuned models like Qwen3 8B follow system/user message structure more
reliably than a single concatenated prompt string.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from rag_knowledge_assistant.config.settings import settings
from rag_knowledge_assistant.generation.prompt import build_messages
from rag_knowledge_assistant.retrieval.store import SearchResult

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+)]")


@dataclass(frozen=True)
class GenerationResult:
    """A generated answer with the sources it actually cited — not just
    every source that was retrieved, only the ones the model referenced
    with a [N] marker. A retrieved chunk that didn't make it into the
    answer isn't a source of that answer.
    """

    answer: str
    cited_sources: list[SearchResult]
    retried: bool = False


def looks_like_degenerate_answer(answer: str) -> bool:
    """Detects two real, observed failure modes, not one:

    1. An empty or near-empty answer - Qwen3's thinking-mode empty-content
       bug (see Phase 3's think=False fix) can apparently still surface
       on rare edge-case prompts even with think explicitly disabled;
       found via a real evaluation run where one answer came back as a
       literal empty string.
    2. A list of related QUESTIONS instead of an answer (see Phase 5
       evaluations original finding).

    Public (not module-private) because it's used both here, as a
    defense-in-depth check before returning an answer to a caller, and in
    evaluation/metrics.py, to correctly score these failure modes rather
    than mistakenly treating them as a legitimate "no claims to check"
    refusal. One canonical implementation, not a duplicated copy.
    """
    stripped = answer.strip()
    # The real observed case was a literal empty string, not merely a
    # short one - checking for genuinely empty/whitespace-only content
    # specifically (rather than an arbitrary length threshold) avoids
    # false-positiving on legitimately short but real answers.
    if not stripped:
        return True

    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if not lines:
        return True
    question_lines = sum(1 for line in lines if line.endswith("?"))
    return question_lines / len(lines) > 0.5


def _extract_cited_source_numbers(answer: str) -> set[int]:
    """Parse [N] citation markers out of the answer text."""
    return {int(match) for match in CITATION_PATTERN.findall(answer)}


def _call_chat(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    timeout: float,
    repeat_penalty: float | None,
) -> str:
    """Single call to /api/chat, returning the raw answer text."""
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if repeat_penalty is not None:
        request_body["options"] = {"repeat_penalty": repeat_penalty}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}/api/chat", json=request_body)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    message = data.get("message")
    if message is None or "content" not in message:
        raise ValueError(f"Ollama response missing 'message.content' field: {data}")
    return str(message["content"])


def generate_answer(
    question: str,
    results: list[SearchResult],
    model: str = settings.generation_model,
    base_url: str = settings.ollama_base_url,
    timeout: float = 300.0,
    repeat_penalty: float | None = None,
    max_retries: int = 1,
) -> GenerationResult:
    """Generate a grounded answer to `question` using `results` as context.

    Returns the answer text plus only the sources the model actually
    cited - not the full retrieved set.

    max_retries (default 1) guards against a real, observed failure mode:
    the model returning a list of related questions instead of an answer
    (see looks_like_degenerate_answer). Rather than resending the identical
    prompt - likely to fail the same way again - a detected failure is
    shown back to the model as its own prior turn, with an explicit
    correction request, a standard LLM self-correction pattern. This lives
    here (not only in evaluation) so both the real /ask endpoint and the
    evaluation harness see - and are measured against - the same actual
    production behavior, retries included.

    Real data motivating this: with qwen3:8b as generator, roughly 3 of
    12 real evaluation questions (25%) triggered this failure with no
    retry. A larger model (qwen3:14b) showed zero such failures across
    the same 12 questions - worth using a larger generation model too,
    but no model size fully eliminates the risk, hence this safety net
    regardless of which model is configured.

    timeout defaults to 300s (up from an initial 120s): confirmed via
    `ollama ps` that the model runs fully on GPU, so this isn't masking a
    CPU-fallback performance problem - a multi-source context (thousands
    of tokens) plus a multi-paragraph generated answer, combined with
    Ollama's default 5-minute model unload/reload cycle between requests,
    can genuinely exceed 120s even with full GPU utilization.

    repeat_penalty defaults to None (no override), which means Ollama's
    own built-in default (1.1) applies. An earlier version of this
    function hardcoded 1.3, based on an unverified assumption that a
    "moderately assertive" value would fix observed answer-padding - in
    real testing this caused a worse regression (incoherent, garbled
    output, off-topic tangents), confirmed against multiple sources
    stating repeat_penalty should rarely exceed ~1.2 and that it can
    aggressively suppress legitimate repeated domain terms. If tuning
    this is ever needed, treat 1.2 as a hard ceiling, not a starting point.
    """
    messages = build_messages(question, results)
    answer = ""
    retried = False

    for attempt in range(max_retries + 1):
        answer = _call_chat(messages, model, base_url, timeout, repeat_penalty)

        if not looks_like_degenerate_answer(answer):
            break

        if attempt < max_retries:
            retried = True
            logger.warning(
                "Degenerate answer detected (attempt %d), retrying: %r",
                attempt + 1,
                answer,
            )
            messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "That response did not answer the question — it was either "
                        "empty or a list of questions rather than a direct answer. "
                        "Answer the original question directly using the sources "
                        "provided, with inline [N] citations."
                    ),
                },
            ]

    cited_numbers = _extract_cited_source_numbers(answer)

    cited_sources = [results[n - 1] for n in sorted(cited_numbers) if 1 <= n <= len(results)]

    return GenerationResult(answer=answer, cited_sources=cited_sources, retried=retried)
