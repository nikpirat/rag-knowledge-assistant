"""Answer generation via a locally-running Ollama instance (Qwen3 8B).

Uses Ollama's /api/chat endpoint (confirmed against the official
ollama/ollama repository docs), not a raw-prompt /api/generate call -
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
    """A generated answer with the sources it actually cited - not just
    every source that was retrieved, only the ones the model referenced
    with a [N] marker. A retrieved chunk that didn't make it into the
    answer isn't a source of that answer.
    """

    answer: str
    cited_sources: list[SearchResult]


def _extract_cited_source_numbers(answer: str) -> set[int]:
    """Parse [N] citation markers out of the answer text."""
    return {int(match) for match in CITATION_PATTERN.findall(answer)}


def generate_answer(
    question: str,
    results: list[SearchResult],
    model: str = settings.generation_model,
    base_url: str = settings.ollama_base_url,
    timeout: float = 300.0,
    repeat_penalty: float | None = None,
) -> GenerationResult:
    """Generate a grounded answer to `question` using `results` as context.

    Returns the answer text plus only the sources the model actually
    cited — not the full retrieved set.

    repeat_penalty defaults to None (no override), which means Ollama's
    own built-in default (1.1) applies. An earlier version of this
    function hardcoded 1.3, based on an unverified assumption that a
    "moderately assertive" value would fix observed answer-padding — in
    real testing this caused a worse regression (incoherent, garbled
    output, off-topic tangents), confirmed against multiple sources
    stating repeat_penalty should rarely exceed ~1.2 and that it can
    aggressively suppress legitimate repeated domain terms. If tuning
    this is ever needed, treat 1.2 as a hard ceiling, not a starting point.
    """
    messages = build_messages(question, results)

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

    answer: str = message["content"]
    cited_numbers = _extract_cited_source_numbers(answer)

    cited_sources = [results[n - 1] for n in sorted(cited_numbers) if 1 <= n <= len(results)]

    return GenerationResult(answer=answer, cited_sources=cited_sources)
