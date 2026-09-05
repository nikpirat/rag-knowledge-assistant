"""Prompt construction for grounded, cited generation.

Builds a strict-grounding system prompt (answer only from provided
context, refuse if the answer isn't there) and a user prompt containing
numbered source blocks the model is instructed to cite inline as [N].

Strict grounding is deliberate, not a default we happened to pick: this
system answers questions about internal technical documentation, where a
confidently-wrong answer sourced from the model's general training data
is worse than no answer at all. A more permissive "augment with general
knowledge" mode would be a deliberate future extension, not this default.
"""

from rag_knowledge_assistant.retrieval.store import SearchResult

SYSTEM_PROMPT = """You are a technical assistant answering questions strictly \
based on the numbered source excerpts provided by the user. Follow these \
rules exactly:

1. Answer ONLY using information contained in the numbered sources. Do \
not use any outside knowledge, even if you know the answer from general \
training.
2. If the sources do not contain enough information to answer the \
question, say so explicitly rather than guessing or filling gaps with \
assumptions.
3. Every factual claim in your answer must be followed by a citation \
marker like [1] or [2] referencing the source number it came from.
4. Do not fabricate source numbers — only cite sources that were \
actually provided to you.
5. Cover only as many distinct points as the sources actually support. \
Do not repeat a point you have already made, and do not pad your answer \
with restated information just to appear more thorough."""


def build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as a numbered source list for the prompt."""
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[Source {i}] ({result.source_document}, section: {result.section_heading})\n"
            f"{result.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """Build the full chat message list for the generation call."""
    context_block = build_context_block(results)
    user_content = (
        f"Sources:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the sources above. Cite each fact with "
        "the matching source number in brackets - for example, write [1] to cite "
        "Source 1, or [2] to cite Source 2. Always replace the number with the "
        "actual source you are using; never write the literal characters '[N]'."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
