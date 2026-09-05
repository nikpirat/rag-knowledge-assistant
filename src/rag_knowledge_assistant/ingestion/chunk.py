"""Structure-aware chunking: splits on detected section headings first,
falling back to fixed-size splitting with overlap only for oversized
sections.

Headings are detected by font size relative to the document's own body
text size (the most common line font size) - not by matching a specific
document's heading ID pattern. This generalizes to other similarly
formatted documents, not just this one corpus.
"""

import logging
from collections import Counter
from dataclasses import dataclass

from rag_knowledge_assistant.ingestion.extract import Line

logger = logging.getLogger(__name__)

HEADING_SIZE_RATIO = 1.15  # heading font must be >=15% larger than body text
MAX_CHUNK_CHARS = 2000  # ~500 tokens at ~4 chars/token, a common RAG default
CHUNK_OVERLAP_CHARS = 200  # ~50 tokens


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text ready for embedding, with source metadata."""

    text: str
    section_heading: str
    source_document: str
    page_number: int
    chunk_index: int


def _detect_body_font_size(lines: list[Line]) -> float:
    """The most common line font size in the document - our proxy for
    'this is body text, not a heading'.
    """
    if not lines:
        raise ValueError("Cannot detect body font size: no lines extracted")
    rounded_sizes = [round(line.font_size, 1) for line in lines]
    most_common_size, _ = Counter(rounded_sizes).most_common(1)[0]
    return most_common_size


def _is_heading(line: Line, body_size: float) -> bool:
    return line.font_size >= body_size * HEADING_SIZE_RATIO


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Fixed-size fallback split, only used for sections exceeding max_chars."""
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end])
        start = end - overlap_chars
    return pieces


def chunk_document(lines: list[Line], source_document: str) -> list[Chunk]:
    """Group lines into sections at detected heading boundaries, then split
    any oversized section further using the fixed-size+overlap fallback.
    """
    if not lines:
        return []

    body_size = _detect_body_font_size(lines)

    sections: list[tuple[str, list[Line]]] = []
    current_heading = "Introduction"
    current_lines: list[Line] = []

    for line in lines:
        if _is_heading(line, body_size):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.text
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    chunk_index = 0
    for heading, section_lines in sections:
        section_text = "\n".join(line.text for line in section_lines)
        page_number = section_lines[0].page_number if section_lines else 0

        for piece in _split_with_overlap(section_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS):
            chunks.append(
                Chunk(
                    text=piece,
                    section_heading=heading,
                    source_document=source_document,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    logger.info(
        "Chunked %s into %d chunks across %d detected sections",
        source_document,
        len(chunks),
        len(sections),
    )
    return chunks
