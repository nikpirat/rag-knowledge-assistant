"""PDF text extraction with structural metadata (font size) preserved.

Uses pdfplumber, not PyMuPDF - PyMuPDF's community edition is AGPL-3.0,
a real licensing concern for commercial use that many companies avoid
without a paid commercial license. pdfplumber is MIT-licensed.

Extracts per-line font size alongside text. The chunking module uses this
to detect section headings by font size relative to the document's own
body text — not by matching a document-specific heading ID pattern (e.g.
this corpus's "REL01-BP01" scheme), so the same logic generalizes to any
similarly-formatted document set, not just AWS whitepapers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Line:
    """A single reconstructed line of text with its dominant font size."""

    text: str
    font_size: float
    page_number: int


def extract_lines(pdf_path: Path) -> list[Line]:
    """Extract text from a PDF as a list of lines, each tagged with the
    page number and average font size of its words.
    """
    lines: list[Line] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue

            lines_by_position: dict[int, list[dict[str, Any]]] = {}
            for word in words:
                line_key = round(word["top"])
                lines_by_position.setdefault(line_key, []).append(word)

            for line_key in sorted(lines_by_position):
                line_words = sorted(lines_by_position[line_key], key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in line_words)
                avg_size = sum(w["size"] for w in line_words) / len(line_words)
                lines.append(Line(text=text, font_size=avg_size, page_number=page_number))

    logger.info("Extracted %d lines from %s", len(lines), pdf_path)
    return lines
