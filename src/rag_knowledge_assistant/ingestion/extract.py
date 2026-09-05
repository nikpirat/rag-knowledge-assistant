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


def filter_boilerplate_lines(
    lines: list[Line], threshold: float = 0.2, min_occurrences: int = 3
) -> list[Line]:
    """Remove lines that repeat identically across many pages - running
    headers/footers, not real content.

    Detected statistically (a line's text appearing on more than
    `threshold` fraction of the document's pages), not by matching a
    specific known header string - this generalizes to any document's
    own boilerplate, discovered empirically against the real AWS
    Well-Architected white papers, which repeat a running header like
    "{Pillar Name} AWS Well-Architected Framework" on nearly every page.

    Repetition is measured as "fraction of DISTINCT PAGES this text
    appears on", not raw occurrence count - a line appearing twice on one
    page shouldn't count as more suspicious than appearing once.

    Requires BOTH the fraction threshold AND a minimum absolute page
    count (min_occurrences) before flagging something as boilerplate.
    Fraction alone breaks down on short documents: on a single-page
    document, every line trivially appears on "100% of pages" - without
    an absolute floor, this would strip all content from short inputs,
    a real bug caught by testing against a genuinely short synthetic PDF.
    """
    if not lines:
        return lines

    total_pages = len({line.page_number for line in lines})
    if total_pages == 0:
        return lines

    pages_by_text: dict[str, set[int]] = {}
    for line in lines:
        pages_by_text.setdefault(line.text, set()).add(line.page_number)

    boilerplate_texts = {
        text
        for text, pages in pages_by_text.items()
        if len(pages) / total_pages > threshold and len(pages) >= min_occurrences
    }

    if boilerplate_texts:
        logger.info(
            "Filtered %d boilerplate line(s) repeating across >%.0f%% of pages: %s",
            len(boilerplate_texts),
            threshold * 100,
            boilerplate_texts,
        )

    return [line for line in lines if line.text not in boilerplate_texts]
