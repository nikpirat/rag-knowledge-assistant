"""Tests for rag_knowledge_assistant.ingestion.chunk.

Uses hand-constructed Line objects directly rather than round-tripping
through PDF generation for every test - faster, and isolates chunking
logic from extraction logic (extraction is tested separately in
test_extract.py against a real generated PDF).
"""

import pytest

from rag_knowledge_assistant.ingestion.chunk import (
    _detect_body_font_size,
    _is_heading,
    _split_with_overlap,
    chunk_document,
)
from rag_knowledge_assistant.ingestion.extract import Line


def _line(text: str, size: float, page: int = 1) -> Line:
    return Line(text=text, font_size=size, page_number=page)


class TestDetectBodyFontSize:
    def test_returns_most_common_size(self) -> None:
        lines = [_line("a", 11.0), _line("b", 11.0), _line("c", 16.0), _line("d", 11.0)]
        assert _detect_body_font_size(lines) == 11.0

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no lines extracted"):
            _detect_body_font_size([])


class TestIsHeading:
    def test_larger_font_is_heading(self) -> None:
        assert _is_heading(_line("Heading", 16.0), body_size=11.0) is True

    def test_body_sized_font_is_not_heading(self) -> None:
        assert _is_heading(_line("Body text", 11.0), body_size=11.0) is False

    def test_slightly_larger_but_under_threshold_is_not_heading(self) -> None:
        assert _is_heading(_line("Slightly bigger", 12.0), body_size=11.0) is False


class TestSplitWithOverlap:
    def test_short_text_returned_as_single_piece(self) -> None:
        pieces = _split_with_overlap("short text", max_chars=100, overlap_chars=10)
        assert pieces == ["short text"]

    def test_long_text_split_with_overlap(self) -> None:
        text = "x" * 250
        pieces = _split_with_overlap(text, max_chars=100, overlap_chars=20)

        assert len(pieces) > 1
        assert pieces[0][-20:] == pieces[1][:20]


class TestChunkDocument:
    def test_groups_lines_by_detected_heading(self) -> None:
        lines = [
            _line("REL01-BP01 First practice", 16.0),
            _line("Body sentence one.", 11.0),
            _line("Body sentence two.", 11.0),
            _line("REL01-BP02 Second practice", 16.0),
            _line("Different body sentence.", 11.0),
        ]

        chunks = chunk_document(lines, source_document="test.pdf")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "REL01-BP01 First practice"
        assert "Body sentence one." in chunks[0].text
        assert "Body sentence two." in chunks[0].text
        assert chunks[1].section_heading == "REL01-BP02 Second practice"
        assert "Different body sentence." in chunks[1].text

    def test_text_before_first_heading_gets_introduction_label(self) -> None:
        lines = [
            _line("Some preamble text.", 11.0),
            _line("REL01-BP01 First practice", 16.0),
            _line("Body sentence.", 11.0),
        ]

        chunks = chunk_document(lines, source_document="test.pdf")

        assert chunks[0].section_heading == "Introduction"
        assert "Some preamble text." in chunks[0].text

    def test_oversized_section_gets_split(self) -> None:
        long_body_lines = [_line(f"Sentence number {i}.", 11.0) for i in range(200)]
        lines = [_line("REL01-BP01 A practice", 16.0), *long_body_lines]

        chunks = chunk_document(lines, source_document="test.pdf")

        assert len(chunks) > 1
        assert all(c.section_heading == "REL01-BP01 A practice" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_input_returns_no_chunks(self) -> None:
        assert chunk_document([], source_document="test.pdf") == []
