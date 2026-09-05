"""Tests for rag_knowledge_assistant.ingestion.extract.

Uses a real PDF generated via reportlab (not a mock) - this is worth
doing for extraction specifically, since the whole point is verifying we
correctly read actual PDF bytes and font metadata, not just our own
in-memory data structures.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag_knowledge_assistant.ingestion.extract import Line, extract_lines, filter_boilerplate_lines


def _build_test_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "First Heading")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, "First body sentence.")
    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Second Page Heading")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, "Second page body sentence.")
    c.showPage()
    c.save()


class TestExtractLines:
    def test_extracts_correct_font_sizes(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _build_test_pdf(pdf_path)

        lines = extract_lines(pdf_path)

        heading_lines = [line for line in lines if line.font_size > 15.0]
        body_lines = [line for line in lines if line.font_size < 12.0]

        assert len(heading_lines) == 2
        assert len(body_lines) == 2
        assert "First Heading" in heading_lines[0].text

    def test_tracks_page_numbers_correctly(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _build_test_pdf(pdf_path)

        lines = extract_lines(pdf_path)

        page_1_lines = [line for line in lines if line.page_number == 1]
        page_2_lines = [line for line in lines if line.page_number == 2]

        assert any("First Heading" in line.text for line in page_1_lines)
        assert any("Second Page Heading" in line.text for line in page_2_lines)

    def test_empty_pdf_returns_no_lines(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "empty.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.showPage()
        c.save()

        lines = extract_lines(pdf_path)
        assert lines == []


class TestFilterBoilerplateLines:
    def test_removes_line_repeating_across_most_pages(self) -> None:
        lines = []
        for page in range(1, 11):
            lines.append(Line(text="Running Header Text", font_size=10.0, page_number=page))
            lines.append(
                Line(text=f"Unique body content page {page}", font_size=11.0, page_number=page)
            )

        filtered = filter_boilerplate_lines(lines, threshold=0.2)

        remaining_texts = {line.text for line in filtered}
        assert "Running Header Text" not in remaining_texts
        assert all(f"Unique body content page {p}" in remaining_texts for p in range(1, 11))

    def test_keeps_line_below_threshold(self) -> None:
        lines = [Line(text="Rare line", font_size=11.0, page_number=1)]
        for page in range(2, 11):
            lines.append(Line(text=f"Body {page}", font_size=11.0, page_number=page))

        filtered = filter_boilerplate_lines(lines, threshold=0.2)

        assert "Rare line" in {line.text for line in filtered}

    def test_empty_input_returns_empty(self) -> None:
        assert filter_boilerplate_lines([]) == []

    def test_short_document_does_not_strip_all_content(self) -> None:
        """The real bug this guards against: on a 1-page document, every
        line trivially appears on 100% of pages. Without an absolute
        occurrence floor (not just a fraction), this would wrongly strip
        all content from short documents - found via a genuinely failing
        test against a single-page synthetic PDF, not anticipated upfront.
        """
        lines = [
            Line(text="Heading", font_size=16.0, page_number=1),
            Line(text="Body content.", font_size=11.0, page_number=1),
        ]

        filtered = filter_boilerplate_lines(lines, threshold=0.2)

        assert len(filtered) == 2
