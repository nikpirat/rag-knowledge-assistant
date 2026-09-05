"""Tests for rag_knowledge_assistant.ingestion.pipeline."""

import json
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag_knowledge_assistant.ingestion.pipeline import run_ingestion


def _build_simple_pdf(path: Path, heading: str, body: str) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, heading)
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, body)
    c.showPage()
    c.save()


class TestRunIngestion:
    def test_processes_all_pdfs_and_writes_jsonl(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _build_simple_pdf(raw_dir / "doc_a.pdf", "Heading A", "Body text A.")
        _build_simple_pdf(raw_dir / "doc_b.pdf", "Heading B", "Body text B.")

        output_path = tmp_path / "processed" / "chunks.jsonl"
        total = run_ingestion(raw_dir=raw_dir, output_path=output_path)

        assert total == 2
        assert output_path.exists()

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

        records = [json.loads(line) for line in lines]
        source_docs = {r["source_document"] for r in records}
        assert source_docs == {"doc_a.pdf", "doc_b.pdf"}

    def test_raises_when_no_pdfs_found(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No PDF files found"):
            run_ingestion(raw_dir=empty_dir, output_path=tmp_path / "out.jsonl")
