"""Tests for rag_knowledge_assistant.retrieval.index_corpus.

Fakes only embed_documents (the one piece requiring a live Ollama daemon,
which isn't available in this environment) - everything else (chunk
loading, batching, real Qdrant indexing) runs for real.
"""

import json
from pathlib import Path

import pytest

from rag_knowledge_assistant.retrieval import index_corpus


def _write_chunks_jsonl(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            record = {
                "text": f"Chunk content number {i}.",
                "section_heading": f"Heading {i}",
                "source_document": "test.pdf",
                "page_number": 1,
                "chunk_index": i,
            }
            f.write(json.dumps(record) + "\n")


class TestRunIndexing:
    def test_embeds_and_indexes_all_chunks_in_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_chunks_jsonl(chunks_path, count=10)

        call_count = 0

        def fake_embed_documents(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            return [[float(i)] + [0.0] * 1023 for i in range(len(texts))]

        monkeypatch.setattr(index_corpus, "embed_documents", fake_embed_documents)

        total = index_corpus.run_indexing(
            chunks_path=chunks_path,
            qdrant_path=tmp_path / "qdrant",
            batch_size=4,
        )

        assert total == 10
        assert call_count == 3

    def test_raises_when_no_chunks_found(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "empty.jsonl"
        chunks_path.write_text("")

        with pytest.raises(ValueError, match="No chunks found"):
            index_corpus.run_indexing(chunks_path=chunks_path, qdrant_path=tmp_path / "qdrant")
