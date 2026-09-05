"""Tests for rag_knowledge_assistant.retrieval.store.

Uses real Qdrant local/embedded instances (temp on-disk paths), not
mocks - Qdrant's local mode is genuinely fast and fully in-process, so
there's no reason to fake what we can actually run.
"""

from pathlib import Path

from qdrant_client import QdrantClient

from rag_knowledge_assistant.ingestion.chunk import Chunk
from rag_knowledge_assistant.retrieval.store import (
    create_collection,
    index_chunks,
    search,
)


def _make_chunk(text: str, heading: str, source: str, index: int) -> Chunk:
    return Chunk(
        text=text,
        section_heading=heading,
        source_document=source,
        page_number=1,
        chunk_index=index,
    )


class TestCreateCollection:
    def test_creates_collection(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)

        assert client.collection_exists("test_collection")

    def test_is_idempotent(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)
        create_collection(client, "test_collection", vector_size=4)

        assert client.collection_exists("test_collection")


class TestIndexAndSearch:
    def test_indexes_and_retrieves_by_similarity(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)

        chunks = [
            _make_chunk("Cost optimization content.", "COST01-BP01", "cost.pdf", 0),
            _make_chunk("Reliability content.", "REL01-BP01", "reliability.pdf", 0),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        written = index_chunks(client, "test_collection", chunks, embeddings)
        assert written == 2

        results = search(client, "test_collection", query_vector=[0.9, 0.1, 0.0, 0.0], limit=2)

        assert len(results) == 2
        assert results[0].text == "Cost optimization content."
        assert results[0].section_heading == "COST01-BP01"
        assert results[0].score > results[1].score

    def test_raises_on_length_mismatch(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)

        chunks = [_make_chunk("text", "heading", "doc.pdf", 0)]
        embeddings: list[list[float]] = []

        try:
            index_chunks(client, "test_collection", chunks, embeddings)
            raise AssertionError("Expected ValueError for mismatched lengths")
        except ValueError as e:
            assert "must be the same length" in str(e)

    def test_empty_chunks_returns_zero(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)

        assert index_chunks(client, "test_collection", [], []) == 0

    def test_reindexing_same_chunk_is_idempotent(self, tmp_path: Path) -> None:
        client = QdrantClient(path=str(tmp_path / "qdrant"))
        create_collection(client, "test_collection", vector_size=4)

        chunk = _make_chunk("Original text.", "Heading", "doc.pdf", 0)
        index_chunks(client, "test_collection", [chunk], [[1.0, 0.0, 0.0, 0.0]])

        updated_chunk = _make_chunk("Updated text.", "Heading", "doc.pdf", 0)
        index_chunks(client, "test_collection", [updated_chunk], [[1.0, 0.0, 0.0, 0.0]])

        results = search(client, "test_collection", query_vector=[1.0, 0.0, 0.0, 0.0], limit=10)

        assert len(results) == 1
        assert results[0].text == "Updated text."
