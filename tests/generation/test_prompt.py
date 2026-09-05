"""Tests for rag_knowledge_assistant.generation.prompt."""

from rag_knowledge_assistant.generation.prompt import (
    build_context_block,
    build_messages,
)
from rag_knowledge_assistant.retrieval.store import SearchResult


def _make_result(text: str, heading: str, source: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        text=text,
        section_heading=heading,
        source_document=source,
        page_number=1,
        score=score,
    )


class TestBuildContextBlock:
    def test_numbers_sources_starting_at_one(self) -> None:
        results = [
            _make_result("First content.", "Heading A", "doc_a.pdf"),
            _make_result("Second content.", "Heading B", "doc_b.pdf"),
        ]

        block = build_context_block(results)

        assert "[Source 1]" in block
        assert "[Source 2]" in block
        assert block.index("[Source 1]") < block.index("[Source 2]")

    def test_includes_source_document_and_heading(self) -> None:
        results = [_make_result("Content here.", "COST01-BP01", "cost.pdf")]

        block = build_context_block(results)

        assert "cost.pdf" in block
        assert "COST01-BP01" in block
        assert "Content here." in block

    def test_empty_results_returns_empty_string(self) -> None:
        assert build_context_block([]) == ""


class TestBuildMessages:
    def test_returns_system_and_user_messages(self) -> None:
        results = [_make_result("Some content.", "Heading", "doc.pdf")]

        messages = build_messages("What is X?", results)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_includes_question_and_sources(self) -> None:
        results = [_make_result("Relevant content.", "Heading", "doc.pdf")]

        messages = build_messages("How do I do X?", results)

        user_content = messages[1]["content"]
        assert "How do I do X?" in user_content
        assert "Relevant content." in user_content
        assert "[Source 1]" in user_content

    def test_system_message_instructs_grounding_and_citations(self) -> None:
        messages = build_messages("Question?", [])

        system_content = messages[0]["content"]
        assert "ONLY" in system_content
        assert "citation" in system_content.lower()

    def test_system_message_discourages_padding_and_repetition(self) -> None:
        messages = build_messages("Question?", [])

        system_content = messages[0]["content"]
        assert "repeat" in system_content.lower()
