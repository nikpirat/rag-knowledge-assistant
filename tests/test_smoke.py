"""Smoke test for Phase 0 — proves package installs and imports correctly."""

from rag_knowledge_assistant import main


def test_main_runs_without_raising() -> None:
    main()
