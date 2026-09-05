"""Ingestion pipeline: extract + chunk every PDF in data/raw, persist the
result as JSONL - the clean interface into Phase 2's embedding step.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from rag_knowledge_assistant.ingestion.chunk import chunk_document
from rag_knowledge_assistant.ingestion.extract import extract_lines

logger = logging.getLogger(__name__)


def run_ingestion(raw_dir: Path, output_path: Path) -> int:
    """Process every PDF in raw_dir, writing all resulting chunks to a
    single JSONL file. Returns the total chunk count.
    """
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {raw_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0

    with output_path.open("w", encoding="utf-8") as f:
        for pdf_path in pdf_paths:
            lines = extract_lines(pdf_path)
            chunks = chunk_document(lines, source_document=pdf_path.name)

            for chunk in chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")
            total_chunks += len(chunks)

    logger.info(
        "Wrote %d total chunks from %d PDFs to %s", total_chunks, len(pdf_paths), output_path
    )
    return total_chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    total = run_ingestion(
        raw_dir=Path("data/raw"),
        output_path=Path("data/processed/chunks.jsonl"),
    )
    print(f"Total chunks written: {total}")


if __name__ == "__main__":
    main()
