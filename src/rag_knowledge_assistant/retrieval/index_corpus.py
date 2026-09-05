"""Embed every chunk in data/processed/chunks.jsonl and index it into Qdrant.

Batches embedding calls (rather than one request per chunk) to reduce
HTTP round-trip overhead against Ollama.
"""

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from rag_knowledge_assistant.embeddings.ollama_embedder import embed_documents
from rag_knowledge_assistant.ingestion.chunk import Chunk
from rag_knowledge_assistant.retrieval.store import create_collection, index_chunks

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
COLLECTION_NAME = "well_architected_chunks"


def _load_chunks(jsonl_path: Path) -> list[Chunk]:
    chunks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    return chunks


def run_indexing(
    chunks_path: Path,
    qdrant_path: Path,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed and index every chunk. Returns the total number of points written."""
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    client = QdrantClient(path=str(qdrant_path))
    create_collection(client, collection_name)

    total_indexed = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [chunk.text for chunk in batch]

        logger.info(
            "Embedding batch %d-%d of %d", batch_start, batch_start + len(batch), len(chunks)
        )
        embeddings = embed_documents(texts)

        total_indexed += index_chunks(client, collection_name, batch, embeddings)

    logger.info("Indexed %d total chunks into '%s'", total_indexed, collection_name)
    return total_indexed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    total = run_indexing(
        chunks_path=Path("data/processed/chunks.jsonl"),
        qdrant_path=Path("qdrant_storage"),
    )
    print(f"Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
