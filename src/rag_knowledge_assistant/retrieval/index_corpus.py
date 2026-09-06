"""Embed every chunk in data/processed/chunks.jsonl and index it into Qdrant.

Batches embedding calls (rather than one request per chunk) to reduce
HTTP round-trip overhead against Ollama.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from rag_knowledge_assistant.config.settings import settings
from rag_knowledge_assistant.embeddings.ollama_embedder import embed_documents
from rag_knowledge_assistant.ingestion.chunk import Chunk
from rag_knowledge_assistant.retrieval.store import create_collection, index_chunks

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _load_chunks(jsonl_path: Path) -> list[Chunk]:
    chunks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    return chunks


def _default_qdrant_client_builder() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def run_indexing(
    chunks_path: Path,
    qdrant_client_builder: Callable[[], QdrantClient] = _default_qdrant_client_builder,
    collection_name: str = settings.qdrant_collection_name,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed and index every chunk. Returns the total number of points written.

    Args:
        qdrant_client_builder: producer of the Qdrant client. Defaults to
            a real server connection via Settings (Phase 6: containerized
            Qdrant, not local/embedded mode); tests pass a builder pointed
            at a temp local-mode client instead, same dependency-injection
            pattern used in serving/app.py.
            :param batch_size:
            :param collection_name:
            :param qdrant_client_builder:
            :param chunks_path:
    """
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    client = qdrant_client_builder()
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
    total = run_indexing(chunks_path=Path(settings.chunks_path))
    print(f"Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
