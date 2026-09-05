"""Qdrant vector store: indexing and retrieval."""

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_knowledge_assistant.ingestion.chunk import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its similarity score, decoupled from
    Qdrant's own point/payload types so callers don't depend on the
    vector store's internal representation.
    """

    text: str
    section_heading: str
    source_document: str
    page_number: int
    score: float


def create_collection(
    client: QdrantClient, collection_name: str, vector_size: int = EMBEDDING_DIM
) -> None:
    """Create the collection if it doesn't already exist. Idempotent."""
    if client.collection_exists(collection_name):
        logger.info("Collection '%s' already exists, skipping creation", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s' (dim=%d, cosine distance)", collection_name, vector_size)


def _chunk_point_id(chunk: Chunk) -> str:
    """Deterministic point ID derived from the chunk's source and index —
    re-indexing the same chunk produces the same ID, making indexing
    idempotent (re-running ingestion updates existing points instead of
    duplicating them).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_document}:{chunk.chunk_index}"))


def index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Upsert chunks and their embeddings into the collection. Returns the
    number of points written.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length"
        )
    if not chunks:
        return 0

    points = [
        PointStruct(
            id=_chunk_point_id(chunk),
            vector=embedding,
            payload={
                "text": chunk.text,
                "section_heading": chunk.section_heading,
                "source_document": chunk.source_document,
                "page_number": chunk.page_number,
            },
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Indexed %d chunks into '%s'", len(points), collection_name)
    return len(points)


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[SearchResult]:
    """Retrieve the top-k most similar chunks to the query vector."""
    results = client.query_points(collection_name=collection_name, query=query_vector, limit=limit)

    search_results = []
    for point in results.points:
        if point.payload is None:
            logger.warning("Point %s has no payload, skipping", point.id)
            continue
        search_results.append(
            SearchResult(
                text=point.payload["text"],
                section_heading=point.payload["section_heading"],
                source_document=point.payload["source_document"],
                page_number=point.payload["page_number"],
                score=point.score,
            )
        )
    return search_results
