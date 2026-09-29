import logging
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from app.config import settings
from app.errors import QdrantUnavailableError, RetrievalError


logger = logging.getLogger(__name__)


COLLECTION_NAME = settings.qdrant_collection_name
SEED_COLLECTION_NAME = settings.qdrant_seed_collection_name
VECTOR_SIZE = 384
DEFAULT_RETRIEVAL_SCORE_THRESHOLD = settings.retrieval_score_threshold

client_options = {
    "timeout": settings.qdrant_timeout_seconds,
    "check_compatibility": False,
}
if settings.qdrant_url:
    client_options["url"] = settings.qdrant_url
else:
    client_options.update({
        "host": settings.qdrant_host,
        "port": settings.qdrant_port,
    })
client = QdrantClient(**client_options)


def create_collection(collection_name: str = COLLECTION_NAME):
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
    except Exception as exc:
        logger.exception("Qdrant collection initialization failed")
        raise QdrantUnavailableError() from exc


def create_chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a stable UUID for a chunk within one document ingestion."""
    return str(uuid5(UUID(document_id), f"chunk:{chunk_index}"))

def add_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    filename: str,
    document_id: str,
    collection_name: str = COLLECTION_NAME,
) -> list[str]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    points = []
    chunk_ids = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = create_chunk_id(document_id, index)
        point = PointStruct(
            id=chunk_id,
            vector=vector,
            payload={
                "text": chunk,
                "filename": filename,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
            }
        )

        points.append(point)
        chunk_ids.append(chunk_id)

    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as exc:
        logger.exception("Qdrant chunk upsert failed")
        raise QdrantUnavailableError() from exc

    return chunk_ids

def search_documents(
    query_vector: list[float],
    limit: int = 5,
    score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
):
    try:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        ).points
    except Exception as exc:
        logger.exception("Qdrant retrieval failed")
        raise RetrievalError() from exc


def scroll_documents(batch_size: int = 256):
    """Return all document points for building local retrieval indexes."""
    points = []
    offset = None

    while True:
        try:
            batch, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.exception("Qdrant scroll failed")
            raise RetrievalError() from exc
        points.extend(batch)
        if offset is None:
            return points


def qdrant_available() -> bool:
    try:
        client.get_collections()
        return True
    except Exception:
        logger.warning("Qdrant health check failed", exc_info=True)
        return False


def close_qdrant_client() -> None:
    try:
        client.close()
    except Exception:
        logger.warning("Failed to close Qdrant client cleanly", exc_info=True)
