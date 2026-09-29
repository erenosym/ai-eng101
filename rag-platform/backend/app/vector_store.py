from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


COLLECTION_NAME = "documents"
VECTOR_SIZE = 384

client = QdrantClient(
    host="localhost",
    port=6333
)


def create_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


def add_documents(
    texts: list[str],
    vectors: list[list[float]]
):
    points = []

    for index, (text, vector) in enumerate(zip(texts, vectors)):
        point = PointStruct(
            id=index,
            vector=vector,
            payload={
                "text": text
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def add_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    filename: str
):
    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = PointStruct(
            id=index,
            vector=vector,
            payload={
                "text": chunk,
                "filename": filename,
                "chunk_index": index
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def search_documents(
    query_vector: list[float],
    limit: int = 5,
    score_threshold: float = 0.45
):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True
    ).points

    return results
    