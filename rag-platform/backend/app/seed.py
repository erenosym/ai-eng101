from uuid import uuid5, NAMESPACE_URL

from app.embeddings import create_embedding
from app.vector_store import (
    SEED_COLLECTION_NAME,
    add_chunks,
    create_collection,
)


documents = [
    "Python is widely used for machine learning and data science.",
    "PostgreSQL is an open-source relational database.",
    "Docker packages applications and their dependencies into containers.",
    "FastAPI is a modern Python framework for building APIs.",
    "Transformers use attention mechanisms to process sequences.",
    "Qdrant is a vector database designed for similarity search.",
]


def seed():
    create_collection(SEED_COLLECTION_NAME)

    vectors = [
        create_embedding(document)
        for document in documents
    ]

    add_chunks(
        chunks=documents,
        vectors=vectors,
        filename="synthetic-seed.txt",
        document_id=str(uuid5(NAMESPACE_URL, "ai-engineering-lab:synthetic-seed")),
        collection_name=SEED_COLLECTION_NAME,
    )


if __name__ == "__main__":
    seed()
