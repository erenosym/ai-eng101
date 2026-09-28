from app.embeddings import create_embedding
from app.vector_store import create_collection, add_documents


documents = [
    "Python is widely used for machine learning and data science.",
    "PostgreSQL is an open-source relational database.",
    "Docker packages applications and their dependencies into containers.",
    "FastAPI is a modern Python framework for building APIs.",
    "Transformers use attention mechanisms to process sequences.",
    "Qdrant is a vector database designed for similarity search.",
]


def seed():
    create_collection()

    vectors = [
        create_embedding(document)
        for document in documents
    ]

    add_documents(
        texts=documents,
        vectors=vectors
    )


if __name__ == "__main__":
    seed()