from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import create_embedding
from app.vector_store import (
    create_collection,
    search_documents,
)



app = FastAPI(
    title="AI Engineering Lab API",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_event():
    create_collection()


class SearchRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/embedding")
async def embedding(request: SearchRequest):
    vector = create_embedding(request.query)

    return {
        "text": request.query,
        "dimensions": len(vector),
        "preview": vector[:10]
    }

@app.post("/search")
async def search(request: SearchRequest):
    query_vector = create_embedding(request.query)

    results = search_documents(
        query_vector=query_vector,
        limit=3
    )

    return {
        "query": request.query,
        "results": [
            {
                "text": result.payload["text"],
                "score": result.score
            }
            for result in results
        ]
    }