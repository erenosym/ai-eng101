import json

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.document_parser import parse_document
from app.chunking import chunk_text
from app.vector_store import (
    create_collection,
    search_documents,
    add_chunks,
)
from app.llm import generate_answer, generate_answer_stream
from app.embeddings import create_embedding


RETRIEVAL_SCORE_THRESHOLD = 0.35


app = FastAPI(
    title="AI Engineering Lab API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/documents/parse")
async def parse_uploaded_document(
    file: UploadFile = File(...)
):
    content = await file.read()

    markdown = parse_document(
        content=content,
        filename=file.filename
    )

    chunks = chunk_text(
        text=markdown,
        chunk_size=1200,
        overlap=200
    )

    vectors = [
        create_embedding(chunk)
        for chunk in chunks
    ]

    add_chunks(
        chunks=chunks,
        vectors=vectors,
        filename=file.filename
    )

    return {
        "filename": file.filename,
        "characters": len(markdown),
        "chunk_count": len(chunks),
        "stored_in_qdrant": True
    }

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
        limit=5
    )

    return {
        "query": request.query,
        "results": [
            {
                "text": result.payload["text"],
                "filename": result.payload.get("filename"),
                "chunk_index": result.payload.get("chunk_index"),
                "score": result.score
            }
            for result in results
        ]
    }
@app.post("/ask")
async def ask(request: SearchRequest):

    query_vector = create_embedding(request.query)

    results = search_documents(
        query_vector=query_vector,
        limit=5,
        score_threshold=RETRIEVAL_SCORE_THRESHOLD
    )

    if not results:
        return {
            "question": request.query,
            "answer": "Bu soruya cevap verecek yeterince ilgili bir bilgi dokümanlarda bulunamadı.",
            "sources": []
        }

    context_parts = []

    for result in results:
        context_parts.append(result.payload["text"])

    context = "\n\n---\n\n".join(context_parts)

    answer = generate_answer(
        question=request.query,
        context=context
    )

    sources = [
        {
            "filename": result.payload.get("filename"),
            "chunk_index": result.payload.get("chunk_index"),
            "score": result.score
        }
        for result in results
    ]

    return {
        "question": request.query,
        "answer": answer,
        "sources": sources
    }

@app.post("/ask/stream")
async def ask_stream(request: SearchRequest):

    query_vector = create_embedding(request.query)

    results = search_documents(
        query_vector=query_vector,
        limit=5,
        score_threshold=0.35,
    )

    def stream_response():

        if not results:
            event = {
                "type": "token",
                "content": (
                    "Bu soruya cevap verecek yeterince ilgili "
                    "bir bilgi dokümanlarda bulunamadı."
                ),
            }

            yield json.dumps(event, ensure_ascii=False) + "\n"

            yield json.dumps({
                "type": "done"
            }) + "\n"

            return

        sources = [
            {
                "filename": result.payload.get("filename"),
                "chunk_index": result.payload.get("chunk_index"),
                "score": result.score,
            }
            for result in results
        ]

        # Önce kaynakları frontend'e gönder.
        yield json.dumps(
            {
                "type": "sources",
                "sources": sources,
            },
            ensure_ascii=False,
        ) + "\n"

        context = "\n\n---\n\n".join(
            result.payload["text"]
            for result in results
        )

        # Ardından LLM cevabını parça parça gönder.
        for token in generate_answer_stream(
            question=request.query,
            context=context,
        ):
            yield json.dumps(
                {
                    "type": "token",
                    "content": token,
                },
                ensure_ascii=False,
            ) + "\n"

        yield json.dumps({
            "type": "done"
        }) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )