import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from app.chunking import chunk_text
from app.config import settings
from app.document_parser import parse_document
from app.embeddings import create_embedding
from app.errors import (
    AppError,
    DocumentParsingError,
    EmbeddingError,
    OllamaUnavailableError,
    QdrantUnavailableError,
    RetrievalError,
    UnsupportedDocumentTypeError,
)
from app.llm import (
    close_ollama_client,
    generate_answer,
    generate_answer_stream,
    ollama_available,
)
from app.vector_store import (
    DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    add_chunks,
    close_qdrant_client,
    create_collection,
    qdrant_available,
    search_documents,
)


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting API (Qdrant=%s:%s, collection=%s, Ollama model=%s)",
        settings.qdrant_host,
        settings.qdrant_port,
        settings.qdrant_collection_name,
        settings.ollama_model,
    )
    try:
        await asyncio.to_thread(create_collection)
        logger.info("Qdrant collection is ready")
    except QdrantUnavailableError:
        logger.warning("API starting in degraded mode: Qdrant unavailable")
    try:
        yield
    finally:
        await asyncio.to_thread(close_qdrant_client)
        await asyncio.to_thread(close_ollama_client)
        logger.info("API shutdown complete")


app = FastAPI(
    title="AI Engineering Lab API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return error_response(exc.code, exc.message, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return error_response("validation_error", "Invalid request input", 422)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error(
        "Unexpected API error",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return error_response("internal_error", "Internal server error", 500)


class SearchRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


async def _check_dependency(check) -> bool:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(check),
            timeout=settings.health_timeout_seconds,
        )
    except Exception:
        return False


@app.get("/health")
async def health():
    qdrant_ok, ollama_ok = await asyncio.gather(
        _check_dependency(qdrant_available),
        _check_dependency(ollama_available),
    )
    services = {
        "api": "ok",
        "qdrant": "ok" if qdrant_ok else "unavailable",
        "ollama": "ok" if ollama_ok else "unavailable",
    }
    return {
        "status": "ok" if qdrant_ok and ollama_ok else "degraded",
        "services": services,
    }


def _embed(text: str) -> list[float]:
    try:
        return create_embedding(text)
    except EmbeddingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected embedding failure")
        raise EmbeddingError() from exc


def _retrieve(query_vector: list[float]):
    started = perf_counter()
    try:
        results = search_documents(
            query_vector=query_vector,
            limit=5,
            score_threshold=DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        logger.exception("Unexpected retrieval failure")
        raise RetrievalError() from exc
    logger.info(
        "Dense retrieval completed in %.2f ms with %d results",
        (perf_counter() - started) * 1000,
        len(results),
    )
    return results


@app.post("/documents/parse")
async def parse_uploaded_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise UnsupportedDocumentTypeError(suffix)

    logger.info("Document indexing started: type=%s", suffix)
    started = perf_counter()
    try:
        content = await file.read()
    except Exception as exc:
        logger.exception("Document upload read failed: type=%s", suffix)
        raise DocumentParsingError() from exc
    try:
        markdown = parse_document(content=content, filename=filename)
    except Exception as exc:
        logger.exception("Document parsing failed: type=%s", suffix)
        raise DocumentParsingError() from exc

    chunks = chunk_text(
        text=markdown,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    vectors = [_embed(chunk) for chunk in chunks]
    document_id = str(uuid4())
    add_chunks(
        chunks=chunks,
        vectors=vectors,
        filename=filename,
        document_id=document_id,
    )
    logger.info(
        "Document indexing completed: chunks=%d duration_ms=%.2f",
        len(chunks),
        (perf_counter() - started) * 1000,
    )
    return {
        "filename": filename,
        "document_id": document_id,
        "characters": len(markdown),
        "chunk_count": len(chunks),
        "stored_in_qdrant": True,
    }


@app.post("/embedding")
async def embedding(request: SearchRequest):
    vector = _embed(request.query)
    return {
        "text": request.query,
        "dimensions": len(vector),
        "preview": vector[:10],
    }


def _serialize_search_result(result):
    return {
        "text": result.payload["text"],
        "filename": result.payload.get("filename"),
        "document_id": result.payload.get("document_id"),
        "chunk_id": result.payload.get("chunk_id"),
        "chunk_index": result.payload.get("chunk_index"),
        "score": result.score,
    }


def _serialize_source(result):
    return {
        "filename": result.payload.get("filename"),
        "document_id": result.payload.get("document_id"),
        "chunk_id": result.payload.get("chunk_id"),
        "chunk_index": result.payload.get("chunk_index"),
        "score": result.score,
    }


@app.post("/search")
async def search(request: SearchRequest):
    results = _retrieve(_embed(request.query))
    return {
        "query": request.query,
        "results": [_serialize_search_result(result) for result in results],
    }


@app.post("/ask")
async def ask(request: SearchRequest):
    results = _retrieve(_embed(request.query))
    if not results:
        return {
            "question": request.query,
            "answer": "Bu soruya cevap verecek yeterince ilgili bir bilgi dokümanlarda bulunamadı.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(result.payload["text"] for result in results)
    answer = generate_answer(question=request.query, context=context)
    return {
        "question": request.query,
        "answer": answer,
        "sources": [_serialize_source(result) for result in results],
    }


@app.post("/ask/stream")
async def ask_stream(request: SearchRequest):
    results = _retrieve(_embed(request.query))

    def stream_response():
        if not results:
            yield json.dumps({
                "type": "token",
                "content": "Bu soruya cevap verecek yeterince ilgili bir bilgi dokümanlarda bulunamadı.",
            }, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        yield json.dumps({
            "type": "sources",
            "sources": [_serialize_source(result) for result in results],
        }, ensure_ascii=False) + "\n"
        context = "\n\n---\n\n".join(result.payload["text"] for result in results)
        try:
            for token in generate_answer_stream(
                question=request.query,
                context=context,
            ):
                yield json.dumps(
                    {"type": "token", "content": token},
                    ensure_ascii=False,
                ) + "\n"
        except OllamaUnavailableError as exc:
            yield json.dumps({
                "type": "error",
                "error": {"code": exc.code, "message": exc.message},
            }) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )
