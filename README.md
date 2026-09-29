# AI Engineering Lab

A production-oriented Retrieval-Augmented Generation (RAG) application built with FastAPI, Qdrant, React, Ollama, and SentenceTransformers.

The project focuses on building an end-to-end document question-answering system rather than only wrapping an LLM API. Uploaded documents are parsed, chunked, embedded, stored in a vector database, retrieved semantically, and used as grounded context for a local language model.

## Features

- PDF, DOCX, and PPTX document ingestion
- Document parsing with MarkItDown
- Overlapping text chunking
- SentenceTransformer embeddings
- Qdrant vector database
- Semantic similarity search
- Configurable Top-K retrieval
- Similarity score threshold filtering
- Retrieval-Augmented Generation
- Grounded LLM answers using retrieved context
- Source metadata and citations
- Retrieval debug panel
- Conversation history
- Streaming LLM responses
- React + TypeScript frontend
- FastAPI backend
- Local LLM inference with Ollama
- Dockerized Qdrant service

## Architecture

```text
                       User
                        │
                        ▼
                React + TypeScript
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Document Upload          User Question
             │                     │
             ▼                     ▼
          FastAPI                FastAPI
             │                     │
             ▼                     ▼
         MarkItDown         SentenceTransformer
             │                     │
             ▼                     ▼
       Markdown Text          Query Embedding
             │                     │
             ▼                     │
          Chunking                  │
             │                     │
             ▼                     │
    SentenceTransformer            │
             │                     │
             ▼                     ▼
       Chunk Embeddings ───────► Qdrant
                                   │
                                   ▼
                          Top-K Semantic Retrieval
                                   │
                                   ▼
                           Score Threshold Filter
                                   │
                                   ▼
                           Retrieved Context
                                   │
                                   ▼
                              Ollama LLM
                                   │
                                   ▼
                          Streaming RAG Answer
                                   │
                                   ▼
                        Answer + Source Metadata
```

## RAG Pipeline

### 1. Document ingestion

Users can upload supported document formats through the frontend.

The backend converts the document into Markdown using MarkItDown.

```text
PDF / DOCX / PPTX
        ↓
     MarkItDown
        ↓
   Markdown text
```

### 2. Chunking

The extracted text is split into overlapping chunks.

Current configuration:

```text
chunk size: 1200 characters
overlap: 200 characters
```

Overlap helps preserve context around chunk boundaries.

### 3. Embeddings

Each chunk is converted into a dense vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings.

```text
Text Chunk
    ↓
SentenceTransformer
    ↓
384-dimensional vector
```

### 4. Vector storage

Embeddings are stored in Qdrant together with metadata.

Each point contains:

```json
{
  "vector": "...",
  "payload": {
    "text": "chunk content",
    "filename": "document.pdf",
    "chunk_index": 12
  }
}
```

### 5. Retrieval

When the user asks a question:

```text
Question
   ↓
Query Embedding
   ↓
Qdrant Similarity Search
   ↓
Top-K Relevant Chunks
```

Cosine similarity is used for vector comparison.

A configurable similarity threshold is applied to reject low-relevance retrieval results.

Current baseline:

```text
score_threshold = 0.35
```

This value is currently treated as a baseline and can later be optimized using retrieval evaluation.

### 6. Grounded generation

Retrieved chunks are combined into a context and passed to the local LLM.

The system prompt instructs the model to answer only using the provided context.

```text
Retrieved Chunks
      ↓
RAG Context
      ↓
Ollama LLM
      ↓
Grounded Answer
```

If sufficiently relevant context is not available, the system rejects the query instead of forcing the LLM to generate an unsupported answer.

## Streaming Responses

The backend exposes a streaming RAG endpoint.

Instead of waiting for the complete LLM response, generated text is sent incrementally to the frontend.

```text
Ollama
  ↓
stream=True
  ↓
FastAPI StreamingResponse
  ↓
NDJSON events
  ↓
React ReadableStream
  ↓
Live assistant response
```

The stream also sends retrieval metadata so the frontend can display the source chunks used for generation.

## Retrieval Debugging

Each assistant answer can expose retrieval information through the frontend debug panel.

Example:

```text
Retrieval debug

thesis.pdf
Chunk: 18
Similarity: 0.672

thesis.pdf
Chunk: 21
Similarity: 0.638
```

This makes it possible to inspect whether poor answers originate from retrieval or generation.

## Project Structure

```text
ai-engineering-lab/
│
└── rag-platform/
    │
    ├── backend/
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── embeddings.py
    │   │   ├── vector_store.py
    │   │   ├── document_parser.py
    │   │   ├── chunking.py
    │   │   ├── llm.py
    │   │   └── seed.py
    │   │
    │   ├── requirements.txt
    │   └── .venv/
    │
    └── frontend/
        ├── src/
        │   ├── App.tsx
        │   └── App.css
        │
        ├── package.json
        └── vite.config.ts
```

## Tech Stack

### Backend

- Python
- FastAPI
- Pydantic
- SentenceTransformers
- Qdrant Python Client
- MarkItDown
- Ollama

### Frontend

- React
- TypeScript
- Vite

### Infrastructure

- Docker
- Qdrant

## Running the Project

### 1. Start Qdrant

From the project root:

```bash
docker run -d \
  --name ai-lab-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

### 2. Start Ollama

Make sure Ollama is installed and the configured model is available.

Example:

```bash
ollama list
```

The current backend configuration uses:

```text
qwen2.5:7b
```

### 3. Start the backend

```bash
cd rag-platform/backend

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8001
```

FastAPI documentation:

```text
http://127.0.0.1:8001/docs
```

### 4. Start the frontend

In another terminal:

```bash
cd rag-platform/frontend

npm install

npm run dev
```

Open:

```text
http://localhost:5173
```

## Current API Endpoints

### Health check

```http
GET /health
```

### Generate an embedding

```http
POST /embedding
```

### Parse and index a document

```http
POST /documents/parse
```

### Semantic search

```http
POST /search
```

### RAG answer

```http
POST /ask
```

### Streaming RAG answer

```http
POST /ask/stream
```

## Example Workflow

1. Start Qdrant.
2. Start Ollama.
3. Start the FastAPI backend.
4. Start the React frontend.
5. Upload a PDF, DOCX, or PPTX document.
6. The document is parsed and chunked.
7. Chunk embeddings are stored in Qdrant.
8. Ask a question about the uploaded document.
9. Qdrant retrieves the most relevant chunks.
10. The chunks are used as context for the local LLM.
11. The generated answer streams into the UI.
12. Retrieval metadata can be inspected through the debug panel.

## Current Limitations

- Chunking is currently character-based rather than token-aware or structure-aware.
- Similarity threshold is manually configured.
- Retrieval currently uses dense semantic search only.
- No reranking layer is implemented yet.
- No lexical or hybrid retrieval is implemented yet.
- Conversation state currently exists only in the frontend session.
- Uploaded document metadata is not persisted in a relational database.
- Scanned/image-only documents may require a dedicated OCR preprocessing pipeline.
- Authentication and multi-user isolation are not implemented.

## Planned Improvements

- PostgreSQL document and conversation persistence
- Hybrid lexical + semantic search
- PostgreSQL full-text search
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Retrieval evaluation with Recall@K and MRR
- Generation faithfulness evaluation
- Latency benchmarking
- Chunk-size and Top-K experiments
- Docker Compose environment
- Improved citation rendering
- Token-aware or structure-aware chunking
- MCP tool integration
- Tool-calling agent workflows

## Engineering Goals

This project is designed to explore the engineering decisions behind production-oriented AI systems, including:

- Why vector databases are used
- How chunk size affects retrieval
- How Top-K affects context quality
- How similarity thresholds affect recall and precision
- How retrieval and generation failures can be separated
- How local LLMs can be integrated behind an API
- How streaming responses improve application UX
- How retrieval metadata improves observability and debugging

The goal is not only to produce a working RAG demo, but also to measure, understand, and explain the trade-offs behind the system.