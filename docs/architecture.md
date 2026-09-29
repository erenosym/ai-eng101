# RAG Platform Architecture

## Components

| Component | Responsibility |
|---|---|
| React frontend | Document upload, questions, streamed answers, citations, and retrieval debugging |
| FastAPI API | Validation, ingestion orchestration, dense retrieval, answer generation, streaming, and health reporting |
| SentenceTransformers | `all-MiniLM-L6-v2` embeddings for document chunks and queries |
| Qdrant | Persistent cosine-vector index plus chunk identity and source metadata |
| BM25 evaluator | In-memory lexical ranking over the same Qdrant chunk payloads |
| RRF / reranker | Evaluation-only rank fusion and optional cross-encoder candidate ordering |
| Ollama | Host-managed `qwen2.5:7b` inference for grounded answers |

## Request and Ingestion Flow

1. The browser sends a supported document to `POST /documents/parse`.
2. FastAPI validates the extension, extracts text, and creates overlapping
   1,200-character chunks with 200-character overlap.
3. One UUID identifies the document; each chunk receives its own deterministic UUID
   derived from that document identity and chunk index.
4. SentenceTransformers embeds each chunk.
5. Qdrant stores each vector with `text`, `filename`, `document_id`, `chunk_id`, and
   `chunk_index`. The globally unique `chunk_id` is also the point ID, preventing
   later uploads from overwriting earlier documents.

## Retrieval and Answer Flow

The production endpoints embed the question once and query Qdrant's cosine index with
the configured top-K and score threshold. `/search` returns ranked chunks and metadata;
`/ask` and `/ask/stream` pass qualifying chunks to Ollama as grounding context and
return the same sources with the answer. The streaming endpoint emits NDJSON events so
the browser can render text incrementally.

The evaluation runner adds two isolated paths over the same corpus and labels:

- BM25 results and dense results are fused with RRF, which combines rank positions
  without treating lexical and vector scores as directly comparable.
- The hybrid-rerank mode applies a local cross-encoder to a bounded fused candidate
  set before calculating metrics.

## Docker Networking and Persistence

Compose runs three services: an Nginx-served React build, FastAPI, and Qdrant. The API
uses the Compose service name `qdrant:6333`; browser traffic uses the host-exposed API
port. Qdrant data is stored in the named `qdrant_data` volume and survives a normal
`docker compose down`.

Ollama stays on the host because model installation, storage, hardware acceleration,
and lifecycle are machine-specific. The API container reaches it through
`host.docker.internal:11434`, avoiding a large model image and duplicated model data.

## Design Rationale

Qdrant provides a purpose-built cosine vector index, metadata payloads, persistent
storage, and stable point identities without adding an application database. It is a
small operational fit for this retrieval-focused project.

Dense retrieval remains the production path even though hybrid + reranking scored
better on the benchmark. It is the established endpoint behavior, has fewer runtime
components, and avoids cross-encoder latency. Hybrid and reranking first remain
isolated evaluation stages so their quality gain, latency cost, corpus refresh
behavior, and operational implications can be validated before a production change.

