# CV Bullet Options

- Built a local-first RAG platform with FastAPI, React, Qdrant, SentenceTransformers,
  and Ollama, supporting document ingestion, streaming grounded answers, citations,
  health checks, structured errors, and Docker Compose deployment.

- Implemented and compared dense vector retrieval, BM25 lexical retrieval, Reciprocal
  Rank Fusion, and local cross-encoder reranking against one unchanged, manually
  labeled 20-query benchmark.

- Improved Hit Rate@3 from 0.85 to 1.00 and MRR from 0.7833 to 0.9750 with hybrid
  retrieval and reranking on the project benchmark, while measuring and documenting
  the associated latency trade-off.

