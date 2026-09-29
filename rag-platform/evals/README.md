# Retrieval Evaluation

## Purpose

This benchmark measures retrieval quality: whether the retriever returns a labeled
relevant chunk and how highly that chunk is ranked. It does not evaluate the quality,
faithfulness, or correctness of the final LLM-generated answer.

## Evaluation Setup

- Dataset: 20 manually labeled, domain-specific questions
- Ground truth: `expected_chunk_ids` in `dataset.json`
- Comparison: the same questions and unchanged labels are used for every mode
- Chunk size: 1,200 characters
- Chunk overlap: 200 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Qdrant with cosine similarity
- Dense score threshold: `0.35`
- Dense mode: Qdrant semantic retrieval
- Hybrid mode: dense retrieval + BM25, combined with Reciprocal Rank Fusion (RRF)
- RRF constant: `k = 60`
- Candidate pool: 50 chunks
- Hybrid-rerank mode: hybrid candidates reranked with
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranking pool: 20 candidates

## Metrics

- **Hit Rate@K:** Fraction of questions for which at least one relevant chunk appears
  within the first K results. Earlier output called this Recall@K; because relevance is
  evaluated as query-level success rather than the fraction of all relevant chunks
  retrieved, Hit Rate@K is the more precise name.
- **MRR:** Mean reciprocal rank of the first relevant result. Higher values indicate
  that relevant chunks tend to appear earlier.
- **Latency:** Runtime of the retrieval pipeline. Hybrid-rerank latency includes the
  additional reranking stage. Latency depends on local hardware and model state.

## Results

| Retrieval mode | Hit Rate@3 | Hit Rate@5 | Hit Rate@10 | MRR | Median total latency | P95 total latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.85 | 0.85 | 0.90 | 0.7833 | — | — |
| Hybrid | 0.90 | 1.00 | 1.00 | 0.8475 | — | — |
| Hybrid + reranker | 1.00 | 1.00 | 1.00 | 0.9750 | ~85 ms | ~177.5 ms |

For hybrid-rerank, average reranking latency was approximately 102.8 ms.

## Interpretation

Dense retrieval provides a useful semantic baseline. Hybrid retrieval improves recall
by combining semantic similarity with lexical matching, recovering every labeled query
by rank 5 in this benchmark. The cross-encoder reranker further improves ordering by
scoring fused candidates directly against the question.

Hybrid + reranker achieved a Hit Rate of 1.00 at K=3, K=5, and K=10 across these 20
questions. This ranking improvement comes with additional model-inference latency.

## Failure Analysis

Dense retrieval missed some fact-heavy or exact-match questions where lexical evidence
was important. Hybrid retrieval recovered all labeled queries by K=5. Reranking then
moved difficult relevant chunks higher: for example, `thesis-001` was a dense miss, but
its relevant chunk was moved to rank 1 by the hybrid-rerank pipeline.

## Reproducibility

Run commands from the repository root. Qdrant must contain the corpus associated with
the unchanged labels in `evals/dataset.json`.

Dense:

```bash
rag-platform/backend/.venv/bin/python \
  rag-platform/evals/run_retrieval_eval.py \
  --retrieval-mode dense
```

Hybrid:

```bash
rag-platform/backend/.venv/bin/python \
  rag-platform/evals/run_retrieval_eval.py \
  --retrieval-mode hybrid \
  --candidate-pool 50 \
  --rrf-k 60
```

Hybrid + reranker:

```bash
rag-platform/backend/.venv/bin/python \
  rag-platform/evals/run_retrieval_eval.py \
  --retrieval-mode hybrid-rerank \
  --candidate-pool 50 \
  --rrf-k 60 \
  --rerank-candidates 20
```

Timestamped JSON reports are written to `rag-platform/evals/results/`.

## Limitations

- The benchmark contains only 20 labeled questions.
- It evaluates a single-document corpus.
- It measures retrieval, not end-to-end answer faithfulness or correctness.
- Results should not be generalized beyond this evaluation set.
