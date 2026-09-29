#!/usr/bin/env python3
"""Evaluate dense, hybrid, and hybrid-plus-reranking retrieval."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable


EVALS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALS_DIR.parent
BACKEND_DIR = PROJECT_DIR / "backend"
DEFAULT_DATASET_PATH = EVALS_DIR / "dataset.json"
DEFAULT_RESULTS_DIR = EVALS_DIR / "results"
DEFAULT_TOP_K = (3, 5, 10)
RETRIEVAL_MODES = ("dense", "hybrid", "hybrid-rerank")


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset does not match the expected schema."""


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise DatasetValidationError("dataset must be a JSON array")

    validated = []
    seen_ids = set()
    for index, case in enumerate(data):
        location = f"case at index {index}"
        if not isinstance(case, dict):
            raise DatasetValidationError(f"{location} must be an object")
        for field in ("id", "question", "expected_document"):
            if field not in case:
                raise DatasetValidationError(f"{location} is missing '{field}'")
            if not _nonempty_string(case[field]):
                raise DatasetValidationError(
                    f"{location} field '{field}' must be a non-empty string"
                )
        if case["id"] in seen_ids:
            raise DatasetValidationError(f"duplicate case id: {case['id']!r}")
        seen_ids.add(case["id"])

        expected_chunk_ids = case.get("expected_chunk_ids")
        if expected_chunk_ids is not None:
            if not isinstance(expected_chunk_ids, list) or not expected_chunk_ids:
                raise DatasetValidationError(
                    f"{location} field 'expected_chunk_ids' must be a non-empty array"
                )
            if not all(_nonempty_string(chunk_id) for chunk_id in expected_chunk_ids):
                raise DatasetValidationError(
                    f"{location} expected_chunk_ids must contain non-empty strings"
                )
            if len(set(expected_chunk_ids)) != len(expected_chunk_ids):
                raise DatasetValidationError(
                    f"{location} expected_chunk_ids must not contain duplicates"
                )

        validated.append({
            "id": case["id"],
            "question": case["question"],
            "expected_document": case["expected_document"],
            "expected_chunk_ids": expected_chunk_ids,
        })
    return validated


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    return validate_dataset(data)


def validate_top_k(top_k: list[int] | tuple[int, ...]) -> list[int]:
    if not top_k or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in top_k
    ):
        raise ValueError("top_k values must be positive integers")
    return sorted(set(top_k))


def is_relevant(case: dict[str, Any], retrieved: dict[str, Any]) -> bool:
    if retrieved.get("filename") != case["expected_document"]:
        return False
    expected_chunk_ids = case.get("expected_chunk_ids")
    if expected_chunk_ids is None:
        return True
    return retrieved.get("chunk_id") in expected_chunk_ids


def score_query(
    case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    top_k: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    cutoffs = validate_top_k(top_k)
    first_relevant_rank = next(
        (
            rank
            for rank, result in enumerate(retrieved, start=1)
            if is_relevant(case, result)
        ),
        None,
    )
    return {
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        "success_at_k": {
            str(k): first_relevant_rank is not None and first_relevant_rank <= k
            for k in cutoffs
        },
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calculate_aggregate(
    query_results: list[dict[str, Any]],
    top_k: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    cutoffs = validate_top_k(top_k)
    latency_values = [result["latency_ms"] for result in query_results]
    aggregate = {
        **{
            f"recall_at_{k}": (
                mean(float(result["success_at_k"][str(k)]) for result in query_results)
                if query_results
                else 0.0
            )
            for k in cutoffs
        },
        "mrr": (
            mean(result["reciprocal_rank"] for result in query_results)
            if query_results
            else 0.0
        ),
        "average_latency_ms": mean(latency_values) if latency_values else 0.0,
        "median_latency_ms": median(latency_values) if latency_values else 0.0,
        "p95_latency_ms": percentile(latency_values, 0.95),
        "average_embedding_latency_ms": (
            mean(result["embedding_latency_ms"] for result in query_results)
            if query_results
            else 0.0
        ),
        "average_retrieval_latency_ms": (
            mean(result["retrieval_latency_ms"] for result in query_results)
            if query_results
            else 0.0
        ),
        "failed_query_ids_at_k": {
            str(k): [
                result["id"]
                for result in query_results
                if not result["success_at_k"][str(k)]
            ]
            for k in cutoffs
        },
    }

    optional_latency_fields = (
        "dense_retrieval_latency_ms",
        "lexical_retrieval_latency_ms",
        "rrf_fusion_latency_ms",
        "reranking_latency_ms",
    )
    for field in optional_latency_fields:
        values = [result[field] for result in query_results if field in result]
        if values:
            aggregate[f"average_{field}"] = mean(values)
    return aggregate


def _serialize_result(result: Any, rank: int) -> dict[str, Any]:
    if isinstance(result, dict):
        serialized = dict(result)
    else:
        payload = result.payload or {}
        serialized = {
            "text": payload.get("text"),
            "filename": payload.get("filename"),
            "document_id": payload.get("document_id"),
            "chunk_id": payload.get("chunk_id"),
            "chunk_index": payload.get("chunk_index"),
            "score": float(result.score),
        }
    serialized["rank"] = rank
    return serialized


def evaluate_cases(
    cases: list[dict[str, Any]],
    top_k: list[int] | tuple[int, ...],
    score_threshold: float,
    create_embedding: Callable[[str], list[float]],
    search_documents: Callable[..., list[Any]],
    retrieval_mode: str = "dense",
    hybrid_retriever: Any = None,
    reranker: Any = None,
    candidate_pool: int = 50,
    rrf_k: int = 60,
    rerank_candidates: int = 20,
) -> list[dict[str, Any]]:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")
    cutoffs = validate_top_k(top_k)
    max_top_k = max(cutoffs)
    if retrieval_mode != "dense" and candidate_pool < max_top_k:
        raise ValueError("candidate_pool must be at least max(top_k)")
    if retrieval_mode == "hybrid-rerank" and rerank_candidates < max_top_k:
        raise ValueError("rerank_candidates must be at least max(top_k)")

    query_results = []
    for case in cases:
        started = perf_counter()
        query_vector = create_embedding(case["question"])
        embedding_latency_ms = (perf_counter() - started) * 1000
        stage_latencies = {}

        if retrieval_mode == "dense":
            started = perf_counter()
            raw_results = search_documents(
                query_vector=query_vector,
                limit=max_top_k,
                score_threshold=score_threshold,
            )
            stage_latencies["dense_retrieval_latency_ms"] = (
                perf_counter() - started
            ) * 1000
        else:
            if hybrid_retriever is None:
                raise ValueError("hybrid_retriever is required for hybrid modes")
            hybrid_limit = (
                rerank_candidates
                if retrieval_mode == "hybrid-rerank"
                else max_top_k
            )
            raw_results, hybrid_latencies = hybrid_retriever.search(
                query=case["question"],
                query_vector=query_vector,
                limit=hybrid_limit,
                candidate_pool=candidate_pool,
                score_threshold=score_threshold,
                rrf_k=rrf_k,
            )
            stage_latencies.update(hybrid_latencies)

            if retrieval_mode == "hybrid-rerank":
                if reranker is None:
                    raise ValueError("reranker is required for hybrid-rerank mode")
                started = perf_counter()
                raw_results = reranker.rerank(
                    case["question"], raw_results, max_top_k
                )
                stage_latencies["reranking_latency_ms"] = (
                    perf_counter() - started
                ) * 1000

        retrieved = [
            _serialize_result(result, rank)
            for rank, result in enumerate(raw_results[:max_top_k], start=1)
        ]
        metrics = score_query(case, retrieved, cutoffs)
        retrieval_latency_ms = sum(stage_latencies.values())
        query_results.append({
            **case,
            "retrieved": retrieved,
            **metrics,
            "embedding_latency_ms": embedding_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "latency_ms": embedding_latency_ms + retrieval_latency_ms,
            **stage_latencies,
        })
    return query_results


def run_evaluation(
    dataset_path: Path,
    results_dir: Path,
    top_k: list[int] | tuple[int, ...],
    retrieval_mode: str = "dense",
    score_threshold: float | None = None,
    candidate_pool: int = 50,
    rrf_k: int = 60,
    rerank_candidates: int = 20,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> tuple[dict[str, Any], Path]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.vector_store import (
        DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
        search_documents,
    )

    cases = load_dataset(dataset_path)
    cutoffs = validate_top_k(top_k)
    threshold = (
        DEFAULT_RETRIEVAL_SCORE_THRESHOLD
        if score_threshold is None
        else score_threshold
    )
    create_embedding = None
    hybrid_retriever = None
    reranker = None
    setup_metadata = {}

    if cases:
        from app.embeddings import create_embedding, get_embedding_model

        # Keep model initialization outside per-query latency measurements.
        get_embedding_model()

    if cases and retrieval_mode != "dense":
        from app.hybrid_retrieval import HybridRetriever
        from app.lexical_retrieval import BM25Retriever

        started = perf_counter()
        lexical_retriever = BM25Retriever.from_qdrant()
        setup_metadata["lexical_index_build_latency_ms"] = (
            perf_counter() - started
        ) * 1000
        setup_metadata["lexical_index_chunk_count"] = len(lexical_retriever.chunks)
        hybrid_retriever = HybridRetriever(lexical_retriever)

    if cases and retrieval_mode == "hybrid-rerank":
        from app.reranker import CrossEncoderReranker

        started = perf_counter()
        reranker = CrossEncoderReranker(reranker_model)
        setup_metadata["reranker_initialization_latency_ms"] = (
            perf_counter() - started
        ) * 1000

    query_results = evaluate_cases(
        cases,
        cutoffs,
        threshold,
        create_embedding,
        search_documents,
        retrieval_mode=retrieval_mode,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        candidate_pool=candidate_pool,
        rrf_k=rrf_k,
        rerank_candidates=rerank_candidates,
    )
    generated_at = datetime.now(timezone.utc)
    metadata = {
        "generated_at": generated_at.isoformat(),
        "dataset": str(dataset_path.resolve()),
        "query_count": len(cases),
        "retrieval_mode": retrieval_mode,
        "top_k": cutoffs,
        "score_threshold": threshold,
        **setup_metadata,
    }
    if retrieval_mode != "dense":
        metadata.update({"candidate_pool": candidate_pool, "rrf_k": rrf_k})
    if retrieval_mode == "hybrid-rerank":
        metadata.update({
            "rerank_candidates": rerank_candidates,
            "reranker_model": reranker_model,
        })
    report = {
        "metadata": metadata,
        "aggregate": calculate_aggregate(query_results, cutoffs),
        "queries": query_results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / generated_at.strftime(
        f"retrieval_eval_{retrieval_mode}_%Y%m%dT%H%M%S_%fZ.json"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return report, output_path


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    metadata = report["metadata"]
    aggregate = report["aggregate"]
    print(f"Retrieval evaluation ({metadata['retrieval_mode']})")
    print(f"Queries: {metadata['query_count']}")
    print(f"Score threshold: {metadata['score_threshold']:.2f}")
    for k in metadata["top_k"]:
        print(f"Hit Rate / Recall@{k}: {aggregate[f'recall_at_{k}']:.3f}")
        failed = aggregate["failed_query_ids_at_k"][str(k)]
        print(f"  Failed query IDs: {', '.join(failed) if failed else 'none'}")
    print(f"MRR: {aggregate['mrr']:.4f}")
    print(f"Average latency: {aggregate['average_latency_ms']:.2f} ms")
    print(f"Median latency: {aggregate['median_latency_ms']:.2f} ms")
    print(f"P95 latency: {aggregate['p95_latency_ms']:.2f} ms")
    print(f"Results: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default="dense")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report, output_path = run_evaluation(
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            score_threshold=args.score_threshold,
            candidate_pool=args.candidate_pool,
            rrf_k=args.rrf_k,
            rerank_candidates=args.rerank_candidates,
            reranker_model=args.reranker_model,
        )
    except (DatasetValidationError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    print_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
