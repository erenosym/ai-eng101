"""Dense + BM25 retrieval combined with Reciprocal Rank Fusion."""

from time import perf_counter
from typing import Any, Callable

from app.lexical_retrieval import BM25Retriever, point_to_chunk
from app.vector_store import (
    DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    search_documents,
)


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_POOL = 50


def dense_point_to_chunk(point: Any) -> dict[str, Any]:
    return {**point_to_chunk(point), "dense_score": float(point.score)}


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    fused: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranking in rankings:
        seen_in_ranking = set()
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {**result, "rrf_score": 0.0}
                first_seen[chunk_id] = seen_counter
                seen_counter += 1
            else:
                for key, value in result.items():
                    if value is not None and key not in fused[chunk_id]:
                        fused[chunk_id][key] = value
            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda result: (-result["rrf_score"], first_seen[result["chunk_id"]]),
    )


class HybridRetriever:
    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        dense_search: Callable[..., list[Any]] = search_documents,
    ):
        self.lexical_retriever = lexical_retriever
        self.dense_search = dense_search

    def search(
        self,
        query: str,
        query_vector: list[float],
        limit: int,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if candidate_pool < limit:
            raise ValueError("candidate_pool must be greater than or equal to limit")

        started = perf_counter()
        dense_points = self.dense_search(
            query_vector=query_vector,
            limit=candidate_pool,
            score_threshold=score_threshold,
        )
        dense_latency_ms = (perf_counter() - started) * 1000
        dense_results = [dense_point_to_chunk(point) for point in dense_points]

        started = perf_counter()
        lexical_results = self.lexical_retriever.search(query, candidate_pool)
        lexical_latency_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        fused = reciprocal_rank_fusion([dense_results, lexical_results], rrf_k)
        fusion_latency_ms = (perf_counter() - started) * 1000

        return fused[:limit], {
            "dense_retrieval_latency_ms": dense_latency_ms,
            "lexical_retrieval_latency_ms": lexical_latency_ms,
            "rrf_fusion_latency_ms": fusion_latency_ms,
        }
