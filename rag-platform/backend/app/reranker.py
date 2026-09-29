"""Optional local cross-encoder reranking."""

from typing import Any, Protocol


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not candidates or limit <= 0:
            return []
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
        return scored[:limit]
