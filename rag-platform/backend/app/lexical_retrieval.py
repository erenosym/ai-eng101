"""Small in-memory BM25 retriever over Qdrant chunk payloads."""

import math
import re
from collections import Counter
from typing import Any, Callable

from app.vector_store import scroll_documents


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def point_to_chunk(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "text": payload.get("text", ""),
        "filename": payload.get("filename"),
        "document_id": payload.get("document_id"),
        "chunk_id": payload.get("chunk_id"),
        "chunk_index": payload.get("chunk_index"),
    }


class BM25Retriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.document_lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        document_frequencies = Counter()
        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())
        document_count = len(chunks)
        self.inverse_document_frequencies = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    @classmethod
    def from_qdrant(
        cls,
        load_points: Callable[[], list[Any]] = scroll_documents,
        **kwargs: Any,
    ) -> "BM25Retriever":
        chunks = [point_to_chunk(point) for point in load_points()]
        return cls(chunks, **kwargs)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        if not query_terms or not self.chunks or limit <= 0:
            return []

        scored = []
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                normalization = frequency + self.k1 * (
                    1 - self.b
                    + self.b * document_length / self.average_document_length
                )
                score += self.inverse_document_frequencies[term] * (
                    frequency * (self.k1 + 1) / normalization
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**self.chunks[index], "lexical_score": score}
            for score, index in scored[:limit]
        ]
