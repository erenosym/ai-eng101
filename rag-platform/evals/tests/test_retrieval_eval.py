import sys
import types
import unittest
from pathlib import Path


EVALS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS_DIR))

from run_retrieval_eval import (  # noqa: E402
    DatasetValidationError,
    calculate_aggregate,
    evaluate_cases,
    score_query,
    validate_dataset,
)


class MetricTests(unittest.TestCase):
    def test_evaluation_searches_once_at_largest_cutoff(self):
        calls = []

        def search_documents(**kwargs):
            calls.append(kwargs)
            return [types.SimpleNamespace(
                payload={
                    "filename": "guide.pdf",
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                },
                score=0.9,
            )]

        results = evaluate_cases(
            [{
                "id": "case",
                "question": "Question?",
                "expected_document": "guide.pdf",
                "expected_chunk_ids": ["chunk-1"],
            }],
            [3, 5, 10],
            0.35,
            lambda question: [0.1],
            search_documents,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["limit"], 10)
        self.assertEqual(calls[0]["score_threshold"], 0.35)
        self.assertTrue(results[0]["success_at_k"]["3"])

    def test_document_level_match(self):
        case = {
            "expected_document": "guide.pdf",
            "expected_chunk_ids": None,
        }
        retrieved = [
            {"filename": "other.pdf", "chunk_id": "one"},
            {"filename": "guide.pdf", "chunk_id": "two"},
        ]

        metrics = score_query(case, retrieved, [3, 1, 5])

        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["success_at_k"], {
            "1": False,
            "3": True,
            "5": True,
        })

    def test_chunk_match_requires_filename_and_chunk_id(self):
        case = {
            "expected_document": "guide.pdf",
            "expected_chunk_ids": ["wanted"],
        }
        retrieved = [
            {"filename": "wrong.pdf", "chunk_id": "wanted"},
            {"filename": "guide.pdf", "chunk_id": "wrong"},
            {"filename": "guide.pdf", "chunk_id": "wanted"},
        ]

        metrics = score_query(case, retrieved, [3, 5, 10])

        self.assertEqual(metrics["first_relevant_rank"], 3)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1 / 3)

    def test_miss_has_zero_reciprocal_rank(self):
        case = {
            "expected_document": "missing.pdf",
            "expected_chunk_ids": None,
        }
        metrics = score_query(case, [], [3, 5, 10])

        self.assertIsNone(metrics["first_relevant_rank"])
        self.assertEqual(metrics["reciprocal_rank"], 0.0)
        self.assertFalse(any(metrics["success_at_k"].values()))

    def test_aggregate_metrics_and_latencies(self):
        query_results = [
            {
                "id": "first",
                "success_at_k": {"3": True, "5": True, "10": True},
                "reciprocal_rank": 1.0,
                "embedding_latency_ms": 10.0,
                "retrieval_latency_ms": 4.0,
                "latency_ms": 14.0,
            },
            {
                "id": "second",
                "success_at_k": {"3": False, "5": True, "10": True},
                "reciprocal_rank": 0.2,
                "embedding_latency_ms": 20.0,
                "retrieval_latency_ms": 8.0,
                "latency_ms": 28.0,
            },
            {
                "id": "third",
                "success_at_k": {"3": False, "5": False, "10": False},
                "reciprocal_rank": 0.0,
                "embedding_latency_ms": 30.0,
                "retrieval_latency_ms": 12.0,
                "latency_ms": 42.0,
            },
        ]

        aggregate = calculate_aggregate(query_results, [3, 5, 10])

        self.assertAlmostEqual(aggregate["recall_at_3"], 1 / 3)
        self.assertAlmostEqual(aggregate["recall_at_5"], 2 / 3)
        self.assertAlmostEqual(aggregate["recall_at_10"], 2 / 3)
        self.assertAlmostEqual(aggregate["mrr"], 0.4)
        self.assertEqual(aggregate["average_embedding_latency_ms"], 20.0)
        self.assertEqual(aggregate["average_retrieval_latency_ms"], 8.0)
        self.assertEqual(aggregate["average_latency_ms"], 28.0)
        self.assertEqual(aggregate["median_latency_ms"], 28.0)
        self.assertAlmostEqual(aggregate["p95_latency_ms"], 40.6)
        self.assertEqual(aggregate["failed_query_ids_at_k"], {
            "3": ["second", "third"],
            "5": ["third"],
            "10": ["third"],
        })

    def test_hybrid_reranking_integration(self):
        class FakeHybridRetriever:
            def search(self, **kwargs):
                return [
                    {"filename": "guide.pdf", "chunk_id": "wrong", "text": "a"},
                    {"filename": "guide.pdf", "chunk_id": "wanted", "text": "b"},
                ], {
                    "dense_retrieval_latency_ms": 1.0,
                    "lexical_retrieval_latency_ms": 2.0,
                    "rrf_fusion_latency_ms": 3.0,
                }

        class FakeReranker:
            def rerank(self, query, candidates, limit):
                return [candidates[1], candidates[0]][:limit]

        results = evaluate_cases(
            [{
                "id": "case",
                "question": "Question?",
                "expected_document": "guide.pdf",
                "expected_chunk_ids": ["wanted"],
            }],
            [1],
            0.35,
            lambda question: [0.1],
            lambda **kwargs: [],
            retrieval_mode="hybrid-rerank",
            hybrid_retriever=FakeHybridRetriever(),
            reranker=FakeReranker(),
            candidate_pool=2,
            rerank_candidates=2,
        )

        self.assertEqual(results[0]["first_relevant_rank"], 1)
        self.assertIn("reranking_latency_ms", results[0])
        self.assertEqual(results[0]["dense_retrieval_latency_ms"], 1.0)


class DatasetValidationTests(unittest.TestCase):
    def test_validates_optional_chunk_ids(self):
        cases = validate_dataset([
            {
                "id": "document-case",
                "question": "Question?",
                "expected_document": "guide.pdf",
            },
            {
                "id": "chunk-case",
                "question": "Another question?",
                "expected_document": "guide.pdf",
                "expected_chunk_ids": ["chunk-1"],
            },
        ])

        self.assertIsNone(cases[0]["expected_chunk_ids"])
        self.assertEqual(cases[1]["expected_chunk_ids"], ["chunk-1"])

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(DatasetValidationError, "duplicate case id"):
            validate_dataset([
                {"id": "same", "question": "One?", "expected_document": "a.pdf"},
                {"id": "same", "question": "Two?", "expected_document": "b.pdf"},
            ])

    def test_rejects_empty_chunk_id_list(self):
        with self.assertRaisesRegex(DatasetValidationError, "non-empty array"):
            validate_dataset([{
                "id": "case",
                "question": "Question?",
                "expected_document": "guide.pdf",
                "expected_chunk_ids": [],
            }])


if __name__ == "__main__":
    unittest.main()
