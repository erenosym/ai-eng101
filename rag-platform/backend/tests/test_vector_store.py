import importlib
import asyncio
import json
import sys
import types
import unittest
from uuid import UUID, uuid4


class FakePointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.collections = {}
        self.last_query = None

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {}

    def upsert(self, collection_name, points):
        collection = self.collections.setdefault(collection_name, {})
        for point in points:
            collection[point.id] = point

    def query_points(self, **kwargs):
        self.last_query = kwargs
        points = list(self.collections.get(kwargs["collection_name"], {}).values())
        results = [
            types.SimpleNamespace(id=point.id, payload=point.payload, score=0.9)
            for point in points[:kwargs["limit"]]
        ]
        return types.SimpleNamespace(points=results)

    def get_collections(self):
        return types.SimpleNamespace(collections=[])

    def close(self):
        pass


class VectorStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qdrant_module = types.ModuleType("qdrant_client")
        models_module = types.ModuleType("qdrant_client.models")
        qdrant_module.QdrantClient = FakeQdrantClient
        models_module.Distance = types.SimpleNamespace(COSINE="cosine")
        models_module.VectorParams = lambda **kwargs: kwargs
        models_module.PointStruct = FakePointStruct
        sys.modules["qdrant_client"] = qdrant_module
        sys.modules["qdrant_client.models"] = models_module
        cls.vector_store = importlib.import_module("app.vector_store")

    def setUp(self):
        self.vector_store.client.collections.clear()
        self.vector_store.create_collection()

    def test_two_documents_do_not_overwrite_each_other(self):
        document_a = str(uuid4())
        document_b = str(uuid4())

        ids_a = self.vector_store.add_chunks(
            ["a0", "a1"], [[0.1], [0.2]], "a.pdf", document_a
        )
        ids_b = self.vector_store.add_chunks(
            ["b0", "b1"], [[0.3], [0.4]], "b.pdf", document_b
        )

        points = self.vector_store.client.collections["documents"]
        self.assertEqual(len(points), 4)
        self.assertTrue(set(ids_a).isdisjoint(ids_b))
        self.assertTrue(all(str(UUID(chunk_id)) == chunk_id for chunk_id in points))

        payloads_a = [points[chunk_id].payload for chunk_id in ids_a]
        payloads_b = [points[chunk_id].payload for chunk_id in ids_b]
        self.assertEqual({p["document_id"] for p in payloads_a}, {document_a})
        self.assertEqual({p["document_id"] for p in payloads_b}, {document_b})
        self.assertEqual([p["chunk_index"] for p in payloads_a], [0, 1])
        self.assertTrue(all(p["chunk_id"] in points for p in payloads_a + payloads_b))

    def test_search_uses_shared_default_but_remains_configurable(self):
        self.vector_store.search_documents([0.1], limit=10)
        self.assertEqual(
            self.vector_store.client.last_query["score_threshold"], 0.35
        )

        self.vector_store.search_documents([0.1], score_threshold=0.2)
        self.assertEqual(
            self.vector_store.client.last_query["score_threshold"], 0.2
        )

    def test_rejects_mismatched_chunks_and_vectors(self):
        with self.assertRaises(ValueError):
            self.vector_store.add_chunks(
                ["one", "two"], [[0.1]], "bad.pdf", str(uuid4())
            )


class ApiCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qdrant_module = types.ModuleType("qdrant_client")
        models_module = types.ModuleType("qdrant_client.models")
        qdrant_module.QdrantClient = FakeQdrantClient
        models_module.Distance = types.SimpleNamespace(COSINE="cosine")
        models_module.VectorParams = lambda **kwargs: kwargs
        models_module.PointStruct = FakePointStruct
        sys.modules["qdrant_client"] = qdrant_module
        sys.modules["qdrant_client.models"] = models_module

        class FakeFastAPI:
            def __init__(self, **kwargs):
                pass

            def add_middleware(self, *args, **kwargs):
                pass

            def _decorator(self, *args, **kwargs):
                return lambda function: function

            post = get = on_event = exception_handler = _decorator

        class FakeBaseModel:
            model_fields = {}

            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.model_fields = dict(getattr(cls, "__annotations__", {}))

            def __init__(self, **kwargs):
                for name in self.model_fields:
                    value = kwargs.get(name, getattr(type(self), name, None))
                    default = getattr(type(self), name, None)
                    if isinstance(value, str) and default is not None:
                        if isinstance(default, bool):
                            value = value.lower() in {"1", "true", "yes"}
                        elif isinstance(default, (int, float)):
                            value = type(default)(value)
                    setattr(self, name, value)

            @classmethod
            def model_validate(cls, values):
                return cls(**values)

        def fake_field(default=None, **kwargs):
            return default

        def fake_field_validator(*args, **kwargs):
            return lambda function: function

        class FakeStreamingResponse:
            def __init__(self, content, media_type):
                self.body_iterator = content
                self.media_type = media_type

        class FakeJSONResponse:
            def __init__(self, status_code, content):
                self.status_code = status_code
                self.content = content

        fastapi_module = types.ModuleType("fastapi")
        fastapi_module.FastAPI = FakeFastAPI
        fastapi_module.UploadFile = object
        fastapi_module.Request = object
        fastapi_module.File = lambda *args, **kwargs: None
        exceptions_module = types.ModuleType("fastapi.exceptions")
        exceptions_module.RequestValidationError = type(
            "RequestValidationError", (Exception,), {}
        )
        middleware_module = types.ModuleType("fastapi.middleware.cors")
        middleware_module.CORSMiddleware = object
        responses_module = types.ModuleType("fastapi.responses")
        responses_module.StreamingResponse = FakeStreamingResponse
        responses_module.JSONResponse = FakeJSONResponse
        pydantic_module = types.ModuleType("pydantic")
        pydantic_module.BaseModel = FakeBaseModel
        pydantic_module.Field = fake_field
        pydantic_module.field_validator = fake_field_validator
        parser_module = types.ModuleType("app.document_parser")
        parser_module.parse_document = lambda content, filename: content.decode()
        chunking_module = types.ModuleType("app.chunking")
        chunking_module.chunk_text = lambda text, chunk_size, overlap: [text]
        embeddings_module = types.ModuleType("app.embeddings")
        embeddings_module.create_embedding = lambda text: [0.1]
        llm_module = types.ModuleType("app.llm")
        llm_module.generate_answer = lambda question, context: f"answer: {context}"
        llm_module.generate_answer_stream = lambda question, context: iter(["answer"])
        llm_module.ollama_available = lambda: True
        llm_module.close_ollama_client = lambda: None

        sys.modules.update({
            "fastapi": fastapi_module,
            "fastapi.exceptions": exceptions_module,
            "fastapi.middleware.cors": middleware_module,
            "fastapi.responses": responses_module,
            "pydantic": pydantic_module,
            "app.document_parser": parser_module,
            "app.chunking": chunking_module,
            "app.embeddings": embeddings_module,
            "app.llm": llm_module,
        })
        cls.vector_store = importlib.import_module("app.vector_store")
        cls.main = importlib.import_module("app.main")
        cls.config = importlib.import_module("app.config")

    def setUp(self):
        self.vector_store.client.collections.clear()
        self.vector_store.create_collection()
        self.document_id = str(uuid4())
        self.chunk_id = self.vector_store.add_chunks(
            ["retrieved context"],
            [[0.1]],
            "document.pdf",
            self.document_id,
        )[0]

    def test_uploading_two_documents_preserves_both(self):
        class Upload:
            def __init__(self, filename, content):
                self.filename = filename
                self.content = content

            async def read(self):
                return self.content

        self.vector_store.client.collections["documents"].clear()
        result_a = asyncio.run(self.main.parse_uploaded_document(
            Upload("a.pdf", b"document a")
        ))
        result_b = asyncio.run(self.main.parse_uploaded_document(
            Upload("b.pdf", b"document b")
        ))

        points = self.vector_store.client.collections["documents"]
        self.assertEqual(len(points), 2)
        self.assertNotEqual(result_a["document_id"], result_b["document_id"])
        self.assertEqual(
            {point.payload["filename"] for point in points.values()},
            {"a.pdf", "b.pdf"},
        )

    def test_search_and_ask_still_return_results(self):
        request = self.main.SearchRequest(query="question")
        search_result = asyncio.run(self.main.search(request))
        ask_result = asyncio.run(self.main.ask(request))

        search_item = search_result["results"][0]
        ask_source = ask_result["sources"][0]
        self.assertEqual(search_item["filename"], "document.pdf")
        self.assertEqual(search_item["document_id"], self.document_id)
        self.assertEqual(search_item["chunk_id"], self.chunk_id)
        self.assertEqual(ask_source["document_id"], self.document_id)
        self.assertEqual(ask_source["chunk_id"], self.chunk_id)
        self.assertEqual(ask_source["chunk_index"], 0)
        self.assertIn("retrieved context", ask_result["answer"])

    def test_ask_stream_still_emits_sources_tokens_and_done(self):
        request = self.main.SearchRequest(query="question")
        response = asyncio.run(self.main.ask_stream(request))
        events = [json.loads(line) for line in response.body_iterator]

        self.assertEqual([event["type"] for event in events], [
            "sources", "token", "done"
        ])
        stream_source = events[0]["sources"][0]
        self.assertEqual(stream_source["filename"], "document.pdf")
        self.assertEqual(stream_source["document_id"], self.document_id)
        self.assertEqual(stream_source["chunk_id"], self.chunk_id)

    def test_health_reports_healthy_dependencies(self):
        self.main.qdrant_available = lambda: True
        self.main.ollama_available = lambda: True

        result = asyncio.run(self.main.health())

        self.assertEqual(result, {
            "status": "ok",
            "services": {"api": "ok", "qdrant": "ok", "ollama": "ok"},
        })

    def test_health_reports_degraded_dependency(self):
        self.main.qdrant_available = lambda: False
        self.main.ollama_available = lambda: True

        result = asyncio.run(self.main.health())

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["services"]["qdrant"], "unavailable")
        self.assertEqual(result["services"]["ollama"], "ok")

    def test_settings_defaults(self):
        settings = self.config.Settings()

        self.assertEqual(settings.qdrant_host, "localhost")
        self.assertEqual(settings.qdrant_port, 6333)
        self.assertEqual(settings.qdrant_collection_name, "documents")
        self.assertEqual(settings.ollama_model, "qwen2.5:7b")
        self.assertEqual(settings.retrieval_score_threshold, 0.35)

    def test_settings_reject_overlap_greater_than_or_equal_to_chunk_size(self):
        for overlap in (1200, 1201):
            with self.subTest(overlap=overlap):
                with self.assertRaisesRegex(
                    ValueError,
                    "CHUNK_OVERLAP must be less than CHUNK_SIZE",
                ):
                    self.config.Settings(chunk_size=1200, chunk_overlap=overlap)

    def test_error_response_structure(self):
        error = importlib.import_module("app.errors").UnsupportedDocumentTypeError(
            ".exe"
        )

        response = asyncio.run(self.main.app_error_handler(None, error))

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.content, {
            "error": {
                "code": "unsupported_document_type",
                "message": "Unsupported document type: .exe",
            }
        })


class HybridRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hybrid = importlib.import_module("app.hybrid_retrieval")
        cls.lexical = importlib.import_module("app.lexical_retrieval")

    def test_rrf_ranking_and_deduplication(self):
        dense = [
            {"chunk_id": "a", "text": "A"},
            {"chunk_id": "b", "text": "B"},
            {"chunk_id": "a", "text": "duplicate A"},
        ]
        lexical = [
            {"chunk_id": "b", "text": "B"},
            {"chunk_id": "c", "text": "C"},
        ]

        fused = self.hybrid.reciprocal_rank_fusion([dense, lexical], rrf_k=60)

        self.assertEqual([result["chunk_id"] for result in fused], ["b", "a", "c"])
        self.assertEqual(len(fused), 3)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 61)

    def test_hybrid_retrieval_preserves_chunk_metadata(self):
        lexical = self.lexical.BM25Retriever([{
            "text": "rare lexical phrase",
            "filename": "lexical.pdf",
            "document_id": "document-lexical",
            "chunk_id": "lexical-chunk",
            "chunk_index": 2,
        }])
        dense_point = types.SimpleNamespace(
            payload={
                "text": "dense result",
                "filename": "dense.pdf",
                "document_id": "document-dense",
                "chunk_id": "dense-chunk",
                "chunk_index": 1,
            },
            score=0.8,
        )
        retriever = self.hybrid.HybridRetriever(
            lexical,
            dense_search=lambda **kwargs: [dense_point],
        )

        results, timings = retriever.search(
            "rare phrase", [0.1], limit=2, candidate_pool=3
        )

        self.assertEqual(
            {result["chunk_id"] for result in results},
            {"dense-chunk", "lexical-chunk"},
        )
        for result in results:
            self.assertIn("filename", result)
            self.assertIn("document_id", result)
            self.assertIn("chunk_index", result)
            self.assertIn("text", result)
        self.assertEqual(set(timings), {
            "dense_retrieval_latency_ms",
            "lexical_retrieval_latency_ms",
            "rrf_fusion_latency_ms",
        })

    def test_reranking_integration_changes_order(self):
        reranker_module = importlib.import_module("app.reranker")

        class FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                return [0.1, 0.9]

        reranker = object.__new__(reranker_module.CrossEncoderReranker)
        reranker.model = FakeModel()
        candidates = [
            {"chunk_id": "first", "text": "first text"},
            {"chunk_id": "second", "text": "second text"},
        ]

        results = reranker.rerank("query", candidates, limit=2)

        self.assertEqual([result["chunk_id"] for result in results], [
            "second", "first"
        ])
        self.assertEqual(results[0]["rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
