"""Application exceptions exposed through structured API responses."""


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class UnsupportedDocumentTypeError(AppError):
    def __init__(self, suffix: str):
        label = suffix or "missing extension"
        super().__init__(
            "unsupported_document_type",
            f"Unsupported document type: {label}",
            415,
        )


class DocumentParsingError(AppError):
    def __init__(self):
        super().__init__("document_parsing_failed", "Document parsing failed", 422)


class EmbeddingError(AppError):
    def __init__(self):
        super().__init__("embedding_failed", "Embedding generation failed", 500)


class QdrantUnavailableError(AppError):
    def __init__(self, message: str = "Qdrant is unavailable"):
        super().__init__("qdrant_unavailable", message, 503)


class RetrievalError(AppError):
    def __init__(self):
        super().__init__("retrieval_failed", "Document retrieval failed", 503)


class OllamaUnavailableError(AppError):
    def __init__(self, message: str = "Ollama is unavailable"):
        super().__init__("ollama_unavailable", message, 503)
