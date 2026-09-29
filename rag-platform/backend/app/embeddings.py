import logging
from functools import lru_cache
from typing import Any

from app.config import settings
from app.errors import EmbeddingError


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def create_embedding(text: str) -> list[float]:
    try:
        embedding = get_embedding_model().encode(text)
        return embedding.tolist()
    except EmbeddingError:
        raise
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise EmbeddingError() from exc
