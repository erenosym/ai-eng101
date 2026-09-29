"""Environment-backed backend configuration."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # Allows lightweight tooling before dependencies are installed.
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env", override=False)


class Settings(BaseModel):
    qdrant_url: str | None = None
    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_timeout_seconds: int = Field(default=3, ge=1)
    qdrant_collection_name: str = "documents"
    qdrant_seed_collection_name: str = "seed_documents"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout_seconds: int = Field(default=60, ge=1)
    health_timeout_seconds: float = Field(default=2.0, gt=0)

    retrieval_score_threshold: float = 0.35
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    backend_host: str = "0.0.0.0"
    backend_port: int = Field(default=8001, ge=1, le=65535)
    frontend_origin: str = "http://localhost:5173"
    log_level: str = "INFO"

    def __init__(self, **data):
        super().__init__(**data)
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

    @classmethod
    def from_environment(cls) -> "Settings":
        values = {}
        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name in os.environ:
                value = os.environ[env_name]
                values[field_name] = None if value == "" else value
        return cls.model_validate(values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()


settings = get_settings()
