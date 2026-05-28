"""
config.py — Central configuration for Customs Advisory POC.

All settings are read from environment variables (via .env file).
Pydantic BaseSettings handles validation, type coercion, and defaults.

Usage:
    from config import settings
    print(settings.LLM_MODEL)
"""

import os
import re
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from schemas.common import LLMProvider, EmbeddingProvider, QdrantMode, AppEnv, LogLevel


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables / .env file.
    Grouped into sections for clarity.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # APP

    APP_ENV: AppEnv = AppEnv.DEVELOPMENT
    LOG_LEVEL: LogLevel = LogLevel.INFO

    # LLM

    LLM_PROVIDER: LLMProvider = LLMProvider.OLLAMA
    LLM_MODEL: str = "qwen2.5"

    # EMBEDDINGS

    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OLLAMA
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # OLLAMA

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # OPENAI
    
    OPENAI_API_KEY: str = ""
    OPENAI_ORG_ID: str = ""

    # GEMINI
    
    GEMINI_API_KEY: str = ""

    # QDRANT

    QDRANT_MODE: QdrantMode = QdrantMode.LOCAL
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_CLOUD_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_BASE_NAME: str = "customs_advisory"

    # LANGFUSE

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # PATHS

    BASE_DIR: str = Field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )

    @property
    def RAW_DOCS_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw")

    @property
    def PROCESSED_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "processed")

    @property
    def REVIEW_STORE_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, "review", "review_store.json")

    # CHUNKING
    
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # RETRIEVAL

    TOP_K: int = 6
    CONFIDENCE_THRESHOLD: float = 0.90    # minimum score to accept an answer
    OUT_OF_CONTEXT_THRESHOLD: float = 0.30  # below this → confidence = 0%

    # DERIVED PROPERTIES

    @property
    def QDRANT_COLLECTION_NAME(self) -> str:
        """
        Auto-generates collection name from base name + sanitized model name.
        e.g. nomic-embed-text → customs_advisory_nomic_embed_text
             text-embedding-3-small → customs_advisory_text_embedding_3_small
        """
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", self.EMBEDDING_MODEL).lower()
        return f"{self.QDRANT_COLLECTION_BASE_NAME}_{sanitized}"

    @property
    def IS_LANGFUSE_ENABLED(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.APP_ENV == AppEnv.PRODUCTION

    # VALIDATORS — catch misconfiguration at startup, not at runtime

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Settings":
        """
        Ensures required API keys are present for the selected providers.
        Fails at startup with a clear message rather than crashing mid-request.
        """
        if self.LLM_PROVIDER == LLMProvider.OPENAI and not self.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        if self.LLM_PROVIDER == LLMProvider.GEMINI and not self.GEMINI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set in .env"
            )
        if self.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI and not self.OPENAI_API_KEY:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        if self.EMBEDDING_PROVIDER == EmbeddingProvider.GEMINI and not self.GEMINI_API_KEY:
            raise ValueError(
                "EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY to be set in .env"
            )
        if self.QDRANT_MODE == QdrantMode.CLOUD and not self.QDRANT_CLOUD_URL:
            raise ValueError(
                "QDRANT_MODE=cloud requires QDRANT_CLOUD_URL to be set in .env"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use this everywhere instead of instantiating Settings() directly.

    Example:
        from config import settings
        print(settings.LLM_MODEL)
    """
    return Settings()


settings = get_settings()