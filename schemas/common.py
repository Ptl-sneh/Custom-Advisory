"""
common.py — Shared enums, constants, and base Pydantic models.

All values that are referenced in more than one place live here.
To change a value system-wide, change it once here — everything picks it up.
"""

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# PROVIDER ENUMS


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


class QdrantMode(str, Enum):
    """Qdrant deployment mode."""

    LOCAL = "local"
    CLOUD = "cloud"


# APP ENUMS


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# DOMAIN ENUMS


class DocType(str, Enum):
    """Freight forwarding / customs document types."""

    CIRCULAR = "Circular"
    NOTIFICATION = "Notification"
    TARIFF_SCHEDULE = "Tariff Schedule"
    HSN_CLASSIFICATION = "HSN Classification"
    CASE_LAW = "Case Law"
    BIS_EXPORT_CONTROL = "BIS / Export Control"
    CUSTOMS_ACT = "Customs Act"
    TRADE_POLICY = "Trade Policy"
    OTHER = "Other"


class IndexingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrievalStrategy(str, Enum):
    """How chunks are retrieved from the vector store."""

    DENSE = "dense"  # pure vector similarity
    SPARSE = "sparse"  # pure BM25 keyword
    HYBRID = "hybrid"  # dense + sparse combined (default)


# EMBEDDING MODEL DIMENSIONS
# Each embedding model produces a fixed-size vector.
# Qdrant collection must be created with the correct dimension.
# Add new models here as you onboard them.

EMBEDDING_DIMENSIONS: dict[str, int] = {
    # Ollama
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-large": 1024,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Gemini
    "models/embedding-001": 768,
    "models/text-embedding-004": 768,
}


def get_embedding_dimension(model_name: str) -> int:
    """
    Returns vector dimension for the given embedding model.
    Raises a clear error if the model is not registered.
    Add new models to EMBEDDING_DIMENSIONS above before using them.
    """
    if model_name not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model: '{model_name}'. "
            f"Add it to EMBEDDING_DIMENSIONS in common.py with its vector dimension."
        )
    return EMBEDDING_DIMENSIONS[model_name]


# CHUNKING CONSTANTS


class ChunkingConstants:
    """
    Chunking behaviour constants.
    Change here — chunker.py picks them up automatically.
    """

    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 120
    MIN_CHUNK_SIZE: int = 100  # chunks smaller than this are discarded
    MAX_CHUNK_SIZE: int = 2000  # safety cap
    SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


# RETRIEVAL CONSTANTS


class RetrievalConstants:
    """
    Retrieval behaviour constants.
    """

    DEFAULT_TOP_K: int = 6
    MAX_TOP_K: int = 20
    BM25_TOP_K: int = 6  # how many BM25 results to fetch
    DENSE_TOP_K: int = 6  # how many dense results to fetch
    RRF_K: int = 60  # Reciprocal Rank Fusion constant


# SCORING CONSTANTS


class ScoringConstants:
    """
    Confidence scoring thresholds.
    """

    CONFIDENCE_THRESHOLD: float = 0.90  # minimum to return a valid answer
    OUT_OF_CONTEXT_THRESHOLD: float = 0.30  # below this → confidence forced to 0.0
    MIN_CITATIONS_REQUIRED: int = 1  # at least one citation required
    SEMANTIC_WEIGHT: float = 0.6  # weight for semantic similarity score
    LEXICAL_WEIGHT: float = 0.4  # weight for BM25 / keyword score


# QDRANT CONSTANTS


class QdrantConstants:
    VECTOR_NAME: str = "dense"  # named vector for dense embeddings
    DEFAULT_DISTANCE: str = "Cosine"  # distance metric
    DEFAULT_HOST: str = "localhost"
    DEFAULT_PORT: int = 6333


# BASE PYDANTIC MODELS
class BaseResponse(BaseModel):
    success: bool
    message: str


class TimestampedModel(BaseModel):
    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None
