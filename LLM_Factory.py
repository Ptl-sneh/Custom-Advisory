import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

from config import settings
from schemas.common import (
    LLMProvider,
    EmbeddingProvider,
    QdrantMode,
    get_embedding_dimension,
)

logger = logging.getLogger(__name__)


# LLM FACTORY


def get_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """
    Returns a LangChain BaseChatModel for the configured provider.

    Args:
        provider: Override the provider from settings (optional).
        model: Override the model name from settings (optional).
        temperature: Sampling temperature. Default 0.0 for deterministic output.
        **kwargs: Any additional kwargs passed directly to the provider constructor.

    Returns:
        BaseChatModel instance ready to use in chains.

    Raises:
        ValueError: If provider is not supported.
    """
    provider = provider or settings.LLM_PROVIDER
    model = model or settings.LLM_MODEL

    logger.info(f"Initializing LLM | provider={provider} | model={model}")

    if provider == LLMProvider.OLLAMA:
        return get_ollama_llm(model, temperature, **kwargs)

    elif provider == LLMProvider.OPENAI:
        return get_openai_llm(model, temperature, **kwargs)

    elif provider == LLMProvider.GEMINI:
        return get_gemini_llm(model, temperature, **kwargs)

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported: {[p.value for p in LLMProvider]}"
        )


def get_ollama_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOllama(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        **kwargs,
    )


def get_openai_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        organization=settings.OPENAI_ORG_ID or None,
        temperature=temperature,
        **kwargs,
    )


def get_gemini_llm(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=temperature,
        **kwargs,
    )


# EMBEDDINGS FACTORY


def get_embeddings(
    provider: EmbeddingProvider | None = None,
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """
    Returns a LangChain Embeddings instance for the configured provider.

    Args:
        provider: Override the provider from settings (optional).
        model: Override the model name from settings (optional).
        **kwargs: Any additional kwargs passed directly to the provider constructor.

    Returns:
        Embeddings instance ready to use with Qdrant / retriever.

    Raises:
        ValueError: If provider is not supported or model dimension is unknown.
    """
    provider = provider or settings.EMBEDDING_PROVIDER
    model = model or settings.EMBEDDING_MODEL

    # Validate dimension is registered before proceeding.
    # Prevents silent failures when creating a Qdrant collection.
    get_embedding_dimension(model)

    logger.info(f"Initializing Embeddings | provider={provider} | model={model}")

    if provider == EmbeddingProvider.OLLAMA:
        return get_ollama_embeddings(model, **kwargs)

    elif provider == EmbeddingProvider.OPENAI:
        return get_openai_embeddings(model, **kwargs)

    elif provider == EmbeddingProvider.GEMINI:
        return get_gemini_embeddings(model, **kwargs)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported: {[p.value for p in EmbeddingProvider]}"
        )


def get_ollama_embeddings(model: str, **kwargs) -> Embeddings:
    return OllamaEmbeddings(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        **kwargs,
    )


def get_openai_embeddings(model: str, **kwargs) -> Embeddings:
    return OpenAIEmbeddings(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        **kwargs,
    )


def get_gemini_embeddings(model: str, **kwargs) -> Embeddings:
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=settings.GEMINI_API_KEY,
        **kwargs,
    )


# QDRANT CLIENT FACTORY


def get_qdrant_client() -> QdrantClient:
    """
    Returns a Qdrant client for the configured mode (local or cloud).

    Returns:
        QdrantClient instance.

    Raises:
        ValueError: If QDRANT_MODE is not supported.
    """
    if settings.QDRANT_MODE == QdrantMode.LOCAL:
        logger.info(
            f"Connecting to local Qdrant | "
            f"host={settings.QDRANT_HOST} | port={settings.QDRANT_PORT}"
        )
        return QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    elif settings.QDRANT_MODE == QdrantMode.CLOUD:
        logger.info(f"Connecting to Qdrant Cloud | url={settings.QDRANT_CLOUD_URL}")
        return QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    else:
        raise ValueError(f"Unsupported QDRANT_MODE: '{settings.QDRANT_MODE}'")


# CACHED SINGLETONS
# Use these in chain.py, retriever.py, embedder.py
# so instances are not re-created on every request.


@lru_cache()
def get_default_llm() -> BaseChatModel:
    """Cached default LLM instance using settings."""
    return get_llm()


@lru_cache()
def get_default_embeddings() -> Embeddings:
    """Cached default Embeddings instance using settings."""
    return get_embeddings()


@lru_cache()
def get_default_qdrant_client() -> QdrantClient:
    """Cached default Qdrant client instance."""
    return get_qdrant_client()
