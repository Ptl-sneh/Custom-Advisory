import json
from typing import Optional
from pydantic import BaseModel
import chromadb
from langchain_ollama import OllamaEmbeddings

from config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CONFIDENCE_THRESHOLD,
    # MIN_CHUNKS_ABOVE_THRESHOLD,
)
from ingestion.embedder import get_chroma_client, get_collection
from schemas.advisory import SourceReference
from logger import get_logger

logger = get_logger(__name__)


class RetrievalResult(BaseModel):
    chunks: list[SourceReference]
    avg_confidence: float
    top_confidence: float
    is_confident: bool
    total_retrieved: int
    chunks_above_threshold: int = 0


# In retriever.py, update the retrieval function:


def retrieve(
    query: str,
    top_k: int = TOP_K,
    filters: Optional[dict] = None,
) -> RetrievalResult:

    logger.info(f"Retrieval started | query_len={len(query)} | top_k={top_k}")

    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        query_vector = embeddings_model.embed_query(query)
        logger.debug(f"Query embedded | vector_dim={len(query_vector)}")

        client = get_chroma_client()
        collection = get_collection(client)

        query_params = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = collection.query(**query_params)
        logger.debug(f"ChromaDB query complete | results={len(results['ids'][0])}")

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        logger.debug(f"RAW distances: {[round(d, 4) for d in distances]}")

        chunks: list[SourceReference] = []
        confidences: list[float] = []
        chunks_above_threshold = 0  # ← ADD THIS COUNTER

        for i, (chunk_id, doc_text, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB cosine distance is 0-2 range
            # Convert to similarity score (0-1)
            similarity = round(1 - (distance / 2), 4)  # ← FIXED: Use correct formula
            confidences.append(similarity)

            # Count chunks above threshold
            if similarity >= CONFIDENCE_THRESHOLD:
                chunks_above_threshold += 1  # ← INCREMENT COUNTER

            try:
                tags = json.loads(meta.get("tags", "[]"))
            except Exception:
                tags = []

            page_num = meta.get("page_number", -1)
            chunks.append(
                SourceReference(
                    doc_id=meta.get("doc_id", ""),
                    source_name=meta.get("source_name", ""),
                    doc_type=meta.get("doc_type", ""),
                    reference_number=meta.get("reference_number") or None,
                    chunk_text=doc_text,
                    page_number=page_num if page_num != -1 else None,
                    similarity_score=similarity,
                )
            )

            logger.debug(
                f"Chunk {i+1} | source={meta.get('source_name')} | "
                f"raw_distance={distance:.4f} | similarity={similarity:.4f} | "
                f"above_threshold={similarity >= CONFIDENCE_THRESHOLD}"  # ← ADD DEBUG
            )

        avg_confidence = (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        )
        top_confidence = round(max(confidences), 4) if confidences else 0.0

        # Score spread check
        score_spread = (
            round(max(confidences) - min(confidences), 4) if confidences else 0.0
        )

        # Update is_confident logic - require at least one chunk above threshold
        is_confident = (
            chunks_above_threshold > 0
        )  # ← FIXED: Use counter instead of top_confidence >= threshold

        logger.info(
            f"Retrieval complete | chunks={len(chunks)} | "
            f"top_score={top_confidence} | avg_score={avg_confidence} | "
            f"spread={score_spread} | chunks_above_threshold={chunks_above_threshold} | "  # ← ADD LOGGING
            f"confident={is_confident}"
        )

        return RetrievalResult(
            chunks=chunks,
            avg_confidence=avg_confidence,
            top_confidence=top_confidence,
            is_confident=is_confident,
            total_retrieved=len(chunks),
            chunks_above_threshold=chunks_above_threshold,  # ← ADD THIS FIELD
        )

    except Exception as e:
        logger.error(f"Retrieval failed | error={str(e)}")
        raise
