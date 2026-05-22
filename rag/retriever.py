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
)
from ingestion.embedder import get_chroma_client, get_collection
from schemas.advisory import SourceReference
from logger import get_logger

logger = get_logger(__name__)


class RetrievalResult(BaseModel):
    chunks:           list[SourceReference]
    avg_confidence:   float
    top_confidence:   float
    is_confident:     bool
    total_retrieved:  int


def retrieve(
    query:   str,
    top_k:   int = TOP_K,
    filters: Optional[dict] = None,
) -> RetrievalResult:
    """
    Embed the query and retrieve top-k most similar chunks from ChromaDB.
    Returns a RetrievalResult with source references and confidence scores.
    """
    logger.info(f"Retrieval started | query_len={len(query)} | top_k={top_k}")

    try:
        # ── Embed the query ────────────────────────────────────────────────
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        try:
            query_vector = embeddings_model.embed_query(query)
            logger.debug(f"Query embedded | vector_dim={len(query_vector)}")
        except Exception as e:
            logger.info(f"Query embedding failed | error={str(e)}")
            raise

        # ── Query ChromaDB ─────────────────────────────────────────────────
        try:
            client     = get_chroma_client()
            collection = get_collection(client)

            query_params = {
                "query_embeddings": [query_vector],
                "n_results":        top_k,
                "include":          ["documents", "metadatas", "distances"],
            }

            if filters:
                query_params["where"] = filters

            results = collection.query(**query_params)
            logger.debug(f"ChromaDB query complete | results={len(results['ids'][0])}")

        except Exception as e:
            logger.info(f"ChromaDB query failed | error={str(e)}")
            raise

        # ── Parse results ──────────────────────────────────────────────────
        chunks:      list[SourceReference] = []
        confidences: list[float]           = []

        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (chunk_id, doc_text, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB cosine distance → similarity score (0 to 1)
            # distance=0 means identical, distance=2 means opposite
            similarity = round(1 - (distance / 2), 4)
            confidences.append(similarity)

            try:
                tags = json.loads(meta.get("tags", "[]"))
            except Exception:
                tags = []

            page_num = meta.get("page_number", -1)

            chunks.append(SourceReference(
                doc_id           = meta.get("doc_id", ""),
                source_name      = meta.get("source_name", ""),
                doc_type         = meta.get("doc_type", ""),
                reference_number = meta.get("reference_number") or None,
                chunk_text       = doc_text,
                page_number      = page_num if page_num != -1 else None,
                similarity_score = similarity,
            ))

            logger.debug(f"Chunk {i+1} | source={meta.get('source_name')} | score={similarity}")

        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        top_confidence = round(max(confidences), 4) if confidences else 0.0
        is_confident   = top_confidence >= CONFIDENCE_THRESHOLD

        logger.info(
            f"Retrieval complete | chunks={len(chunks)} | "
            f"top_score={top_confidence} | avg_score={avg_confidence} | "
            f"confident={is_confident}"
        )

        return RetrievalResult(
            chunks          = chunks,
            avg_confidence  = avg_confidence,
            top_confidence  = top_confidence,
            is_confident    = is_confident,
            total_retrieved = len(chunks),
        )

    except Exception as e:
        logger.info(f"Retrieval failed | error={str(e)}")
        raise