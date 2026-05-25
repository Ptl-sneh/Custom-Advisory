"""
rag/retriever.py  —  Optimized Retriever
==========================================

Changes from original:
  1. FIXED similarity formula: 1 - distance (not 1 - distance/2)
     The old formula inflated every score — a real 0.70 showed as 0.85.
     ChromaDB cosine space: distance = 1 - cosine_similarity, so
     cosine_similarity = 1 - distance. That's it.

  2. EmbeddingManager singleton: no longer re-initializes OllamaEmbeddings
     on every retrieve() call.

  3. Uses confidence_score.py properly: the 4-signal confidence calculation
     now drives is_confident and human_review, not a manual threshold check.

  4. MMR reranking: after fetching top_k * 2 candidates, picks the most
     diverse top_k using Maximal Marginal Relevance. Avoids returning 6
     near-identical chunks about the same paragraph.

  5. Nomic prefix: if using nomic-embed-text, query must be prefixed with
     "search_query: " for correct asymmetric encoding.
"""

import json
import math
from typing import Optional
from pydantic import BaseModel

from config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)
from ingestion.embedder import get_chroma_client, get_collection, EmbeddingManager
from rag.confidence_score import calculate_confidence, ConfidenceResult
from schemas.advisory import SourceReference
from logger import get_logger
from .bm25_manager import BM25Manager

logger = get_logger(__name__)


bm25_manager = BM25Manager()
bm25_manager.load_index()

embedding_manager = EmbeddingManager()


class RetrievalResult(BaseModel):
    chunks: list[SourceReference]

    # Raw scores
    top_similarity: float
    avg_similarity: float

    # Confidence (from confidence_score.py — 4-signal)
    confidence_score: float
    retrieval_quality: float
    source_authority: float
    source_agreement: float
    coverage_bonus: float

    # Review
    is_confident: bool
    human_review_required: bool
    human_review_reason: Optional[str] = None

    total_retrieved: int
    strong_chunks: int  # chunks with similarity >= 0.70


# MMR Reranking


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Dot product of two already-normalized vectors = cosine similarity."""
    return sum(a * b for a, b in zip(v1, v2))


def mmr_rerank(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    candidate_indices: list[int],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """
    Maximal Marginal Relevance reranking.

    Balances relevance to query vs diversity among selected chunks.
    lambda_param=0.7 means 70% relevance, 30% diversity.
    Higher lambda → more relevance focused.
    Lower lambda  → more diversity focused.

    For legal docs: 0.7 is a good balance — we want relevant chunks
    but not 6 copies of the same paragraph.

    Returns indices (from candidate_indices) of selected chunks.
    """
    if not candidate_vectors or top_k <= 0:
        return candidate_indices[:top_k]

    selected = []
    remaining = list(range(len(candidate_vectors)))

    for _ in range(min(top_k, len(candidate_vectors))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # Relevance: similarity to query
            relevance = cosine_similarity(query_vector, candidate_vectors[idx])

            # Diversity: max similarity to already selected chunks
            if selected:
                redundancy = max(
                    cosine_similarity(candidate_vectors[idx], candidate_vectors[s])
                    for s in selected
                )
            else:
                redundancy = 0.0

            # MMR score
            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    # Map back to original candidate indices
    return [candidate_indices[i] for i in selected]


# Core retrieval


def add_query_prefix(query: str) -> str:
    """
    nomic-embed-text requires "search_query: " prefix at query time.
    Other models (MiniLM, BGE, E5) handle this differently —
    BGE/E5 use their own prefix in EmbeddingManager.embed_query().
    Only nomic needs it injected here.
    """
    if "nomic" in EMBEDDING_MODEL.lower():
        return f"search_query: {query}"
    return query


def retrieve(
    query: str,
    top_k: int = TOP_K,
    filters: Optional[dict] = None,
    use_mmr: bool = True,
) -> RetrievalResult:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query:    User's natural language question
        top_k:    Number of chunks to return (after MMR reranking)
        filters:  Optional ChromaDB metadata filter e.g. {"doc_type": "Case Law"}
        use_mmr:  If True, apply MMR reranking to reduce redundancy

    Returns:
        RetrievalResult with chunks, similarity scores, and confidence breakdown
    """
    logger.info(f"Retrieval started | query_len={len(query)} | top_k={top_k}")

    query_words = len(query.split())
    if query_words <= 5:
        top_k = 5
    elif query_words <= 12:
        top_k = 6
    else:
        top_k = 8

    try:
        # 1. Embed query
        prefixed_query = add_query_prefix(query)
        query_vector = embedding_manager.embed_query(prefixed_query)
        logger.debug(f"Query embedded | dim={len(query_vector)}")

        # 2. Fetch candidates from ChromaDB
        # Fetch 2x top_k so MMR has candidates to choose from
        fetch_k = max(top_k * 4, 20) if use_mmr else top_k

        client = get_chroma_client()
        collection = get_collection(client)

        query_params = {
            "query_embeddings": [query_vector],
            "n_results": min(fetch_k, collection.count()),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if filters:
            query_params["where"] = filters

        results = collection.query(**query_params)

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = results.get("embeddings", [[]])[0]  # for MMR

        logger.debug(f"ChromaDB returned {len(ids)} candidates")
        logger.debug(f"Raw distances: {[round(d, 4) for d in distances]}")

        # 3. FIX: Convert distance → similarity
        # ChromaDB cosine space: distance = 1 - cosine_similarity
        # Therefore: cosine_similarity = 1 - distance
        # The original code used: 1 - (distance / 2)  ← WRONG
        # That formula maps [0,2] → [0,1] but distances for normalized
        # vectors are already in [0,1] for positive cosine similarities.
        # Dividing by 2 was artificially halving the distance and
        # inflating every score by roughly 15-20 points.
        similarities = [round(1.0 - d, 4) for d in distances]
        logger.debug(f"Similarities (fixed): {similarities}")

        # 4. MMR reranking
        if use_mmr and embeddings and len(embeddings) >= top_k:
            selected_indices = mmr_rerank(
                query_vector=query_vector,
                candidate_vectors=embeddings,
                candidate_indices=list(range(len(ids))),
                top_k=top_k,
                lambda_param=0.7,
            )
            logger.debug(f"MMR selected indices: {selected_indices}")
        else:
            selected_indices = list(range(min(top_k, len(ids))))

        # 5. Build SourceReference list
        chunks: list[SourceReference] = []
        final_similarities: list[float] = []
        final_doc_types: list[str] = []

        for idx in selected_indices:
            doc_text = documents[idx]
            meta = metadatas[idx]
            sim = similarities[idx]

            page_num = meta.get("page_number", -1)

            chunks.append(
                SourceReference(
                    doc_id=meta.get("doc_id", ""),
                    source_name=meta.get("source_name", ""),
                    doc_type=meta.get("doc_type", ""),
                    reference_number=meta.get("reference_number") or None,
                    chunk_text=doc_text,
                    page_number=page_num if page_num != -1 else None,
                    similarity_score=sim,
                )
            )
            final_similarities.append(sim)

            MIN_SIMILARITY = 0.45
            filtered = [
                (chunk, sim, doc)
                for chunk, sim, doc in zip(chunks, final_similarities, final_doc_types)
                if sim >= MIN_SIMILARITY
            ]

            final_doc_types.append(meta.get("doc_type", "Other"))

            logger.debug(
                f"Chunk idx={idx} | source={meta.get('source_name')} | "
                f"sim={sim:.4f} | doc_type={meta.get('doc_type')}"
            )

        # 6. Confidence calculation (4-signal)
        # Use confidence_score.py — NOT a manual average
        conf: ConfidenceResult = calculate_confidence(
            similarity_scores=final_similarities,
            doc_types=final_doc_types,
            query=query,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        strong_chunks = sum(1 for s in final_similarities if s >= 0.70)

        logger.info(
            f"Retrieval complete | chunks={len(chunks)} | "
            f"top_sim={conf.signals.get('top_similarity')} | "
            f"confidence={conf.score} | strong_chunks={strong_chunks} | "
            f"review={conf.human_review_required}"
        )

        return RetrievalResult(
            chunks=chunks,
            top_similarity=conf.signals.get("top_similarity", 0.0),
            avg_similarity=conf.signals.get("weighted_mean_similarity", 0.0),
            confidence_score=conf.score,
            retrieval_quality=conf.retrieval_quality,
            source_authority=conf.source_authority,
            source_agreement=conf.source_agreement,
            coverage_bonus=conf.coverage_bonus,
            is_confident=not conf.human_review_required,
            human_review_required=conf.human_review_required,
            human_review_reason=conf.human_review_reason,
            total_retrieved=len(chunks),
            strong_chunks=strong_chunks,
        )

    except Exception as e:
        logger.error(f"Retrieval failed | error={str(e)}")
        raise
