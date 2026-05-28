
from typing import Optional
from pydantic import BaseModel

from config import (
    EMBEDDING_MODEL,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)
from ingestion.embedder import get_chroma_client, get_collection, EmbeddingManager
from rag.confidence_score import calculate_confidence, ConfidenceResult
from schemas.advisory import SourceReference
from logger import get_logger
from .bm25_manager import BM25Manager
from rag.confidence_score import calculate_confidence, ConfidenceResult, STRONG_MATCH_THRESHOLD

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
    lambda_param: float = 0.95,
) -> list[int]:
    """
    Maximal Marginal Relevance reranking.

    Balances relevance to query vs diversity among selected chunks.
    lambda_param=0.95 means 95% relevance, 5% diversity.
    Higher lambda → more relevance focused.
    Lower lambda  → more diversity focused.
    This helps prevent returning 5 chunks that are all near-duplicates of the same paragraph,
    but not 6 copies of the same paragraph.

    Returns indices (from candidate_indices) of selected chunks.
    """
    logger.info(f"Starting MMR reranking | top_k={top_k} | candidates={len(candidate_vectors) if candidate_vectors else 0} | lambda_param={lambda_param}")
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

        
    logger.info(f"Adjusted top_k based on query_words ({query}): {top_k}")

    try:
        # 1. Embed query
        prefixed_query = add_query_prefix(query)
        query_vector = embedding_manager.embed_query(prefixed_query)
        logger.debug(f"Query embedded | dim={len(query_vector)}")

        # 2. Fetch candidates from ChromaDB
        # Fetch 2x top_k so MMR has candidates to choose from
        fetch_k = max(top_k * 4, 20) if use_mmr else top_k
        logger.info(f"Fetching candidates from ChromaDB | fetch_k={fetch_k}")

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

        logger.info(f"ChromaDB returned {len(ids)} candidates")
        logger.info(f"Raw distances: {[round(d, 4) for d in distances]}")

        similarities = [round(1.0 - d, 4) for d in distances]

        # BM25 retrieval

        bm25_results = bm25_manager.search(
            query,
            top_k=10,
        )
        logger.info(f"BM25 retrieval returned {len(bm25_results) if bm25_results else 0} results")
        bm25_scores = {}
        if bm25_results:
            max_score = max(score for _, score in bm25_results)
            if max_score > 0:
                for idx, score in bm25_results:
                    chunk_id = bm25_manager.chunk_ids[idx]
                    bm25_scores[chunk_id] = score / max_score

        logger.info(f"Similarities (fixed): {similarities}")

        # 4. MMR reranking
        if use_mmr and embeddings and len(embeddings) >= top_k:
            selected_indices = mmr_rerank(
                query_vector=query_vector,
                candidate_vectors=embeddings,
                candidate_indices=list(range(len(ids))),
                top_k=top_k,
                lambda_param=0.95,
            )
            logger.info(f"MMR selected indices: {selected_indices}")
        else:
            selected_indices = list(range(min(top_k, len(ids))))

        # 5. Build SourceReference list
        chunks: list[SourceReference] = []
        final_similarities: list[float] = []
        final_doc_types: list[str] = []

        for idx in selected_indices:
            doc_text = documents[idx]
            meta = metadatas[idx]
            dense_score = similarities[idx]
            chunk_id = ids[idx]
            bm25_score = bm25_scores.get(
                chunk_id,
                0.0,
            )

            if bm25_score > 0:
                sim = round(
                    dense_score * 0.75 + bm25_score * 0.25,
                    4,
                )
            else:
                sim = dense_score

            page_num = meta.get("page_number", -1)

            logger.debug(f"page metadata={meta.get('page_number')}")

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

            final_doc_types.append(meta.get("doc_type", "Other"))

            logger.info(
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

        strong_chunks = sum(1 for s in final_similarities if s >= STRONG_MATCH_THRESHOLD)

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
