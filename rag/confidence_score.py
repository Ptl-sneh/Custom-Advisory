"""
confidence_score.py  —  Multi-Signal Confidence Calculator
============================================================

The Problem with the Original Approach
---------------------------------------
Most RAG systems make a critical mistake: they treat the average cosine similarity
of retrieved chunks as the confidence score. This fails for several reasons:

  1. Raw similarity scores are not calibrated probabilities
     - A score of 0.72 from all-MiniLM means something very different from
       0.72 with bge-large. The scale varies by model.

  2. Averaging hides the distribution
     - 5 chunks at [0.8, 0.8, 0.8, 0.8, 0.3] and [0.62, 0.62, 0.62, 0.62, 0.62]
       have the same mean (0.74), but the first set is clearly more reliable.

  3. It ignores non-similarity signals
     - Doc type authority (Customs Act > Blog post)
     - Source agreement (do multiple docs say the same thing?)
     - Query specificity (vague query = lower confidence regardless of scores)

This module replaces that with a proper multi-signal approach.

Confidence Score Formula
-------------------------
  confidence = (
      0.45 * retrieval_quality    +   # how good are the retrieved chunks?
      0.25 * source_authority     +   # how authoritative are the sources?
      0.20 * source_agreement     +   # do sources agree with each other?
      0.10 * coverage_bonus           # do we have enough chunks?
  )

Each signal is independently computed and normalized to [0, 1].
"""

import statistics
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Authority weights by doc_type
# Higher = more authoritative for legal/customs advisory
# ─────────────────────────────────────────────
DOC_TYPE_AUTHORITY: dict[str, float] = {
    "Customs Act":         1.00,
    "Tariff Schedule":     0.95,
    "HSN Classification":  0.90,
    "Notification":        0.85,
    "Circular":            0.80,
    "Case Law":            0.75,
    "Trade Policy":        0.70,
    "BIS / Export Control":0.65,
    "Other":               0.40,
}

# Thresholds
STRONG_MATCH_THRESHOLD = 0.70   # similarity >= this → strong retrieval signal
WEAK_MATCH_THRESHOLD   = 0.45   # similarity below this → weak / irrelevant chunk
MIN_CHUNKS_FOR_COVERAGE = 3     # ideal minimum chunks to consider coverage good


@dataclass
class ConfidenceResult:
    score: float                    # final 0.0–1.0 confidence score
    human_review_required: bool
    human_review_reason: Optional[str]
    # Component scores (for debugging / transparency)
    retrieval_quality: float
    source_authority: float
    source_agreement: float
    coverage_bonus: float
    signals: dict                   # raw signal values for logging


def calculate_confidence(
    similarity_scores: list[float],
    doc_types: list[str],
    query: str = "",
    confidence_threshold: float = 0.65,
) -> ConfidenceResult:
    """
    Calculate a calibrated confidence score from multiple signals.

    Args:
        similarity_scores: Cosine similarities for each retrieved chunk (0–1).
                           Must already be converted from ChromaDB distances
                           via: similarity = 1 - distance
        doc_types:         Doc type string for each retrieved chunk.
                           Must be same length as similarity_scores.
        query:             The user's query text (used for length heuristic).
        confidence_threshold: Below this → flag for human review.

    Returns:
        ConfidenceResult with score, human_review flag, and component breakdown.
    """
    if not similarity_scores:
        return ConfidenceResult(
            score=0.0,
            human_review_required=True,
            human_review_reason="No chunks retrieved — query may be out of scope",
            retrieval_quality=0.0,
            source_authority=0.0,
            source_agreement=0.0,
            coverage_bonus=0.0,
            signals={},
        )

    n = len(similarity_scores)
    scores = similarity_scores[:]
    scores.sort(reverse=True)

    # ─────────────────────────────────────────────
    # Signal 1: Retrieval Quality (weight 0.45)
    # Uses a weighted mean that gives more importance to top chunks.
    # Harmonic weights: rank 1 → weight 1.0, rank 2 → 0.5, rank 3 → 0.33, ...
    # ─────────────────────────────────────────────
    weights = [1.0 / (i + 1) for i in range(n)]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weight_total = sum(weights)
    weighted_mean = weighted_sum / weight_total

    # Penalize if even the top chunk is below weak threshold
    top_score = scores[0]
    if top_score < WEAK_MATCH_THRESHOLD:
        # Scale down aggressively — best chunk is barely relevant
        retrieval_quality = top_score * 0.5
    elif top_score < STRONG_MATCH_THRESHOLD:
        # Partial credit
        retrieval_quality = weighted_mean * 0.85
    else:
        # Good retrieval
        retrieval_quality = weighted_mean

    retrieval_quality = _clamp(retrieval_quality)

    # ─────────────────────────────────────────────
    # Signal 2: Source Authority (weight 0.25)
    # Average authority weight of retrieved doc types.
    # ─────────────────────────────────────────────
    authority_scores = [
        DOC_TYPE_AUTHORITY.get(dt, DOC_TYPE_AUTHORITY["Other"])
        for dt in doc_types
    ]
    source_authority = _clamp(statistics.mean(authority_scores))

    # ─────────────────────────────────────────────
    # Signal 3: Source Agreement (weight 0.20)
    # Measures how consistent the similarity scores are.
    # High variance = chunks are mixed quality (some relevant, some not)
    # → less trustworthy answer.
    # ─────────────────────────────────────────────
    if n >= 2:
        score_stdev = statistics.stdev(similarity_scores)
        # Normalize: stdev 0 → agreement 1.0, stdev 0.3+ → agreement ~0
        source_agreement = _clamp(1.0 - (score_stdev / 0.30))
    else:
        # Single chunk: no agreement signal, give neutral value
        source_agreement = 0.5

    # ─────────────────────────────────────────────
    # Signal 4: Coverage Bonus (weight 0.10)
    # Reward having multiple strong chunks (not just one lucky match).
    # ─────────────────────────────────────────────
    strong_chunks = sum(1 for s in similarity_scores if s >= STRONG_MATCH_THRESHOLD)
    coverage_ratio = min(strong_chunks / MIN_CHUNKS_FOR_COVERAGE, 1.0)
    coverage_bonus = _clamp(coverage_ratio)

    # ─────────────────────────────────────────────
    # Final weighted combination
    # ─────────────────────────────────────────────
    final_score = (
        0.45 * retrieval_quality
        + 0.25 * source_authority
        + 0.20 * source_agreement
        + 0.10 * coverage_bonus
    )
    final_score = round(_clamp(final_score), 3)

    # ─────────────────────────────────────────────
    # Human review decision
    # ─────────────────────────────────────────────
    review_required, review_reason = _should_review(
        score=final_score,
        top_score=top_score,
        doc_types=doc_types,
        threshold=confidence_threshold,
        strong_chunks=strong_chunks,
        query=query,
    )

    signals = {
        "n_chunks": n,
        "top_similarity": round(top_score, 4),
        "weighted_mean_similarity": round(weighted_mean, 4),
        "score_stdev": round(statistics.stdev(similarity_scores) if n >= 2 else 0.0, 4),
        "strong_chunks": strong_chunks,
        "unique_doc_types": list(set(doc_types)),
    }

    return ConfidenceResult(
        score=final_score,
        human_review_required=review_required,
        human_review_reason=review_reason,
        retrieval_quality=round(retrieval_quality, 3),
        source_authority=round(source_authority, 3),
        source_agreement=round(source_agreement, 3),
        coverage_bonus=round(coverage_bonus, 3),
        signals=signals,
    )


def _should_review(
    score: float,
    top_score: float,
    doc_types: list[str],
    threshold: float,
    strong_chunks: int,
    query: str,
) -> tuple[bool, Optional[str]]:
    """
    Decide if human review is required and provide a reason.
    Multiple independent triggers — any one is sufficient.
    """
    # Rule 1: Overall confidence too low
    if score < threshold:
        return True, f"Confidence score {score:.2f} is below threshold {threshold:.2f}"

    # Rule 2: Best chunk is weak — even if average looks ok
    if top_score < WEAK_MATCH_THRESHOLD:
        return True, f"Best retrieved chunk has low similarity ({top_score:.2f}) — query may be out of scope"

    # Rule 3: Case law involved — inherently subjective
    if "Case Law" in doc_types:
        return True, "Response based on case law — requires legal interpretation"

    # Rule 4: No strong chunks at all (all moderate, none confident)
    if strong_chunks == 0:
        return True, "No strongly matching chunks found — answer may be incomplete"

    # Rule 5: Very short query — likely ambiguous
    if query and len(query.split()) < 4:
        return True, "Query is very short — may be ambiguous or under-specified"

    return False, None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))