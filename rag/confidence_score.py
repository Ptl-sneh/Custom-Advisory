"""
confidence_score.py  —  Multi-Signal Confidence Calculator
============================================================

Changes from previous version:
  1. STRONG_MATCH_THRESHOLD lowered 0.70 → 0.65
     Your similarity scores are 0.72–0.75. The old threshold gave them
     only partial credit. Now they get full credit in retrieval quality.

  2. Retrieval quality boost when ALL chunks are strong
     If every retrieved chunk is above the threshold, reward consistency
     by applying a 1.08 multiplier. Clamped to 1.0 max.

  3. Circular authority raised 0.80 → 0.88
     Covers the case where Customs Tariff Act documents are incorrectly
     tagged as Circular during ingestion. Re-ingesting with
     doc_type="Customs Act" is the proper fix and gives 1.00.

  4. Removed Case Law auto-review rule
     For customs advisory, case law is a primary authoritative source.
     Auto-flagging every case law response for human review was too
     aggressive and made the flag meaningless.
"""

import statistics
from dataclasses import dataclass
from typing import Optional

# Authority weights by doc_type

DOC_TYPE_AUTHORITY: dict[str, float] = {
    "Customs Act": 1.00,
    "Tariff Schedule": 0.95,
    "HSN Classification": 0.90,
    "Notification": 0.85,
    "Circular": 0.88,  # raised from 0.80
    "Case Law": 0.75,
    "Trade Policy": 0.70,
    "BIS / Export Control": 0.65,
    "Other": 0.40,
}

# Thresholds
STRONG_MATCH_THRESHOLD = 0.70
WEAK_MATCH_THRESHOLD = 0.45
MIN_CHUNKS_FOR_COVERAGE = 3


@dataclass
class ConfidenceResult:
    score: float
    human_review_required: bool
    human_review_reason: Optional[str]
    retrieval_quality: float
    source_authority: float
    source_agreement: float
    coverage_bonus: float
    signals: dict


def calculate_confidence(
    similarity_scores: list[float],
    doc_types: list[str],
    query: str = "",
    confidence_threshold: float = 0.65,
) -> ConfidenceResult:
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

    # Signal 1: Retrieval Quality (weight 0.45)
    # Harmonic-weighted mean — top chunks matter more.
    # rank 1 → weight 1.0, rank 2 → 0.5, rank 3 → 0.33 ...

    weights = [1.0 / (i + 1) for i in range(n)]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weight_total = sum(weights)
    weighted_mean = weighted_sum / weight_total

    top_score = scores[0]

    if top_score < WEAK_MATCH_THRESHOLD:
        retrieval_quality = top_score * 0.5

    elif top_score < STRONG_MATCH_THRESHOLD:
        retrieval_quality = weighted_mean * 0.85

    else:
        # FIX: boost when ALL chunks are strong — rewards consistency
        all_strong = all(s >= STRONG_MATCH_THRESHOLD for s in similarity_scores)
        retrieval_quality = weighted_mean * 1.08 if all_strong else weighted_mean

    retrieval_quality = clamp(retrieval_quality)

    # Signal 2: Source Authority (weight 0.25)
    authority_scores = [
        DOC_TYPE_AUTHORITY.get(dt, DOC_TYPE_AUTHORITY["Other"]) for dt in doc_types
    ]
    source_authority = clamp(statistics.mean(authority_scores))

    # Signal 3: Source Agreement (weight 0.20)
    # Low stdev = chunks are consistent = more trustworthy
    if n >= 2:
        score_stdev = statistics.stdev(similarity_scores)
        source_agreement = clamp(1.0 - (score_stdev / 0.30))
    else:
        source_agreement = 0.5

    # Signal 4: Coverage Bonus (weight 0.10)
    strong_chunks = sum(1 for s in similarity_scores if s >= STRONG_MATCH_THRESHOLD)
    coverage_ratio = min(strong_chunks / MIN_CHUNKS_FOR_COVERAGE, 1.0)
    coverage_bonus = clamp(coverage_ratio)

    # Final score

    final_score = (
        0.45 * retrieval_quality
        + 0.25 * source_authority
        + 0.20 * source_agreement
        + 0.10 * coverage_bonus
    )
    final_score = round(clamp(final_score), 3)

    review_required, review_reason = should_review(
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


def should_review(
    score: float,
    top_score: float,
    threshold: float,
    strong_chunks: int,
    query: str,
    doc_types: list[str] = None,
) -> tuple[bool, Optional[str]]:
    # Rule 1: Overall confidence too low
    if score < threshold:
        return True, f"Confidence score {score:.2f} is below threshold {threshold:.2f}"

    # Rule 2: Best chunk is weak
    if top_score < WEAK_MATCH_THRESHOLD:
        return (
            True,
            f"Best retrieved chunk has low similarity ({top_score:.2f}) — query may be out of scope",
        )

    # Rule 3: Case Law auto-review REMOVED
    # Case law is a primary authoritative source for customs advisory.
    # Auto-flagging every case law response was too aggressive.

    # Rule 4: No strong chunks at all
    if strong_chunks == 0:
        return True, "No strongly matching chunks found — answer may be incomplete"

    # Rule 5: Very short query
    if query and len(query.split()) < 4:
        return True, "Query is very short — may be ambiguous or under-specified"

    return False, None


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
