"""
rag/chain.py  —  Optimized Advisory Chain
==========================================

Changes from original:
  1. Removed duplicate calculate_confidence() — was conflicting with
     confidence_score.py. Confidence now comes purely from retriever.py
     which already uses the 4-signal calculation.

  2. LLM singleton: OllamaLLM no longer re-initialized on every call.

  3. Smarter human_review_required logic:
     Old: triggered on ANY risk flag → almost always True → useless
     New: triggered only on HIGH severity conditions with clear reasons.

  4. Prompt improved: forces JSON-like output which is far more reliably
     parsed than freeform section headers.

  5. Robust parser: tries JSON first, falls back to section parsing,
     then falls back to raw text. Three layers of defense.

  6. LLM response validation: detects empty or garbled responses
     and raises a clear error instead of silently returning bad data.

  7. Out-of-scope gate preserved: still rejects before calling LLM
     if retrieval confidence is too low.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from config import LLM_MODEL, CONFIDENCE_THRESHOLD
from rag.retriever import retrieve, RetrievalResult
from schemas.advisory import AdvisoryQuery, AdvisoryResponse, SourceReference
from schemas.common import ReviewStatus
from logger import get_logger
import time
from rag.answer_confidence import (calculate_answer_confidence)

logger = get_logger(__name__)


# LLM Singleton


class LLMManager:
    """
    Singleton so OllamaLLM is initialized once, not per request.
    temperature=0.1: low randomness for legal/compliance answers.
    We want deterministic, factual responses not creative ones.
    """

    instance = None
    llm = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def get_llm(self) -> OllamaLLM:
        if self.llm is None:
            try:
                self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
                logger.debug(f"LLM initialized | model={LLM_MODEL}")
            except Exception as e:
                logger.error(f"LLM init failed | error={str(e)}")
                raise
        return self.llm


llm_manager = LLMManager()

ADVISORY_PROMPT = PromptTemplate(
    input_variables=["query", "context", "source_list"],
    template="""You are a senior customs and trade compliance expert in India.
Answer ONLY based on the provided source documents. Do NOT use external knowledge.
If the answer is not in the sources, say so explicitly.

USER QUERY:
{query}

SOURCE DOCUMENTS:
{context}

SOURCES:
{source_list}

You MUST respond with valid JSON only. No text before or after the JSON block.
Use exactly this structure:

{{
  "short_answer": "2-3 sentence direct answer to the query",
  "classification": "HSN code or advisory recommendation, or null if not applicable",
  "reasoning": "Detailed reasoning citing source names inline. Be specific about sections, headings, and page numbers.",
  "alternate_views": "Any conflicting interpretations or alternate classifications found in sources. null if none.",
  "risk_flags": ["risk 1", "risk 2"],
  "confidence_note": "Brief note on how well sources cover this query"
}}

Rules:
- risk_flags must be a JSON array, empty [] if no risks
- classification must be null (not the string "N/A") if not applicable
- alternate_views must be null if genuinely none found
- Do not hallucinate. If a fact is not in the sources, say "not specified in provided documents"
""",
)

# Context builder


def build_context(chunks: list[SourceReference]) -> tuple[str, str]:
    """
    Build context string and source list from retrieved chunks.
    Returns (context_text, source_list_text)
    """
    logger.info(f"Building context from {len(chunks)} chunks")
    context_parts = []
    seen_sources = {}

    for i, chunk in enumerate(chunks, start=1):
        ref_part = f" | Ref: {chunk.reference_number}" if chunk.reference_number else ""
        page_part = f" | Page: {chunk.page_number}" if chunk.page_number else ""

        context_parts.append(
            f"[SOURCE {i}] {chunk.source_name}{ref_part}{page_part}\n"
            f"{chunk.chunk_text}"
        )
        seen_sources[chunk.doc_id] = chunk.source_name

    context_text = "\n\n---\n\n".join(context_parts)
    source_list = "\n".join(f"- {name}" for name in seen_sources.values())

    logger.info(f"Context built | unique_sources={len(seen_sources)} | context_chars={len(context_text)}")
    return context_text, source_list


# Response parsers


def parse_json(raw: str) -> Optional[dict]:
    """Layer 1: Try direct JSON parse."""
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        return json.loads(cleaned)
    except Exception:
        return None


def parse_sections(raw: str) -> Optional[dict]:
    """
    Layer 2: Fall back to section-header parsing if JSON fails.
    Handles the old prompt format in case LLM ignores JSON instruction.
    """
    sections = {
        "SHORT_ANSWER": "",
        "CLASSIFICATION": "",
        "REASONING": "",
        "ALTERNATE_VIEWS": "",
        "RISK_FLAGS": "",
        "CONFIDENCE_NOTE": "",
    }
    current = None

    for line in raw.strip().split("\n"):
        stripped = line.strip()
        matched = False
        for key in sections:
            if stripped.startswith(f"{key}:"):
                current = key
                inline = stripped[len(key) + 1 :].strip()
                if inline:
                    sections[key] = inline
                matched = True
                break
        if not matched and current:
            sections[current] += "\n" + line if sections[current] else line

    # Only return if we got something meaningful
    if any(v.strip() for v in sections.values()):
        flags_text = sections.get("RISK_FLAGS", "")
        flags = []
        if flags_text and flags_text.upper() not in ("NONE", ""):
            flags = [
                l.strip().lstrip("-•*").strip()
                for l in flags_text.split("\n")
                if l.strip().lstrip("-•*").strip()
            ]

        return {
            "short_answer": sections["SHORT_ANSWER"].strip(),
            "classification": sections["CLASSIFICATION"].strip() or None,
            "reasoning": sections["REASONING"].strip(),
            "alternate_views": sections["ALTERNATE_VIEWS"].strip() or None,
            "risk_flags": flags,
            "confidence_note": sections["CONFIDENCE_NOTE"].strip(),
        }
    return None


def parse_fallback(raw: str) -> dict:
    """Layer 3: Raw text fallback — at minimum return something."""
    logger.warning("Both JSON and section parsers failed — using raw fallback")
    return {
        "short_answer": raw[:500].strip() if raw else "Unable to parse LLM response.",
        "classification": None,
        "reasoning": raw.strip() if raw else "",
        "alternate_views": None,
        "risk_flags": ["llm_response_parse_failed"],
        "confidence_note": "Response parsing failed — manual review required.",
    }


def parse_llm_response(raw: str) -> dict:
    """
    Parse LLM output using 3-layer defense.
    JSON first → section headers → raw fallback.
    """
    if not raw or not raw.strip():
        logger.error("LLM returned empty response")
        return parse_fallback("")

    result = parse_json(raw)
    if result:
        logger.info("Successfully parsed LLM response via JSON")
        # Normalize fields
        result.setdefault("short_answer", "")
        result.setdefault("classification", None)
        result.setdefault("reasoning", "")
        result.setdefault("alternate_views", None)
        result.setdefault("risk_flags", [])
        result.setdefault("confidence_note", "")
        # Ensure risk_flags is a list
        if isinstance(result["risk_flags"], str):
            result["risk_flags"] = (
                [result["risk_flags"]] if result["risk_flags"] else []
            )
        return result

    result = parse_sections(raw)
    if result:
        logger.info("Successfully parsed LLM response via section headers")
        return result

    return parse_fallback(raw)


# Human review decision

# Risk flags severe enough to always require human review
HIGH_SEVERITY_FLAGS = {
    "penalty",
    "confiscation",
    "fraud",
    "willful",
    "suppression",
    "mis-declaration",
    "mis-classification",
    "sanction",
    "prohibited",
    "banned",
    "seized",
    "detained",
    "prosecution",
    "llm_response_parse_failed",
    "query_out_of_scope",
}


def requires_human_review(
    confidence: float,
    risk_flags: list[str],
    retrieval: RetrievalResult,
    parsed: dict,
) -> tuple[bool, str]:
    """
    Decide if human review is required and return (bool, reason).

    Old logic: ANY risk flag → review required → almost always True.
    New logic: only HIGH severity flags, low confidence, or parse failure.
    """
    # 1. Confidence below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        reason = f"Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
        logger.info(f"Human review required: {reason}")
        return True, reason

    start = time.time()

    # 2. Retrieval itself flagged for review
    if retrieval.human_review_required and retrieval.human_review_reason:
        return True, retrieval.human_review_reason

    retrieval_ms = (time.time() - start) * 1000
    logger.debug(f"Review check retrieval signal evaluated in {retrieval_ms:.2f} ms")

    # 3. High severity risk flags
    flags_lower = {f.lower() for f in risk_flags}
    triggered = HIGH_SEVERITY_FLAGS & flags_lower
    if triggered:
        reason = f"High severity risk flags: {', '.join(triggered)}"
        logger.info(f"Human review required: {reason}")
        return True, reason

    # 4. LLM said it couldn't find the answer
    short_answer = parsed.get("short_answer", "").lower()
    not_found_phrases = [
        "not specified",
        "not mentioned",
        "not found",
        "not provided",
        "not contained",
        "no information",
        "cannot find",
    ]
    if any(p in short_answer for p in not_found_phrases):
        reason = "LLM indicated answer not found in provided documents"
        logger.info(f"Human review required: {reason}")
        return True, reason

    # 5. Parse failed
    if "llm_response_parse_failed" in flags_lower:
        reason = "LLM response could not be parsed"
        logger.info(f"Human review required: {reason}")
        return True, reason

    logger.info("No human review required based on chain logic.")
    return False, None


# Main advisory generation

def preprocess_query(query: str):
    logger.info(f"Preprocessing query | original_len={len(query)}")
    query = query.strip()
    query = " ".join(query.split())
    logger.info(f"Preprocessed query | final_len={len(query)}")
    return query


def generate_advisory(query_obj: AdvisoryQuery) -> AdvisoryResponse:
    session_id = str(uuid.uuid4())
    logger.info(f"Advisory started | session_id={session_id}")

    try:
        # Step 1: Retrieve
        logger.info(f"[1/3] Retrieving | session_id={session_id}")

        normalized_query = preprocess_query(query_obj.query)

        BAD_PATTERNS = [
            "ignore previous",
            "system prompt",
            "forget instructions",
            "override",
        ]

        query_lower = query_obj.query.lower()
        for pattern in BAD_PATTERNS:
            if pattern in query_lower:
                logger.info(f"Unsafe query detected matching pattern: '{pattern}'")
                raise ValueError("Unsafe query detected")

        retrieval: RetrievalResult = retrieve(
            query=normalized_query,
            top_k=query_obj.top_k,
        )

        logger.info(f"Retrieval complete | chunks_returned={len(retrieval.chunks) if retrieval.chunks else 0} | is_confident={retrieval.is_confident}")

        if not retrieval.chunks:
            raise ValueError("No relevant documents found for this query.")

        # Gate: reject if retrieval not confident
        if not retrieval.is_confident:
            logger.warning(
                f"Out-of-scope query | session_id={session_id} | "
                f"top_sim={retrieval.top_similarity} | "
                f"reason={retrieval.human_review_reason}"
            )
            return AdvisoryResponse(
                session_id=session_id,
                query=query_obj.query,
                short_answer=(
                    "The uploaded documents do not contain sufficient information "
                    "to answer this query reliably."
                ),
                classification=None,
                reasoning=(
                    f"Retrieval confidence too low — best similarity was "
                    f"{retrieval.top_similarity:.4f}. "
                    f"Reason: {retrieval.human_review_reason}"
                ),
                alternate_views=None,
                risk_flags=["query_out_of_scope"],
                source_references=[],
                confidence_score=0.0,
                human_review_required=True,
                review_status=ReviewStatus.PENDING,
                created_at=datetime.utcnow(),
            )

        # Step 2: LLM call

        llm_start = time.time()

        logger.info(f"[2/3] LLM call | session_id={session_id} | model={LLM_MODEL}")
        context_text, source_list = build_context(retrieval.chunks)
        prompt = ADVISORY_PROMPT.format(
            query=query_obj.query,
            context=context_text,
            source_list=source_list,
        )

        llm = llm_manager.get_llm()
        raw_response = llm.invoke(prompt)

        if not raw_response or not raw_response.strip():
            raise ValueError("LLM returned an empty response")

        llm_ms = (time.time() - llm_start) * 1000
        
        logger.info(f"LLM responded | chars={len(raw_response)} | time_ms={llm_ms:.2f}")

        print(
            f"LLM call took {llm_ms:.2f} ms | session_id={session_id} | model={LLM_MODEL}"
        )

        # Step 3: Parse + assemble response
        logger.info(f"[3/3] Parsing | session_id={session_id}")

        parsed = parse_llm_response(raw_response)

        retrieval_confidence = retrieval.confidence_score

        # Final answer text from LLM
        final_answer = parsed.get(
            "short_answer",
            ""
        )

        # NEW: answer confidence
        answer_confidence = (
            calculate_answer_confidence(
                answer=final_answer,
                retrieved_chunks=retrieval.chunks,
                retrieval_confidence=retrieval_confidence
            )
        )

        risk_flags = parsed.get("risk_flags", [])

        if not isinstance(risk_flags, list):
            risk_flags = (
                [risk_flags]
                if risk_flags
                else []
            )

        # Use answer confidence downstream
        confidence = answer_confidence

        review_required, review_reason = requires_human_review(
            confidence=confidence,
            risk_flags=risk_flags,
            retrieval=retrieval,
            parsed=parsed,
        )

        logger.info(
            f"Advisory complete | session_id={session_id} | "
            f"confidence={confidence:.3f} | risk_flags={len(risk_flags)} | "
            f"review={review_required}"
        )

        return AdvisoryResponse(
            session_id=session_id,
            query=query_obj.query,
            short_answer=parsed.get("short_answer", ""),
            classification=parsed.get("classification") or None,
            reasoning=parsed.get("reasoning", ""),
            alternate_views=parsed.get("alternate_views") or None,
            risk_flags=risk_flags,
            source_references=retrieval.chunks,
            confidence_score=confidence,
            human_review_required=review_required,
            review_status=ReviewStatus.PENDING,
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Advisory failed | session_id={session_id} | error={str(e)}")
        raise
