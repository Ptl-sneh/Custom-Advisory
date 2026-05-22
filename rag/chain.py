import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from config import LLM_MODEL, CONFIDENCE_THRESHOLD
from rag.retriever import retrieve, RetrievalResult
from schemas.advisory import AdvisoryQuery, AdvisoryResponse, SourceReference
from schemas.comon import ReviewStatus
from logger import get_logger

logger = get_logger(__name__)


# Prompt template
ADVISORY_PROMPT = PromptTemplate(
    input_variables=["query", "context", "source_list"],
    template="""
You are a senior customs and trade compliance expert in India.
Your job is to provide structured advisory responses based ONLY on the provided source documents.
Do NOT use any external knowledge. If the answer is not found in the sources, say so clearly.

USER QUERY:
{query}

SOURCE DOCUMENTS:
{context}

SOURCES AVAILABLE:
{source_list}

Respond in the following exact format. Do not add any text outside this format:

SHORT_ANSWER:
<A 2-3 sentence direct answer to the query>

CLASSIFICATION:
<HSN code, tariff heading, or advisory recommendation if applicable. Write N/A if not applicable>

REASONING:
<Detailed reasoning based strictly on the source documents. Cite source names inline.>

ALTERNATE_VIEWS:
<Any conflicting interpretations, alternate classifications, or differing positions found in the sources. Write NONE if not found.>

RISK_FLAGS:
<Bullet list of risk areas, compliance gaps, or penalty triggers relevant to this query. Write NONE if no risks identified.>

CONFIDENCE_NOTE:
<Brief note on how well the sources cover this query. Mention if sources are limited or conflicting.>
""",
)


def _build_context(chunks: list[SourceReference]) -> tuple[str, str]:
    """
    Build context string and source list string from retrieved chunks.
    Returns (context_text, source_list_text)
    """
    context_parts = []
    source_set = {}

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[SOURCE {i}] {chunk.source_name}"
            f"{f' | Ref: {chunk.reference_number}' if chunk.reference_number else ''}"
            f"{f' | Page: {chunk.page_number}' if chunk.page_number else ''}\n"
            f"{chunk.chunk_text}"
        )
        source_set[chunk.doc_id] = chunk.source_name

    context_text = "\n\n---\n\n".join(context_parts)
    source_list = "\n".join(f"- {name}" for name in source_set.values())

    return context_text, source_list


def _parse_llm_response(raw_response: str) -> dict:
    """
    Parse the structured LLM response into a dict.
    Extracts each section by its label.
    """
    sections = {
        "SHORT_ANSWER": "",
        "CLASSIFICATION": "",
        "REASONING": "",
        "ALTERNATE_VIEWS": "",
        "RISK_FLAGS": "",
        "CONFIDENCE_NOTE": "",
    }

    current_section = None
    lines = raw_response.strip().split("\n")

    for line in lines:
        stripped = line.strip()

        # Check if this line is a section header
        matched = False
        for key in sections:
            if stripped.startswith(f"{key}:"):
                current_section = key
                # Capture inline content after the colon if any
                inline = stripped[len(key) + 1 :].strip()
                if inline:
                    sections[key] = inline
                matched = True
                break

        if not matched and current_section:
            sections[current_section] += (
                "\n" + line if sections[current_section] else line
            )

    # Clean up each section
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def _extract_risk_flags(risk_text: str) -> list[str]:
    """Convert risk flags text block into a clean list."""
    if not risk_text or risk_text.upper() == "NONE":
        return []

    flags = []
    for line in risk_text.split("\n"):
        cleaned = line.strip().lstrip("-•*").strip()
        if cleaned:
            flags.append(cleaned)

    return flags


def _calculate_confidence(
    retrieval: RetrievalResult,
    parsed: dict,
) -> float:
    """
    Final confidence score combining:
    - Retrieval similarity (70% weight)
    - Response completeness (30% weight)
    """
    retrieval_score = retrieval.avg_confidence

    # Completeness: penalize if key sections are empty or N/A
    filled_sections = sum(
        1
        for key in ["SHORT_ANSWER", "REASONING", "CLASSIFICATION"]
        if parsed.get(key, "").strip()
        and parsed.get(key, "").upper() not in ("N/A", "NONE", "")
    )
    completeness_score = filled_sections / 3

    final_score = round((retrieval_score * 0.7) + (completeness_score * 0.3), 4)

    logger.debug(
        f"Confidence breakdown | retrieval={retrieval_score} | "
        f"completeness={completeness_score} | final={final_score}"
    )

    return final_score


def generate_advisory(query_obj: AdvisoryQuery) -> AdvisoryResponse:
    """
    Main RAG chain entry point.
    Retrieves relevant chunks and generates a structured advisory response.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Advisory generation started | session_id={session_id}")

    try:
        # Step 1: Retrieve relevant chunks
        logger.info(f"[1/3] Retrieving chunks | session_id={session_id}")
        retrieval: RetrievalResult = retrieve(
            query=query_obj.query,
            top_k=query_obj.top_k,
        )

        if not retrieval.chunks:
            logger.info(f"No chunks retrieved | session_id={session_id}")
            raise ValueError("No relevant documents found for this query.")

        # Step 2: Build prompt and call LLM
        logger.info(f"[2/3] Calling LLM | session_id={session_id} | model={LLM_MODEL}")

        context_text, source_list = _build_context(retrieval.chunks)

        prompt = ADVISORY_PROMPT.format(
            query=query_obj.query,
            context=context_text,
            source_list=source_list,
        )

        try:
            llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
            raw_response = llm.invoke(prompt)
            logger.debug(f"LLM response received | chars={len(raw_response)}")
        except Exception as e:
            logger.info(f"LLM call failed | session_id={session_id} | error={str(e)}")
            raise

        # Step 3: Parse + structure response
        logger.info(f"[3/3] Parsing response | session_id={session_id}")

        try:
            parsed = _parse_llm_response(raw_response)
            risk_flags = _extract_risk_flags(parsed.get("RISK_FLAGS", ""))
            confidence = _calculate_confidence(retrieval, parsed)
        except Exception as e:
            logger.info(
                f"Response parsing failed | session_id={session_id} | error={str(e)}"
            )
            raise

        human_review_required = (
            confidence < CONFIDENCE_THRESHOLD
            or len(risk_flags) > 0
            or not retrieval.is_confident
        )

        logger.info(
            f"Advisory complete | session_id={session_id} | "
            f"confidence={confidence} | risk_flags={len(risk_flags)} | "
            f"review_required={human_review_required}"
        )

        return AdvisoryResponse(
            session_id=session_id,
            query=query_obj.query,
            short_answer=parsed.get("SHORT_ANSWER", ""),
            classification=parsed.get("CLASSIFICATION") or None,
            reasoning=parsed.get("REASONING", ""),
            alternate_views=parsed.get("ALTERNATE_VIEWS") or None,
            risk_flags=risk_flags,
            source_references=retrieval.chunks,
            confidence_score=confidence,
            human_review_required=human_review_required,
            review_status=ReviewStatus.PENDING,
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.info(
            f"Advisory generation failed | session_id={session_id} | error={str(e)}"
        )
        raise
