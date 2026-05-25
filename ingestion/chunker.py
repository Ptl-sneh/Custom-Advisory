"""
chunker.py  —  Optimized Document Chunker
==========================================
Key changes vs original:
  1. extract_page_number → returns FIRST page tag in chunk (not last)
  2. MIN_CHUNK_LENGTH guard: skip noise chunks under 60 chars
  3. CHUNK_SIZE raised to 800 (500 chars ≈ ~125 tokens; legal docs need more context)
  4. CHUNK_OVERLAP raised to 120 (richer context bridging across boundaries)
  5. Separators ordered to respect legal doc structure better
  6. Chunk token_estimate added to metadata for downstream confidence scoring
  7. Logger levels fixed: parse errors → logger.error not logger.info
"""

import re
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.parser import ParsedDocument
from logger import get_logger

logger = get_logger(__name__)

MIN_CHUNK_LENGTH = 120  # skip chunks shorter than this — they're usually noise


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    token_estimate: int  # NEW: rough token count for confidence scoring
    page_number: Optional[int] = None
    doc_type: str
    source_name: str
    issuing_authority: Optional[str] = None
    issue_date: Optional[str] = None
    reference_number: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    section: Optional[str] = None
    chapter: Optional[str] = None
    notification: Optional[str] = None
    hs_code: Optional[str] = None


def extract_page_number(text: str) -> Optional[int]:
    """
    Returns the FIRST [PAGE N] tag found in a chunk.

    Original bug: re.search returns the first match anyway, but this makes
    intent explicit and handles edge cases where the tag appears mid-chunk.
    """
    match = re.search(r"\[PAGE (\d+)\]", text)
    return int(match.group(1)) if match else None


def extract_clean_text(text: str) -> str:
    text = re.sub(r"\[PAGE \d+\]", "", text)
    text = re.sub(r"\[SHEET:[^\]]+\]", "", text)  # also strip sheet tags from XLSX
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces/tabs
    return text.strip()


def estimate_tokens(text):
    words = len(text.split())
    return int(words * 1.3)


LEGAL_BOUNDARIES = [
    r"\n\d+[A-Z]?\.",
    r"\nSection\s+\d+[A-Z]?",
    r"\nCHAPTER\s+[IVXLC]+",
    r"\nPART\s+[IVXLC]+",
    r"\nRule\s+\d+",
    r"\nExplanation\.",
    r"\nProvided\s+that",
    r"\nProvided\s+further",
    r"\nIllustration",
]


def split_legal_sections(text):
    pattern = "(" + "|".join(LEGAL_BOUNDARIES) + ")"
    pieces = re.split(pattern, text, flags=re.IGNORECASE)
    merged = []
    current = ""
    for piece in pieces:
        if re.match(pattern, piece, flags=re.IGNORECASE):
            if current:
                merged.append(current.strip())
            current = piece
        else:
            current += piece
    if current:
        merged.append(current.strip())
    return merged


def chunk_document(parsed_doc: ParsedDocument, doc_id: str) -> list[DocumentChunk]:
    logger.info(f"Chunking started | doc_id={doc_id} | file={parsed_doc.filename}")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,  # see config.py — now 800 recommended
            chunk_overlap=CHUNK_OVERLAP,  # now 120 recommended
            length_function=len,
            separators=[
                # Legal structure markers (highest priority)
                "\nCHAPTER ",
                "\nSECTION ",
                "\nPART ",
                "\nSCHEDULE ",
                "\nANNEXURE ",
                # Rule / clause level
                "\nRule ",
                "\nRULE ",
                "\nClause ",
                "\nArticle ",
                "\nSection ",
                # Document type markers
                "\nNotification ",
                "\nCircular ",
                "\nOrder ",
                # Generic structure
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        legal_sections = split_legal_sections(parsed_doc.raw_text)
        raw_chunks = []

        for section in legal_sections:
            raw_chunks.extend(splitter.split_text(section))
        if not raw_chunks:
            raise ValueError(f"No chunks generated for: {parsed_doc.filename}")

        chunks: list[DocumentChunk] = []
        skipped = 0

        for idx, chunk_text in enumerate(raw_chunks):
            page_number = extract_page_number(chunk_text)
            clean_text = extract_clean_text(chunk_text)

            if len(clean_text) < MIN_CHUNK_LENGTH:
                skipped += 1
                logger.debug(f"Skipped short chunk | idx={idx} | len={len(clean_text)}")
                continue

            section_match = re.search(
                r"Section\s+(\d+[A-Z]?)", clean_text, re.IGNORECASE
            )
            chapter_match = re.search(r"CHAPTER\s+(\d{1,2})", clean_text, re.IGNORECASE)
            notification_match = re.search(
                r"Notification\s+No\.?\s*([\w/-]+)", clean_text, re.IGNORECASE
            )
            hs_code_match = re.search(
                r"\b\d{2}\.\d{2}(?:\.\d{2}(?:\.\d{2})?)?\b|\b\d{4,8}\b", clean_text
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=parsed_doc.filename,
                    chunk_index=idx,
                    text=clean_text,
                    token_estimate=estimate_tokens(clean_text),
                    page_number=page_number,
                    doc_type=parsed_doc.doc_type.value,
                    source_name=parsed_doc.source_name,
                    issuing_authority=parsed_doc.issuing_authority,
                    issue_date=parsed_doc.issue_date,
                    reference_number=parsed_doc.reference_number,
                    tags=parsed_doc.tags,
                    section=section_match.group(1) if section_match else None,
                    chapter=chapter_match.group(1) if chapter_match else None,
                    notification=(
                        notification_match.group(1) if notification_match else None
                    ),
                    hs_code=hs_code_match.group(0) if hs_code_match else None,
                )
            )

        logger.info(
            f"Chunking complete | doc_id={doc_id} | chunks={len(chunks)} | skipped={skipped}"
        )
        return chunks

    except Exception as e:
        logger.error(
            f"Chunking failed | doc_id={doc_id} | error={str(e)}"
        )  # FIX: was logger.info
        raise
