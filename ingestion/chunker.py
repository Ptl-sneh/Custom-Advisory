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

MIN_CHUNK_LENGTH = 60  # skip chunks shorter than this — they're usually noise


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    token_estimate: int          # NEW: rough token count for confidence scoring
    page_number: Optional[int] = None
    doc_type: str
    source_name: str
    issuing_authority: Optional[str] = None
    issue_date: Optional[str] = None
    reference_number: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


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
    text = re.sub(r"\[SHEET:[^\]]+\]", "", text)   # also strip sheet tags from XLSX
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)          # collapse multiple spaces/tabs
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English legal text."""
    return max(1, len(text) // 4)


def chunk_document(parsed_doc: ParsedDocument, doc_id: str) -> list[DocumentChunk]:
    logger.info(f"Chunking started | doc_id={doc_id} | file={parsed_doc.filename}")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,       # see config.py — now 800 recommended
            chunk_overlap=CHUNK_OVERLAP, # now 120 recommended
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

        raw_chunks = splitter.split_text(parsed_doc.raw_text)

        if not raw_chunks:
            raise ValueError(f"No chunks generated for: {parsed_doc.filename}")

        chunks: list[DocumentChunk] = []
        skipped = 0

        for idx, chunk_text in enumerate(raw_chunks):
            page_number = extract_page_number(chunk_text)
            clean_text = extract_clean_text(chunk_text)

            # FIX: skip very short/noise chunks
            if len(clean_text) < MIN_CHUNK_LENGTH:
                skipped += 1
                logger.debug(f"Skipped short chunk | idx={idx} | len={len(clean_text)}")
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=parsed_doc.filename,
                    chunk_index=idx,
                    text=clean_text,
                    token_estimate=estimate_tokens(clean_text),  # NEW
                    page_number=page_number,
                    doc_type=parsed_doc.doc_type.value,
                    source_name=parsed_doc.source_name,
                    issuing_authority=parsed_doc.issuing_authority,
                    issue_date=parsed_doc.issue_date,
                    reference_number=parsed_doc.reference_number,
                    tags=parsed_doc.tags,
                )
            )

        logger.info(
            f"Chunking complete | doc_id={doc_id} | chunks={len(chunks)} | skipped={skipped}"
        )
        return chunks

    except Exception as e:
        logger.error(f"Chunking failed | doc_id={doc_id} | error={str(e)}")  # FIX: was logger.info
        raise