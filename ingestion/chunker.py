import re
import uuid
from typing import Optional
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.parser import ParsedDocument
from logger import get_logger

logger = get_logger(__name__)


class DocumentChunk(BaseModel):
    chunk_id: str  # unique ID for this specific chunk
    doc_id: str  # parent document ID (same for all chunks of one file)
    filename: str
    chunk_index: int  # position of this chunk within the document
    text: str  # the actual clean text sent to the embedder
    page_number: Optional[int] = None
    doc_type: str
    source_name: str
    issuing_authority: Optional[str] = None
    issue_date: Optional[str] = None
    reference_number: Optional[str] = None
    tags: list[str] = []


def extract_page_number(text: str) -> Optional[int]:
    match = re.search(r"\[PAGE (\d+)\]", text)
    return int(match.group(1)) if match else None


def extract_clean_text(text: str) -> str:
    text = re.sub(r"\[PAGE \d+\]", "", text)  # remove the tag itself
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines
    return text.strip()


def chunk_document(parsed_doc: ParsedDocument, doc_id: str) -> list[DocumentChunk]:
    logger.info(f"Chunking started | doc_id={doc_id} | file={parsed_doc.filename}")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        raw_chunks = splitter.split_text(parsed_doc.raw_text)

        if not raw_chunks:
            raise ValueError(f"No chunks generated for: {parsed_doc.filename}")

        chunks: list[DocumentChunk] = []
        skipped = 0

        for idx, chunk_text in enumerate(raw_chunks):
            page_number = extract_page_number(chunk_text)  # then clean the text
            clean_text = extract_clean_text(chunk_text)  # then clean the text

            if not clean_text:
                skipped += 1
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),  # fresh UUID per chunk
                    doc_id=doc_id,  # same for all chunks of this doc
                    chunk_index=idx,  # preserves original order
                    text=clean_text,
                    page_number=page_number,
                    doc_type=parsed_doc.doc_type.value,  # .value converts enum to string
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
        logger.info(f"Chunking failed | doc_id={doc_id} | error={str(e)}")
        raise
