import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from config import VECTOR_STORE_DIR, PROCESSED_DIR, EMBEDDING_MODEL
from ingestion.parser import ParsedDocument, parse_document
from ingestion.chunker import DocumentChunk, chunk_document
from schemas import IndexingStatus
from logger import get_logger

logger = get_logger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    try:
        client = chromadb.PersistentClient(
            path=VECTOR_STORE_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.debug(f"ChromaDB connected | path={VECTOR_STORE_DIR}")
        return client
    except Exception as e:
        logger.info(f"ChromaDB connection failed | error={str(e)}")
        raise


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    try:
        collection = client.get_or_create_collection(
            name="customs_advisory",
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(f"Collection ready | count={collection.count()}")
        return collection
    except Exception as e:
        logger.info(f"Collection init failed | error={str(e)}")
        raise


def get_embeddings() -> OllamaEmbeddings:
    try:
        model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        logger.debug(f"Embedding model loaded | model={EMBEDDING_MODEL}")
        return model
    except Exception as e:
        logger.info(
            f"Embedding model load failed | model={EMBEDDING_MODEL} | error={str(e)}"
        )
        raise


def ingest_document(
    file_path: str,
    metadata: dict,
    doc_id: Optional[str] = None,
) -> dict:
    doc_id = doc_id or str(uuid.uuid4())

    logger.info(f"Ingestion started | doc_id={doc_id} | file={Path(file_path).name}")

    result = {
        "doc_id": doc_id,
        "filename": Path(file_path).name,
        "status": IndexingStatus.PROCESSING,
        "chunk_count": 0,
        "error": None,
    }

    try:
        # Step 1: Parse
        logger.info(f"[1/3] Parsing | doc_id={doc_id}")
        parsed_doc: ParsedDocument = parse_document(file_path, metadata)

        # Step 2: Chunk
        logger.info(f"[2/3] Chunking | doc_id={doc_id}")
        chunks: list[DocumentChunk] = chunk_document(parsed_doc, doc_id)

        # Step 3: Embed + store
        logger.info(f"[3/3] Embedding | doc_id={doc_id} | chunks={len(chunks)}")
        embeddings_model = get_embeddings()
        client = get_chroma_client()
        collection = get_collection(client)

        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                texts = [c.text for c in batch]
                ids = [c.chunk_id for c in batch]
                metadatas = [
                    {
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number or -1,
                        "doc_type": c.doc_type,
                        "source_name": c.source_name,
                        "issuing_authority": c.issuing_authority or "",
                        "issue_date": c.issue_date or "",
                        "reference_number": c.reference_number or "",
                        "tags": json.dumps(c.tags),
                    }
                    for c in batch
                ]

                vectors = embeddings_model.embed_documents(texts)
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                logger.debug(f"Batch {batch_num}/{total_batches} stored")

            except Exception as e:
                logger.info(
                    f"Batch {batch_num}/{total_batches} failed | doc_id={doc_id} | error={str(e)}"
                )
                continue

        _save_processed_record(doc_id, parsed_doc, len(chunks))

        result["status"] = IndexingStatus.COMPLETED
        result["chunk_count"] = len(chunks)
        logger.info(f"Ingestion complete | doc_id={doc_id} | chunks={len(chunks)}")

    except Exception as e:
        result["status"] = IndexingStatus.FAILED
        result["error"] = str(e)
        logger.info(f"Ingestion failed | doc_id={doc_id} | error={str(e)}")

    return result


def delete_document(doc_id: str) -> bool:
    logger.info(f"Deleting document | doc_id={doc_id}")
    try:
        client = get_chroma_client()
        collection = get_collection(client)

        results = collection.get(where={"doc_id": doc_id})
        if not results["ids"]:
            logger.info(f"No chunks found | doc_id={doc_id}")
            return False

        collection.delete(ids=results["ids"])
        _delete_processed_record(doc_id)

        logger.info(
            f"Document deleted | doc_id={doc_id} | chunks_removed={len(results['ids'])}"
        )
        return True

    except Exception as e:
        logger.info(f"Delete failed | doc_id={doc_id} | error={str(e)}")
        return False


def _save_processed_record(
    doc_id: str, parsed_doc: ParsedDocument, chunk_count: int
) -> None:
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        record_path = Path(PROCESSED_DIR) / f"{doc_id}.json"

        record = {
            "doc_id": doc_id,
            "filename": parsed_doc.filename,
            "source_name": parsed_doc.source_name,
            "doc_type": parsed_doc.doc_type.value,
            "issuing_authority": parsed_doc.issuing_authority,
            "issue_date": parsed_doc.issue_date,
            "reference_number": parsed_doc.reference_number,
            "tags": parsed_doc.tags,
            "page_count": parsed_doc.page_count,
            "file_size_kb": parsed_doc.file_size_kb,
            "chunk_count": chunk_count,
            "status": IndexingStatus.COMPLETED.value,
            "ingested_at": datetime.utcnow().isoformat(),
        }

        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

        logger.debug(f"Record saved | doc_id={doc_id}")

    except Exception as e:
        logger.info(f"Record save failed | doc_id={doc_id} | error={str(e)}")


def _delete_processed_record(doc_id: str) -> None:
    try:
        record_path = Path(PROCESSED_DIR) / f"{doc_id}.json"
        if record_path.exists():
            record_path.unlink()
            logger.debug(f"Record deleted | doc_id={doc_id}")
    except Exception as e:
        logger.info(f"Record delete failed | doc_id={doc_id} | error={str(e)}")


def get_all_document_records() -> list[dict]:
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        records = []

        for record_file in Path(PROCESSED_DIR).glob("*.json"):
            with open(record_file) as f:
                records.append(json.load(f))

        logger.debug(f"Records loaded | count={len(records)}")
        return sorted(records, key=lambda x: x.get("ingested_at", ""), reverse=True)

    except Exception as e:
        logger.info(f"Records load failed | error={str(e)}")
        return []

