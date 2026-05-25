"""
embedder.py  —  Optimized Embedding + Vector Store Pipeline
=============================================================
Key changes vs original:
  1. EmbeddingManager singleton: model is loaded ONCE, not on every ingest call
  2. Retry logic: failed batches are retried up to MAX_BATCH_RETRIES times
  3. Embedding normalization: unit-normalize vectors before storage for
     reliable cosine similarity scores in [0, 1] range
  4. Duplicate detection: skip doc_id if already ingested (idempotent)
  5. Logger levels fixed: errors → logger.error
  6. ChromaDB query_with_scores helper added (used by retriever)
"""

import json
import math
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from config import VECTOR_STORE_DIR, PROCESSED_DIR, EMBEDDING_MODEL
from ingestion.parser import ParsedDocument, parse_document
from ingestion.chunker import DocumentChunk, chunk_document
from schemas import IndexingStatus
from logger import get_logger
from rag.bm25_manager import BM25Manager

logger = get_logger(__name__)

BATCH_SIZE = 32
MAX_BATCH_RETRIES = 2
RETRY_DELAY_SECONDS = 2


# Singleton: model loads once, reused per call


class EmbeddingManager:
    """
    Singleton wrapper so OllamaEmbeddings is initialized only once.
    Original code called get_embeddings() inside ingest_document() which
    re-initialized the model on every single document ingestion.
    """

    instance: Optional["EmbeddingManager"] = None
    model: Optional[OllamaEmbeddings] = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def get_model(self) -> OllamaEmbeddings:
        if self.model is None:
            try:
                self.model = OllamaEmbeddings(model=EMBEDDING_MODEL)
                logger.debug(f"Embedding model initialized | model={EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Embedding model init failed | error={str(e)}")
                raise
        return self.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed and L2-normalize so cosine similarity = dot product."""
        vectors = self.get_model().embed_documents(texts)
        return [l2_normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query and L2-normalize."""
        vector = self.get_model().embed_query(text)
        return l2_normalize(vector)


embedding_manager = EmbeddingManager()


def l2_normalize(vector: list[float]) -> list[float]:
    """
    Unit-normalize a vector.
    Why: ChromaDB cosine space computes 1 - cosine_similarity as the distance.
    If vectors are already unit-normalized, dot product == cosine similarity,
    and distances reliably map to [0, 1]. Without normalization, raw magnitudes
    can distort similarity scores — leading to unreliable confidence scores.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


# ChromaDB helpers


def get_chroma_client() -> chromadb.ClientAPI:
    try:
        client = chromadb.PersistentClient(
            path=VECTOR_STORE_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.debug(f"ChromaDB connected | path={VECTOR_STORE_DIR}")
        return client
    except Exception as e:
        logger.error(f"ChromaDB connection failed | error={str(e)}")
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
        logger.error(f"Collection init failed | error={str(e)}")
        raise


def query_collection(query_text: str, top_k: int = 6, metadata_filter=None) -> dict:
    """
    Retrieve top-k chunks for a query.
    Returns raw ChromaDB result dict with distances converted to similarity scores.

    distance → similarity: since space=cosine, ChromaDB returns
    distance = 1 - cosine_sim, so similarity = 1 - distance.
    """
    client = get_chroma_client()
    collection = get_collection(client)

    query_vector = embedding_manager.embed_query(query_text)

    where_filter = {"doc_type": metadata_filter} if metadata_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Convert distances → similarity scores in place
    if results.get("distances"):
        results["similarities"] = [
            [round(1.0 - d, 4) for d in dist_list] for dist_list in results["distances"]
        ]

    return results


# Core ingestion


def is_already_ingested(document_hash: str):
    records = get_all_document_records()
    for record in records:
        if record.get("document_hash") == document_hash:
            return True
    return False


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

    # Idempotency check
    if is_already_ingested(parsed_doc.document_hash):
        logger.info("Duplicate file skipped")
        result["status"] = IndexingStatus.COMPLETED
        result["error"] = "Duplicate document"
        return result

    try:
        # Step 1: Parse
        logger.info(f"[1/3] Parsing | doc_id={doc_id}")
        parsed_doc: ParsedDocument = parse_document(file_path, metadata)

        # Step 2: Chunk
        logger.info(f"[2/3] Chunking | doc_id={doc_id}")
        chunks: list[DocumentChunk] = chunk_document(parsed_doc, doc_id)

        # Step 3: Embed + store
        logger.info(f"[3/3] Embedding | doc_id={doc_id} | chunks={len(chunks)}")
        client = get_chroma_client()
        collection = get_collection(client)

        total_stored = 0
        total_failed = 0
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
            batch = chunks[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            stored = store_batch_with_retry(
                collection, batch, batch_num, total_batches, doc_id
            )
            if stored:
                total_stored += len(batch)
            else:
                total_failed += len(batch)

        if total_failed > 0:
            logger.warning(
                f"Partial ingestion | doc_id={doc_id} | stored={total_stored} | failed={total_failed}"
            )

        save_processed_record(doc_id, parsed_doc, total_stored)
        result["status"] = IndexingStatus.COMPLETED
        result["chunk_count"] = total_stored
        logger.info(
            f"Ingestion complete | doc_id={doc_id} | chunks_stored={total_stored}"
        )

    except Exception as e:
        result["status"] = IndexingStatus.FAILED
        result["error"] = str(e)
        logger.error(f"Ingestion failed | doc_id={doc_id} | error={str(e)}")

    return result


def store_batch_with_retry(
    collection: chromadb.Collection,
    batch: list[DocumentChunk],
    batch_num: int,
    total_batches: int,
    doc_id: str,
) -> bool:
    """
    Embed and store one batch. Retries up to MAX_BATCH_RETRIES times on failure.
    Returns True if successful, False if all retries exhausted.
    """
    for attempt in range(1, MAX_BATCH_RETRIES + 2):  # +2: initial + retries
        try:
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "token_estimate": c.token_estimate,
                    "page_number": (c.page_number if c.page_number is not None else -1),
                    "doc_type": c.doc_type,
                    "source_name": c.source_name,
                    "issuing_authority": (c.issuing_authority or ""),
                    "issue_date": (c.issue_date or ""),
                    "reference_number": (c.reference_number or ""),
                    "tags": json.dumps(c.tags),
                    "section": (c.section or ""),
                    "chapter": (c.chapter or ""),
                    "notification": (c.notification or ""),
                    "hs_code": (c.hs_code or ""),
                }
                for c in batch
            ]
            # Normalize vectors before storage
            vectors = embedding_manager.embed_texts(texts)

            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

            bm25 = BM25Manager()
            bm25.chunk_store = texts
            bm25.metadata_store = metadatas
            bm25.chunk_ids = ids
            bm25.build_index(texts)
            bm25.save_index()

            logger.debug(
                f"Batch {batch_num}/{total_batches} stored | attempt={attempt}"
            )
            return True

        except Exception as e:
            logger.warning(
                f"Batch {batch_num}/{total_batches} attempt {attempt} failed | "
                f"doc_id={doc_id} | error={str(e)}"
            )
            if attempt <= MAX_BATCH_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(
                    f"Batch {batch_num}/{total_batches} permanently failed after "
                    f"{MAX_BATCH_RETRIES} retries | doc_id={doc_id}"
                )
                return False
    return False


# Document management


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
        delete_processed_record(doc_id)

        logger.info(
            f"Document deleted | doc_id={doc_id} | chunks_removed={len(results['ids'])}"
        )
        return True

    except Exception as e:
        logger.error(f"Delete failed | doc_id={doc_id} | error={str(e)}")
        return False


def save_processed_record(
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
            "document_hash": parsed_doc.document_hash,
        }
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        logger.debug(f"Record saved | doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Record save failed | doc_id={doc_id} | error={str(e)}")


def delete_processed_record(doc_id: str) -> None:
    try:
        record_path = Path(PROCESSED_DIR) / f"{doc_id}.json"
        if record_path.exists():
            record_path.unlink()
            logger.debug(f"Record deleted | doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Record delete failed | doc_id={doc_id} | error={str(e)}")


def get_all_document_records() -> list[dict]:
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        records = []
        for record_file in Path(PROCESSED_DIR).glob("*.json"):
            with open(record_file) as f:
                records.append(json.load(f))
        return sorted(records, key=lambda x: x.get("ingested_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"Records load failed | error={str(e)}")
        return []
