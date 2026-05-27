# Custom Advisory Codebase Explained

This project is a customs/trade-compliance RAG system. The idea is:

1. Ingest legal/customs documents like PDFs, DOCX, and Excel files.
2. Extract clean text plus useful metadata.
3. Split the text into retrieval-friendly chunks.
4. Create embeddings and store them in ChromaDB.
5. Build a BM25 keyword index for lexical search.
6. Retrieve the best chunks for a user question.
7. Score how trustworthy the retrieval is.
8. Ask an LLM to generate a grounded advisory answer only from retrieved sources.

The current repo already has the core ingestion + retrieval + advisory logic, but the FastAPI app wiring is still mostly scaffolding.

## End-to-End Flow

### 1. Document ingestion

`ingestion/parser.py` reads a file, extracts text, removes repeated headers/footers/noise, and enriches metadata like section, chapter, notification number, and HS code.

### 2. Chunk creation

`ingestion/chunker.py` breaks the parsed document into legal-aware chunks, keeps page tracking, and adds chunk metadata like token estimate and section info.

### 3. Embedding + storage

`ingestion/embedder.py` embeds those chunks using Ollama embeddings, stores them in ChromaDB, writes a processed-record JSON, and rebuilds the BM25 index.

### 4. Retrieval

`rag/retriever.py` embeds the user query, searches ChromaDB, blends dense similarity with BM25 keyword scores, reranks with MMR, and returns structured source references.

### 5. Confidence scoring

`rag/confidence_score.py` estimates whether retrieval is strong enough by combining retrieval quality, source authority, agreement between chunks, and coverage.

### 6. Advisory generation

`rag/chain.py` builds a source-grounded prompt, asks the LLM for JSON output, parses that output, and decides whether human review is needed.

## Actual Value Flow And Calculations

This section explains how the important values are really calculated in code.

### 1. Parsing stage values

In `ingestion/parser.py`:

- `file_size_kb` is calculated with `os.path.getsize(file_path) / 1024`.
- `document_hash` is `sha256(cleaned_text.encode("utf-8"))`.
- `page_count` comes from the parser:
  - PDF: `len(doc)`
  - DOCX/XLSX: currently `None`
- `chapter`, `section`, `notification_number`, `hs_code`, and `customs_section` are extracted with regex from the cleaned text.

So at this point the code creates one normalized `ParsedDocument` with both raw text and derived metadata.

### 2. Chunking stage values

In `ingestion/chunker.py`:

- The raw parsed text is first split into legal sections with regex boundaries like `Section`, `CHAPTER`, `Rule`, `PART`, and `Explanation`.
- Then LangChain's `RecursiveCharacterTextSplitter` applies:
  - `chunk_size = 800`
  - `chunk_overlap = 120`
- `page_number` is tracked from `[PAGE N]` markers.
  - If a chunk has a page tag, that page becomes `last_seen_page`
  - If not, the chunk inherits the most recent page
- `token_estimate` is calculated as:

```python
words = len(text.split())
token_estimate = int(words * 1.3)
```

- Very short chunks are skipped if:

```python
len(clean_text) < 120
```

This means every final chunk has text, page reference, source metadata, and a rough size estimate.

### 3. Embedding stage values

In `ingestion/embedder.py`:

- Each chunk text is sent to `OllamaEmbeddings`.
- The returned vector is L2-normalized:

```python
norm = sqrt(sum(x * x for x in vector))
normalized = [x / norm for x in vector]
```

Why this matters:

- Normalized vectors make cosine similarity stable.
- ChromaDB is configured with cosine space, so retrieval scores become easier to interpret.

### 4. Chroma retrieval values

In `rag/retriever.py`:

- The query is embedded the same way as document chunks.
- Chroma returns `distances`, not direct similarities.
- Because the collection uses cosine space, the code converts:

```python
similarity = 1.0 - distance
```

So if Chroma returns:

- distance `0.10` -> similarity `0.90`
- distance `0.35` -> similarity `0.65`

This `similarities` list becomes the dense retrieval score list.

### 5. BM25 lexical score values

In `rag/bm25_manager.py` and `rag/retriever.py`:

- BM25 tokenizes text by:
  - lowercasing
  - removing punctuation except `/` and `-`
  - splitting on whitespace
- BM25 search returns raw scores for matching chunks.
- In `retrieve(...)`, BM25 scores are normalized by dividing each score by the maximum score in that result set:

```python
normalized_bm25 = score / max_score
```

That puts BM25 scores roughly into a `0 to 1` scale for blending.

### 6. Final per-chunk retrieval score

Still inside `retrieve(...)`:

- `dense_score` comes from Chroma similarity.
- `bm25_score` comes from normalized BM25.
- If BM25 found a match, the final displayed chunk score is:

```python
final_score = dense_score * 0.75 + bm25_score * 0.25
```

- If there is no BM25 score for that chunk:

```python
final_score = dense_score
```

So the system trusts semantic retrieval most, but gives a 25% bonus to chunks that also match lexically.

### 7. MMR reranking logic

If `use_mmr=True`, the retriever does not just take the top dense results directly.

For each candidate chunk:

```python
mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
```

Where:

- `lambda_param = 0.95`
- `relevance` = cosine similarity between query vector and chunk vector
- `redundancy` = maximum similarity to already selected chunks

This means:

- 95% importance is relevance to the query
- 5% penalty is for near-duplicate chunks

So the returned context is slightly more diverse and less repetitive.

### 8. Confidence score calculation

In `rag/confidence_score.py`, `calculate_confidence(...)` combines 4 signals.

#### A. Retrieval quality

The chunk similarity scores are sorted descending.

Weights are harmonic:

```python
weights = [1.0, 0.5, 0.333..., 0.25, ...]
weighted_mean = sum(score * weight) / sum(weights)
```

Then:

- if top score `< 0.45`, retrieval quality is penalized heavily
- if top score is between `0.45` and `0.65`, weighted mean is multiplied by `0.85`
- if top score `>= 0.65`, weighted mean is used
- if all chunks are strong (`>= 0.65`), quality gets a `1.08` boost and is clamped to `1.0`

#### B. Source authority

Each `doc_type` has a fixed weight, for example:

- `Customs Act = 1.00`
- `Tariff Schedule = 0.95`
- `Notification = 0.85`
- `Case Law = 0.75`
- `Other = 0.40`

Then:

```python
source_authority = mean(authority_scores)
```

#### C. Source agreement

If multiple chunks are retrieved, the code checks how spread out their similarity scores are:

```python
score_stdev = statistics.stdev(similarity_scores)
source_agreement = 1.0 - (score_stdev / 0.30)
```

Then it clamps the result to `0..1`.

Meaning:

- low standard deviation -> chunks agree -> better confidence
- high standard deviation -> chunks are uneven -> lower confidence

#### D. Coverage bonus

The code counts strong chunks:

```python
strong_chunks = count(score >= 0.65)
coverage_ratio = min(strong_chunks / 3, 1.0)
```

So:

- 1 strong chunk -> `0.33`
- 2 strong chunks -> `0.67`
- 3 or more strong chunks -> `1.0`

#### E. Final confidence score

The final confidence is:

```python
final_score = (
    0.45 * retrieval_quality
    + 0.25 * source_authority
    + 0.20 * source_agreement
    + 0.10 * coverage_bonus
)
```

Then it is clamped to `0..1` and rounded to 3 decimals.

### 9. Retrieval human-review rules

Still in `rag/confidence_score.py`, retrieval is flagged for review when any of these are true:

- final confidence is below threshold
- top chunk similarity is below `0.45`
- there are zero strong chunks
- the query has fewer than 4 words

So even before the LLM answers, the retriever can already say "this evidence is weak".

### 10. Advisory generation values

In `rag/chain.py`:

- `session_id = str(uuid.uuid4())`
- `normalized_query` is just stripped and whitespace-normalized
- Retrieved chunks are turned into the prompt context by `build_context(...)`
- The LLM is asked to return JSON fields:
  - `short_answer`
  - `classification`
  - `reasoning`
  - `alternate_views`
  - `risk_flags`
  - `confidence_note`

The chain does not calculate confidence again. It reuses:

```python
confidence = retrieval.confidence_score
```

### 11. Final human-review decision

In `requires_human_review(...)` inside `rag/chain.py`, review is required if:

- confidence `< CONFIDENCE_THRESHOLD`
- retrieval already asked for review
- any high-severity risk flag appears
- the LLM says information is not found in the sources
- LLM response parsing failed

So the final decision is a mix of:

- retrieval quality
- legal risk indicators
- whether the answer was actually grounded and parseable

### 12. Final response object values

The returned `AdvisoryResponse` contains:

- `short_answer`, `classification`, `reasoning`, `alternate_views`, `risk_flags`: from parsed LLM JSON
- `source_references`: from retriever output
- `confidence_score`: from retrieval confidence calculation
- `human_review_required`: from chain review logic
- `review_status`: always starts as `ReviewStatus.PENDING`
- `created_at`: `datetime.utcnow()`

So the final response is basically:

```text
retrieved evidence
-> confidence score
-> LLM grounded answer
-> review decision
-> structured advisory response
```

## File-by-File Guide

## Root files

### `main.py`

Currently empty. This is probably intended to become the FastAPI entrypoint later.

### `config.py`

Central configuration file.

- Builds important paths like `data/raw`, `data/processed`, `vector_store`, and `review_store.json`.
- Reads model names from environment variables.
- Stores chunking defaults like `CHUNK_SIZE` and `CHUNK_OVERLAP`.
- Stores retrieval defaults like `TOP_K`.
- Defines the review threshold with `CONFIDENCE_THRESHOLD`.
- Lists the allowed document types in `DOC_TYPES`.

This file is the shared settings layer used by ingestion, retrieval, and chain generation.

### `logger.py`

Creates the project logger.

- Makes a `logs/` folder automatically.
- Writes logs to both console and a rotating daily file.
- Uses `MaxLevelFilter` so only `INFO` and below go to the configured handlers.
- Exposes `get_logger(name)` so every module can create a consistent logger.

This is infrastructure code used across the project.

### `requirements.txt`

Dependency list for the project.

- FastAPI + Uvicorn for API serving.
- LangChain + Ollama for LLM and embeddings.
- ChromaDB for vector storage.
- PyMuPDF / python-docx / openpyxl for file parsing.
- Pydantic and utility packages.

### `test_ingestion.py`

Simple script to ingest every file inside `data/raw`.

- Builds one metadata dictionary.
- Loops through files.
- Calls `ingest_document(...)`.
- Prints the result for each file.

This is more of a manual smoke test than a formal automated test.

### `test_retrieval.py`

Detailed retrieval-evaluation script.

- Contains a fixed set of domain-specific test queries.
- Calls `query_collection(...)`.
- Calculates confidence scores.
- Checks whether expected keywords appear in retrieved chunks.
- Prints chunk previews, similarity scores, and summary metrics.

This file is used to judge whether the retrieval pipeline works well on the current customs documents.

### `test_pipeline.py`

End-to-end manual test script.

- Calls `retrieve(...)` directly.
- Prints retrieval metrics and retrieved chunks.
- Calls `generate_advisory(...)`.
- Prints the final advisory response.

This is the quickest way to test the whole RAG flow manually from one script.

## `schemas/`

This folder contains Pydantic models and enums used to keep data structured.

### `schemas/common.py`

Defines common enums and a base response model.

- `DocType`: allowed document categories like Circular, Notification, Customs Act, Case Law, etc.
- `IndexingStatus`: pending/processing/completed/failed.
- `ReviewStatus`: pending/approved/rejected/needs_edit.
- `BaseResponse`: generic `{success, message}` model.

### `schemas/advisory.py`

Defines advisory-related request/response models.

- `SourceReference`: one retrieved chunk plus its source metadata.
- `AdvisoryQuery`: user query input with `top_k`.
- `AdvisoryResponse`: final structured advisory output, including reasoning, references, confidence, and review flags.

This schema is the output contract for the advisory pipeline.

### `schemas/document.py`

Currently empty. Likely intended for future document metadata/request models.

### `schemas/review.py`

Currently empty. Likely intended for review workflow models.

### `schemas/__init__.py`

Re-exports common enums/models so they can be imported more cleanly from `schemas`.

## `ingestion/`

This folder turns raw uploaded files into structured chunks ready for indexing.

### `ingestion/parser.py`

This is the document parsing and cleanup layer.

Main pieces:

- `ParsedDocument`: Pydantic model representing one fully parsed document.
- `fingerprint(text)`: creates a SHA-256 hash of cleaned text for duplicate detection.
- `clean_extracted_text(text)`: removes page numbers, footnote markers, separators, boilerplate, and normalizes whitespace.
- `remove_repeated_lines(text)`: removes lines repeated across almost all pages, which is useful for headers/footers.
- `validate_extraction(text)`: generates a small extraction-quality report.
- `enrich_metadata(raw_text, metadata)`: extracts chapter, section, notification number, HS code, and customs section from the text itself.
- `extract_pdf_page(page)`: extracts both table content and normal block text from a PDF page while avoiding duplication.
- `parse_pdf(...)`, `parse_docx(...)`, `parse_xlsx(...)`: format-specific parsers.
- `parse_document(...)`: main entrypoint that chooses the parser, cleans the text, enriches metadata, computes hash, and returns a `ParsedDocument`.

What we are doing here:

We are converting messy legal documents into a single reliable internal format before chunking. This step matters because retrieval quality depends heavily on text cleanliness and metadata quality.

### `ingestion/chunker.py`

This is the legal-aware chunking layer.

Main pieces:

- `DocumentChunk`: structured chunk model with chunk text, page number, source fields, and metadata.
- `extract_page_number(...)`: pulls the first `[PAGE N]` marker from chunk text.
- `extract_clean_text(...)`: removes page/sheet tags from final chunk content.
- `estimate_tokens(...)`: rough token estimate for later scoring.
- `LEGAL_BOUNDARIES`: regex hints for common legal section boundaries.
- `split_legal_sections(...)`: splits long text by legal structure before recursive chunking.
- `chunk_document(...)`: main entrypoint that uses `RecursiveCharacterTextSplitter`, preserves legal structure, tracks last seen page number, drops tiny/noisy chunks, and extracts per-chunk metadata.

What we are doing here:

We are trying to chunk legal text in a way that keeps sections meaningful. Instead of blindly splitting every N characters, the code respects chapter/section/rule boundaries so retrieval returns more complete legal reasoning.

### `ingestion/embedder.py`

This is the main indexing pipeline.

Main pieces:

- `EmbeddingManager`: singleton wrapper around `OllamaEmbeddings` so the model loads once.
- `l2_normalize(...)`: normalizes vectors before storage.
- `get_chroma_client()` and `get_collection()`: open the persistent ChromaDB store.
- `query_collection(...)`: simple direct vector-search helper returning similarities.
- `is_already_ingested(...)`: checks processed JSON records to avoid duplicate documents.
- `ingest_document(...)`: full pipeline for parse -> chunk -> embed -> store -> rebuild BM25 -> save record.
- `store_batch_with_retry(...)`: embeds/stores chunk batches with retry logic.
- `delete_document(...)`: removes a document from Chroma and processed-record storage.
- `save_processed_record(...)`, `delete_processed_record(...)`, `get_all_document_records(...)`: manage JSON records under `data/processed`.

What we are doing here:

We are building the searchable knowledge base. Chunks become embeddings in ChromaDB, and the same chunk set is also indexed lexically with BM25 so retrieval can use both semantic and keyword matching.

### `ingestion/test_parser.py`

Quick parser-only test script.

- Parses one PDF.
- Prints the first lines of extracted text.
- Prints stats like character count, page count, and document hash.

Useful when debugging extraction quality before doing full ingestion.

### `ingestion/__init__.py`

Currently empty package marker.

## `rag/`

This folder contains retrieval, confidence scoring, and LLM orchestration.

### `rag/bm25_manager.py`

Keyword-search manager using BM25.

Main pieces:

- `BM25Manager` stores chunk text, chunk metadata, and chunk IDs.
- `tokenize(...)` lowercases and strips punctuation.
- `build_index(...)` builds a BM25 index from chunk text.
- `search(...)` returns top-scoring lexical matches.
- `save_index(...)` and `load_index(...)` persist the BM25 state to `rag/bm25_index.pkl`.

What we are doing here:

We are adding classic keyword retrieval to complement embeddings. This helps when exact legal phrases, section numbers, tariff codes, or references matter.

### `rag/retriever.py`

This is the core retrieval engine.

Main pieces:

- Loads `BM25Manager` and `EmbeddingManager`.
- `RetrievalResult`: structured output of retrieval plus confidence details.
- `cosine_similarity(...)`: dot product for normalized vectors.
- `mmr_rerank(...)`: reranks results to reduce redundancy and improve diversity.
- `add_query_prefix(...)`: adds `search_query:` for `nomic-embed-text`.
- `retrieve(...)`: full retrieval pipeline.

What `retrieve(...)` does:

1. Adjusts `top_k` based on query length.
2. Expands `top_k` for cross-document style questions.
3. Embeds the query.
4. Fetches candidates from ChromaDB.
5. Converts Chroma cosine distances into similarities.
6. Gets BM25 results.
7. Blends dense similarity and BM25 scores.
8. Optionally reranks with MMR.
9. Builds `SourceReference` objects.
10. Calls `calculate_confidence(...)`.
11. Returns a `RetrievalResult`.

What we are doing here:

We are finding the most relevant evidence chunks and estimating how trustworthy that evidence set is before the LLM answers.

### `rag/confidence_score.py`

This file scores retrieval confidence using multiple signals.

Main pieces:

- `DOC_TYPE_AUTHORITY`: authority weights for different document types.
- Thresholds like `STRONG_MATCH_THRESHOLD` and `WEAK_MATCH_THRESHOLD`.
- `ConfidenceResult`: structured output model.
- `calculate_confidence(...)`: combines four signals.
- `should_review(...)`: decides if retrieval should be flagged for human review.
- `clamp(...)`: bounds values to `[0, 1]`.

The four signals are:

1. `retrieval_quality`: how strong the similarity scores are.
2. `source_authority`: how authoritative the retrieved document types are.
3. `source_agreement`: whether scores are consistent across retrieved chunks.
4. `coverage_bonus`: whether enough strong chunks were found.

What we are doing here:

We are not trusting retrieval blindly. This file tries to estimate if the evidence is strong enough for a reliable advisory answer.

### `rag/chain.py`

This is the advisory-generation layer.

Main pieces:

- `LLMManager`: singleton wrapper around `OllamaLLM`.
- `ADVISORY_PROMPT`: strict prompt that forces grounded JSON output.
- `build_context(...)`: builds the source context block for the LLM.
- `parse_json(...)`, `parse_sections(...)`, `parse_fallback(...)`, `parse_llm_response(...)`: defensive response parsers.
- `HIGH_SEVERITY_FLAGS`: risk flags that always trigger review.
- `requires_human_review(...)`: decides whether the generated answer needs manual review.
- `preprocess_query(...)`: trims and normalizes user input.
- `generate_advisory(...)`: full retrieve -> prompt -> LLM -> parse -> review-decision flow.

What we are doing here:

We are turning retrieved evidence into a structured customs advisory while trying to reduce hallucination risk. The model is explicitly told to answer only from the provided sources.

### `rag/__init__.py`

Currently empty package marker.

### `rag/bm25_index.pkl`

Saved BM25 index artifact.

- This is generated data, not handwritten source code.
- It stores the pickled BM25 object plus chunk text, chunk IDs, and metadata.

## `review/`

This folder looks reserved for human-review persistence.

### `review/store.py`

Currently empty. Likely meant to store pending/approved/rejected advisory review records.

### `review/__init__.py`

Currently empty package marker.

## `routers/`

These files are FastAPI route placeholders right now.

### `routers/advisory.py`

Currently empty. Likely intended to expose advisory-generation endpoints.

### `routers/documents.py`

Currently empty. Likely intended for document listing/deletion endpoints.

### `routers/ingestion.py`

Currently empty. Likely intended for upload/ingestion endpoints.

### `routers/review.py`

Currently empty. Likely intended for human-review endpoints.

### `routers/__init__.py`

Currently empty package marker.

## Other project files

### `.env`

Currently empty in the repo snapshot. Intended for environment variables like model names or runtime settings.

### `.gitignore`

Git ignore rules for local/generated files.

### `eval.json`

Evaluation dataset for the customs RAG system.

- Defines 20 questions.
- Groups them into categories like factual, table-based, multi-page, cross-document, and out-of-domain.
- Specifies expected answer/source behavior.
- Describes what metrics to track.

This is not application logic, but it is very important for measuring retrieval and answer quality.

## Runtime / Generated Folders

These are part of the project state, but not really source-code modules:

- `data/raw/`: input documents to ingest.
- `data/processed/`: per-document processed JSON records.
- `vector_store/`: ChromaDB persistent storage.
- `logs/`: rotating log files.
- `__pycache__/`: Python bytecode cache.
- `.venv/`: local virtual environment and installed packages.

## What Is Already Built vs Not Built Yet

### Already built

- Multi-format document parsing.
- Cleanup and metadata enrichment.
- Legal-aware chunking.
- Embedding storage in ChromaDB.
- BM25 keyword index.
- Retrieval with score blending and MMR.
- Confidence scoring.
- LLM-based advisory generation.

### Not built or only scaffolded

- FastAPI app entrypoint in `main.py`.
- API routers in `routers/`.
- Review persistence logic in `review/store.py`.
- Dedicated document/review schema files.

## In One Sentence

This codebase is building a grounded customs-advisory assistant: it ingests legal documents, indexes them for retrieval, scores the reliability of retrieved evidence, and then uses an LLM to produce a source-based advisory answer with human-review safeguards.
