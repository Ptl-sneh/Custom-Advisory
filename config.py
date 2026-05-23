# config.py  —  Optimized configuration
# ═══════════════════════════════════════
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
VECTOR_STORE_DIR= os.path.join(BASE_DIR, "vector_store")
REVIEW_STORE_PATH = os.path.join(BASE_DIR, "review", "review_store.json")

# ── Models ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen2.5")

# ── Chunking ─────────────────────────────────────────────────────────
# WHY 800 not 500:
#   500 chars ≈ 125 tokens — too small for legal clauses that span 3–4 sentences.
#   Legal reasoning requires a full clause or subsection to be meaningful.
#   800 chars ≈ 200 tokens — captures one full legal provision with context.
#   Result: richer chunks → better embeddings → higher similarity scores.
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120   # ~15% of chunk size — standard for legal docs

# ── Retrieval ────────────────────────────────────────────────────────
TOP_K = 6   # retrieve 6 chunks; confidence score uses all 6 signals

# ── Confidence thresholds ─────────────────────────────────────────────
# CONFIDENCE_THRESHOLD: below this → flag for human review
# WHY 0.65 not 0.80:
#   0.80 is too aggressive — it flags almost everything since embedding
#   cosine similarities for domain-specific legal text rarely exceed 0.85.
#   0.65 is a calibrated threshold based on the multi-signal formula in
#   confidence_score.py (not raw similarity scores).
CONFIDENCE_THRESHOLD = 0.85

DOC_TYPES = [
    "Circular",
    "Notification",
    "Tariff Schedule",
    "HSN Classification",
    "Case Law",
    "BIS / Export Control",
    "Customs Act",
    "Trade Policy",
    "Other",
]