import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
REVIEW_STORE_PATH = os.path.join(BASE_DIR, "review", "review_store.json")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5")

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Retrieval
TOP_K = 6

CONFIDENCE_THRESHOLD = 0.8

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
