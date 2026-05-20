import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
REVIEW_STORE_PATH = os.path.join(BASE_DIR, "review", "review_store.json")

# Ollama models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Chunking
CHUNK_SIZE = 500  # Each document is split into pieces of ~500 tokens/characters before being stored in ChromaDB.
CHUNK_OVERLAP = 80  # When splitting documents into chunks, this is the number of tokens/characters that overlap between consecutive chunks. This helps maintain context across chunks during retrieval.

# Retrieval
TOP_K = 6  # number of chunks to retrieve
CONFIDENCE_THRESHOLD = 0.8  # below this → flag low confidence

# Document types (for metadata tagging)
DOC_TYPES = [
    "Circular",
    "Notification",
    "Tariff Schedule",
    "HSN Classification",
    "Case Law",
    "BIS / Export Control",
    "Customs Act",
    "Trade Policy",
    "Other"
]