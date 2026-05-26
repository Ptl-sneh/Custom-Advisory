from rank_bm25 import BM25Okapi
import re
import pickle
from pathlib import Path

BM25_DIR = Path("rag")

BM25_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BM25_PATH = BM25_DIR / "bm25_index.pkl"


class BM25Manager:

    def __init__(self):
        self.bm25 = None
        self.chunk_store = []
        self.metadata_store = []
        self.chunk_ids = []

    def tokenize(self, text):

        text = text.lower()
        text = re.sub(r"[^\w\s/-]", " ", text)
        return text.split()

    def build_index(self, chunks):

        self.chunk_store = chunks
        tokenized_chunks = [self.tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=20):

        if self.bm25 is None:
            return []
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save_index(self):

        with open(BM25_PATH, "wb") as f:

            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunks": self.chunk_store,
                    "metadata": self.metadata_store,
                    "chunk_ids": self.chunk_ids,
                },
                f,
            )

    def load_index(self):
        if not BM25_PATH.exists():
            return False
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_store = data["chunks"]
        self.metadata_store = data["metadata"]
        self.chunk_ids = data["chunk_ids"]
        return True
