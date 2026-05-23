from pathlib import Path

from ingestion.embedder import ingest_document

RAW_DIR = Path("./data/raw")

metadata = {
    "doc_type": "Circular",
    "source_name": "CBIC",
    "issuing_authority": "CBIC",
    "tags": ["customs"],
}

for file in RAW_DIR.iterdir():

    if file.is_file():

        result = ingest_document(str(file), metadata)

        print(result)
