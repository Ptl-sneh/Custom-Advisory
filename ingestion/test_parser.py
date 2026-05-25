from ingestion.parser import parse_document
from schemas.common import DocType

pdf_path = "data/raw/customs_tariff_act.pdf"

metadata = {"doc_type": DocType.OTHER, "source_name": "Customs Tariff"}


parsed = parse_document(pdf_path, metadata)


print("\n===== EXTRACTION SAMPLE =====\n")

lines = parsed.raw_text.splitlines()

for line in lines[:100]:
    print(line)


print("\n===== STATS =====")

print("Characters:", len(parsed.raw_text))
print("Pages:", parsed.page_count)
print("Hash:", parsed.document_hash)
