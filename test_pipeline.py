from rag.retriever import retrieve
from rag.chain import generate_advisory
from schemas.advisory import AdvisoryQuery


print("\n" + "=" * 80)
print("RETRIEVAL TEST")
print("=" * 80)

query = "What is python?"

retrieval = retrieve(query)

print(f"Top Confidence: " f"{retrieval.top_confidence}")

print(f"Average Confidence: " f"{retrieval.avg_confidence}")

print(f"Retrieved: " f"{retrieval.total_retrieved}")

for idx, chunk in enumerate(retrieval.chunks, 1):

    print("\n" + "-" * 60)

    print(f"Chunk {idx}")

    print(f"Source: " f"{chunk.source_name}")

    print(f"Score: " f"{chunk.similarity_score}")

    print(chunk.chunk_text[:250])


print("\n")
print("=" * 80)
print("FULL ADVISORY TEST")
print("=" * 80)

query_obj = AdvisoryQuery(query=query, top_k=6)

response = generate_advisory(query_obj)


print("\nSHORT ANSWER\n")

print(response.short_answer)

print("\nCLASSIFICATION\n")

print(response.classification)

print("\nREASONING\n")

print(response.reasoning)

print("\nRISK FLAGS\n")

print(response.risk_flags)

print("\nCONFIDENCE\n")

print(response.confidence_score)

print("\nREVIEW REQUIRED\n")

print(response.human_review_required)

print("\nSOURCES")

for s in response.source_references:

    print(f"{s.source_name}" f" | " f"{s.page_number}" f" | " f"{s.similarity_score}")


# print("\n")
# print("=" * 80)
# print("HALLUCINATION TEST")
# print("=" * 80)

# fake = AdvisoryQuery(query=("What duty applies " "to moon rock imports?"))

# fake_response = generate_advisory(fake)

# print(fake_response.short_answer)

# print(fake_response.confidence_score)

# print(fake_response.human_review_required)
