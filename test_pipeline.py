from rag.retriever import retrieve
from rag.chain import generate_advisory
from schemas.advisory import AdvisoryQuery

QUERY = ("How fast is porsche 911? What are the safety ratings and how does it compare to similar sports cars?")


print("\n" + "=" * 80)
print("RETRIEVAL TEST")
print("=" * 80)
print(f"Query: {QUERY}\n")

retrieval = retrieve(QUERY)

print(f"Top Similarity:{retrieval.top_similarity}")
print(f"Avg Similarity:{retrieval.avg_similarity}")
print(f"Confidence Score:{retrieval.confidence_score}")
print(f"Retrieval Quality:{retrieval.retrieval_quality}")
print(f"Source Authority:{retrieval.source_authority}")
print(f"Source Agreement:{retrieval.source_agreement}")
print(f"Coverage Bonus:{retrieval.coverage_bonus}")
print(f"Strong Chunks:{retrieval.strong_chunks}")
print(f"Total Retrieved:{retrieval.total_retrieved}")
print(f"Is Confident:{retrieval.is_confident}")
print(f"Human Review:{retrieval.human_review_required}")

if retrieval.human_review_reason:
    print(f"Review Reason:{retrieval.human_review_reason}")

print("\nRetrieved Chunks:")
for idx, chunk in enumerate(retrieval.chunks, 1):
    print("\n" + "-" * 60)
    print(f"Chunk {idx}")
    print(f"Source:{chunk.source_name}")
    print(f"Doc Type:{chunk.doc_type}")
    print(f"Page:{chunk.page_number}")
    print(f"Score:{chunk.similarity_score}")
    print(f"Reference:{chunk.reference_number}")
    print(f"Text:{chunk.chunk_text}")


print("\n\n" + "=" * 80)
print("FULL ADVISORY TEST")
print("=" * 80)
print(f"Query: {QUERY}\n")

query_obj = AdvisoryQuery(query=QUERY, top_k=6)
response = generate_advisory(query_obj)

print(f"Session ID:{response.session_id}")
print(f"Confidence:{response.confidence_score}")
print(f"Human Review:{response.human_review_required}")

print("\nSHORT ANSWER")
print("-" * 40)
print(response.short_answer)

print("\nCLASSIFICATION")
print("-" * 40)
print(response.classification or "N/A")

print("\nREASONING")
print("-" * 40)
print(response.reasoning)

print("\nALTERNATE VIEWS")
print("-" * 40)
print(response.alternate_views or "None")

print("\nRISK FLAGS")
print("-" * 40)
if response.risk_flags:
    for flag in response.risk_flags:
        print(f"  - {flag}")
else:
    print("None")

print("\nSOURCES")
print("-" * 40)
for s in response.source_references:
    print(f"  {s.source_name} | page {s.page_number} | score {s.similarity_score}")
