from rag.retriever import retrieve
from rag.chain import generate_advisory
from schemas.advisory import AdvisoryQuery

# Change this query to test different things
# Good test queries (from your actual documents):
#   "What is the HSN classification for a 36-port 100GE interface card?"
#   "What penalty was imposed on Huawei for misclassification?"
#   "What is the difference between CTH 8517 7010 and CTH 8517 6290?"
#   "What are the General Rules for Interpretation in customs tariff?"
#   "What is the ruling on Optical Transport Network product classification?"
#
# Bad query (tests out-of-scope gate):
#   "What is RAG?" → should return confidence 0.0 and human_review=True

QUERY = "What is the maximum integrated tax rate applicable on imported articles under the Customs Tariff Act, 1975?"


print("\n" + "=" * 80)
print("RETRIEVAL TEST")
print("=" * 80)
print(f"Query: {QUERY}\n")

retrieval = retrieve(QUERY)

print(f"Top Similarity:      {retrieval.top_similarity}")
print(f"Avg Similarity:      {retrieval.avg_similarity}")
print(f"Confidence Score:    {retrieval.confidence_score}")
print(f"Retrieval Quality:   {retrieval.retrieval_quality}")
print(f"Source Authority:    {retrieval.source_authority}")
print(f"Source Agreement:    {retrieval.source_agreement}")
print(f"Coverage Bonus:      {retrieval.coverage_bonus}")
print(f"Strong Chunks:       {retrieval.strong_chunks}")
print(f"Total Retrieved:     {retrieval.total_retrieved}")
print(f"Is Confident:        {retrieval.is_confident}")
print(f"Human Review:        {retrieval.human_review_required}")
if retrieval.human_review_reason:
    print(f"Review Reason:       {retrieval.human_review_reason}")

print("\nRetrieved Chunks:")
for idx, chunk in enumerate(retrieval.chunks, 1):
    print("\n" + "-" * 60)
    print(f"Chunk {idx}")
    print(f"Source:    {chunk.source_name}")
    print(f"Doc Type:  {chunk.doc_type}")
    print(f"Page:      {chunk.page_number}")
    print(f"Score:     {chunk.similarity_score}")
    print(f"Text:      {chunk.chunk_text[:300]}...")


print("\n\n" + "=" * 80)
print("FULL ADVISORY TEST")
print("=" * 80)
print(f"Query: {QUERY}\n")

query_obj = AdvisoryQuery(query=QUERY, top_k=6)
response = generate_advisory(query_obj)

print(f"Session ID:    {response.session_id}")
print(f"Confidence:    {response.confidence_score}")
print(f"Human Review:  {response.human_review_required}")

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


# Uncomment to test out-of-scope gate
# Should return: confidence=0.0, human_review=True, empty sources

# print("\n\n" + "=" * 80)
# print("OUT-OF-SCOPE GATE TEST")
# print("=" * 80)
# fake = AdvisoryQuery(query="What duty applies to moon rock imports?")
# fake_response = generate_advisory(fake)
# print(f"Short Answer: {fake_response.short_answer}")
# print(f"Confidence:   {fake_response.confidence_score}")
# print(f"Review:       {fake_response.human_review_required}")
