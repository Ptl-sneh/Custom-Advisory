"""
test_retrieval.py
==================
Run this from backend/ after ingesting your two documents.

    python test_retrieval.py

Tests retrieval quality against the actual content of:
  1. customs_excise_and_service_tax.pdf  (CESTAT case law — Huawei vs Customs)
  2. customs_tariff_act.pdf              (Customs Tariff Act)

For each query, shows:
  - Retrieved chunks with similarity scores
  - Which document/page each chunk came from
  - Confidence score breakdown (4 signals)
  - Human review flag + reason
"""

import sys
import json
from pathlib import Path

# Make sure backend/ is in path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.embedder import query_collection
from rag.confidence_score import calculate_confidence
from config import TOP_K, CONFIDENCE_THRESHOLD


# Test queries — written specifically for the two documents you have

TEST_QUERIES = [
    # From customs_excise_and_service_tax.pdf
    {
        "query": "What is the correct HSN classification for a 36-port 100GE interface card?",
        "expected_keywords": [
            "8517 7010",
            "8517 70",
            "parts",
            "PPCBA",
            "interface card",
        ],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "HSN Classification",
    },
    {
        "query": "What penalty was imposed on Huawei for customs misclassification?",
        "expected_keywords": ["2,30,000", "Section 112", "penalty", "confiscation"],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Penalty",
    },
    {
        "query": "What is the difference between CTH 8517 7010 and CTH 8517 6290?",
        "expected_keywords": ["8517 70", "8517 62", "parts", "apparatus", "OTN"],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Classification Dispute",
    },
    {
        "query": "What is the ruling on Optical Transport Network product classification?",
        "expected_keywords": [
            "OTN",
            "Optical Transport Network",
            "8517 6290",
            "notification",
        ],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Case Law",
    },
    {
        "query": "What are the General Rules for Interpretation in customs tariff classification?",
        "expected_keywords": [
            "GIR",
            "General Rules",
            "headings",
            "Section Notes",
            "Chapter Notes",
        ],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Tariff Classification Rules",
    },
    {
        "query": "What is the customs duty rate for populated printed circuit boards under 8517?",
        "expected_keywords": [
            "8517 70 10",
            "Free",
            "populated",
            "printed circuit board",
            "PPCBA",
        ],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Duty Rate",
    },
    # Cross-document / general
    {
        "query": "What is Section 28 of the Customs Act?",
        "expected_keywords": ["Section 28", "duty", "short-levied", "recovery"],
        "doc_hint": "customs_excise_and_service_tax.pdf or customs_tariff_act.pdf",
        "category": "Customs Act",
    },
    {
        "query": "How are parts of machinery classified under Section XVI?",
        "expected_keywords": [
            "Section XVI",
            "parts",
            "Note 2",
            "heading",
            "Chapter 84",
            "Chapter 85",
        ],
        "doc_hint": "customs_excise_and_service_tax.pdf",
        "category": "Classification Rules",
    },
]


# Helpers

def check_keywords(chunk_texts: list[str], keywords: list[str]) -> list[str]:
    """Return which expected keywords appear in any retrieved chunk."""
    combined = " ".join(chunk_texts).lower()
    return [kw for kw in keywords if kw.lower() in combined]


def print_separator(char="─", width=70):
    print(char * width)


def run_test(query_config: dict, top_k: int) -> dict:
    query = query_config["query"]
    keywords = query_config["expected_keywords"]
    category = query_config["category"]

    # 1. Retrieve from ChromaDB
    results = query_collection(query_text=query, top_k=top_k)

    if not results or not results.get("ids") or not results["ids"][0]:
        return {
            "query": query,
            "category": category,
            "status": "NO_RESULTS",
            "chunks_retrieved": 0,
        }

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    similarities = results.get("similarities", [[]])[0]

    # 2. Calculate confidence 
    doc_types = [m.get("doc_type", "Other") for m in metadatas]
    confidence = calculate_confidence(
        similarity_scores=similarities,
        doc_types=doc_types,
        query=query,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # 3. Keyword hit check
    found_keywords = check_keywords(documents, keywords)
    keyword_hit_rate = len(found_keywords) / len(keywords) if keywords else 0

    return {
        "query": query,
        "category": category,
        "status": "OK",
        "chunks_retrieved": len(ids),
        "similarities": similarities,
        "documents": documents,
        "metadatas": metadatas,
        "confidence": confidence,
        "found_keywords": found_keywords,
        "missing_keywords": [kw for kw in keywords if kw not in found_keywords],
        "keyword_hit_rate": keyword_hit_rate,
    }


# Main



def main():
    print("\n" + "═" * 70)
    print("  RETRIEVAL TEST — Real Document Validation")
    print("  Documents: customs_excise_and_service_tax.pdf + customs_tariff_act.pdf")
    print("═" * 70)

    all_results = []
    total_queries = len(TEST_QUERIES)
    passed = 0  # keyword hit rate > 0.5

    for i, qconfig in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{i}/{total_queries}] {qconfig['category']}")
        print_separator()
        print(f"  Query: {qconfig['query']}")
        print(f"  Expected keywords: {qconfig['expected_keywords']}")
        print()

        result = run_test(qconfig, TOP_K)
        all_results.append(result)

        if result["status"] == "NO_RESULTS":
            print("  ✗ NO RESULTS — ChromaDB returned nothing")
            print("    → Check: is the document ingested? Run ingest first.")
            continue

        # Print retrieved chunks
        print(f"  Retrieved {result['chunks_retrieved']} chunks:")
        for j, (doc, meta, sim) in enumerate(
            zip(result["documents"], result["metadatas"], result["similarities"]),
            start=1,
        ):
            source = meta.get("source_name", meta.get("filename", "unknown"))
            page = meta.get("page_number", -1)
            doc_type = meta.get("doc_type", "?")
            page_str = f"p.{page}" if page and page != -1 else "p.?"
            sim_bar = "█" * int(sim * 20)  # visual bar

            print(f"\n  Chunk {j} | sim={sim:.4f} [{sim_bar:<20}]")
            print(f"          | source={source} | {page_str} | type={doc_type}")
            # Show first 200 chars of chunk
            preview = doc[:200].replace("\n", " ")
            print(f'          | "{preview}..."')

        # Confidence breakdown
        c = result["confidence"]
        print(
            f"\n  Confidence Score: {c.score:.3f}  {'✓' if c.score >= CONFIDENCE_THRESHOLD else '✗ LOW'}"
        )
        print(f"    retrieval_quality : {c.retrieval_quality:.3f} (weight 45%)")
        print(f"    source_authority  : {c.source_authority:.3f} (weight 25%)")
        print(f"    source_agreement  : {c.source_agreement:.3f} (weight 20%)")
        print(f"    coverage_bonus    : {c.coverage_bonus:.3f} (weight 10%)")
        print(
            f"    human_review      : {'YES — ' + c.human_review_reason if c.human_review_required else 'No'}"
        )

        # Keyword check
        hit_rate = result["keyword_hit_rate"]
        status_icon = "✓" if hit_rate >= 0.5 else "⚠" if hit_rate > 0 else "✗"
        print(f"\n  Keyword hit rate  : {status_icon} {hit_rate:.0%}")
        if result["found_keywords"]:
            print(f"    Found   : {result['found_keywords']}")
        if result["missing_keywords"]:
            print(f"    Missing : {result['missing_keywords']}")

        if hit_rate >= 0.5:
            passed += 1

    # Summary

    print("\n\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)

    ok_results = [r for r in all_results if r["status"] == "OK"]

    if ok_results:
        avg_top1_sim = sum(
            r["similarities"][0] for r in ok_results if r["similarities"]
        ) / len(ok_results)
        avg_confidence = sum(r["confidence"].score for r in ok_results) / len(
            ok_results
        )
        review_required = sum(
            1 for r in ok_results if r["confidence"].human_review_required
        )

        print(f"\n  Queries run          : {total_queries}")
        print(f"  Keyword pass (≥50%)  : {passed}/{total_queries}")
        print(f"  Avg top-1 similarity : {avg_top1_sim:.4f}")
        print(f"  Avg confidence score : {avg_confidence:.3f}")
        print(f"  Human review flagged : {review_required}/{len(ok_results)}")

        print(f"\n  Similarity interpretation:")
        print(f"    > 0.70 → strong match (LLM will give good answer)")
        print(f"    0.50–0.70 → moderate (answer may be incomplete)")
        print(f"    < 0.50 → weak (chunk may be irrelevant — check chunking)")

        # Diagnose problems
        low_sim = [
            r for r in ok_results if r["similarities"] and r["similarities"][0] < 0.50
        ]
        low_kw = [r for r in ok_results if r["keyword_hit_rate"] == 0]

        if low_sim:
            print(f"\n  ⚠ Low similarity queries ({len(low_sim)}):")
            for r in low_sim:
                print(f"    - [{r['category']}] top-1 sim = {r['similarities'][0]:.4f}")
            print(f"    → Possible causes:")
            print(f"       1. CHUNK_SIZE too small — legal clauses split mid-sentence")
            print(f"       2. Embedding model weak on this doc type")
            print(f"       3. Query phrasing doesn't match document language")

        if low_kw:
            print(f"\n  ⚠ Zero keyword hits ({len(low_kw)}):")
            for r in low_kw:
                print(f"    - [{r['category']}] '{r['query'][:50]}...'")
            print(f"    → The right chunks weren't retrieved at all")
            print(f"       Try: increase TOP_K in config.py, or re-check ingestion")

    else:
        print("\n  ✗ All queries returned NO RESULTS")
        print("  → Documents not ingested. Run ingestion first:")
        print()
        print("     from ingestion.embedder import ingest_document")
        print("     ingest_document('data/raw/customs_excise_and_service_tax.pdf',")
        print("         {'doc_type': 'Case Law', 'source_name': 'CESTAT Huawei 2023'})")
        print("     ingest_document('data/raw/customs_tariff_act.pdf',")
        print(
            "         {'doc_type': 'Customs Act', 'source_name': 'Customs Tariff Act 1975'})"
        )

    print("\n" + "═" * 70)
    print("  WHAT TO DO NEXT BASED ON RESULTS")
    print("═" * 70)
    print("""
  If avg top-1 similarity > 0.65 and keyword hit rate > 60%:
    → Retrieval is working. Ready to build chain.py (LLM advisory generation).

  If avg top-1 similarity is 0.40–0.65:
    → Retrieval is weak. Two options:
       A. Increase CHUNK_SIZE from 800 → 1000 in config.py and re-ingest
       B. Switch embedding model (run Colab benchmark first)

  If avg top-1 similarity < 0.40:
    → Embedding model is not matching query language to document language.
       → Run the Colab benchmark to pick a better model.

  If confidence score is consistently low (< 0.50):
    → Not an embedding problem — check DOC_TYPE metadata on ingestion.
       Case law should be tagged doc_type="Case Law"
       Tariff act should be tagged doc_type="Customs Act"
       These affect the source_authority signal in confidence_score.py
""")


if __name__ == "__main__":
    main()
