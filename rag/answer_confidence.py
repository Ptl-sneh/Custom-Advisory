from sentence_transformers import CrossEncoder
import re

class RerankerManager:

    instance = None
    reranker = None

    def __new__(cls):

        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    def get_model(self):

        if self.reranker is None:

            self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        return self.reranker


reranker_manager = RerankerManager()


def extract_claims(answer: str):

    lines = re.split(r"[.!?\n]", answer)

    claims = []

    for line in lines:

        line = line.strip()

        if len(line) > 15:
            claims.append(line)

    return claims


def calculate_answer_confidence(answer, retrieved_chunks, retrieval_confidence):

    context = " ".join(chunk.chunk_text for chunk in retrieved_chunks)

    # 1 Evidence coverage
    reranker = reranker_manager.get_model()

    evidence_score = reranker.predict([(answer, context)])[0]

    evidence_coverage = max(0, min(1, float(evidence_score)))

    # 2 Claim support

    claims = extract_claims(answer)

    supported = 0

    supported = 0

    reranker = reranker_manager.get_model()

    for claim in claims:

        best_score = 0.0

        for chunk in retrieved_chunks:

            score = reranker.predict([
                (claim, chunk.chunk_text)
            ])[0]

            best_score = max(
                best_score,
                float(score)
            )

        if best_score >= 0.7:
            supported += 1



    citation_support = supported / len(claims) if claims else 0.5

    # 3 Hallucination penalty

    hallucination_penalty = 0.0 if citation_support > 0.8 else 0.2

    final = (
        0.40 * evidence_coverage
        + 0.30 * citation_support
        + 0.20 * retrieval_confidence
        + 0.10 * min(len(retrieved_chunks) / 5, 1.0)
    )

    final -= hallucination_penalty

    return round(max(0, min(1, final)), 3)
