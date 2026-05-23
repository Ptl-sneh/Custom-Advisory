from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from schemas.common import ReviewStatus

class SourceReference(BaseModel):
    doc_id: str
    source_name: str
    doc_type: str
    reference_number: Optional[str] = None
    page_number: Optional[int] = None
    chunk_text: str
    similarity_score: float

class AdvisoryQuery(BaseModel):
    query: str
    top_k: int = 6

class AdvisoryResponse(BaseModel):
    session_id: str
    query: str
    short_answer: str
    classification: Optional[str] = None
    reasoning: str
    alternate_views: Optional[str] = None
    risk_flags: list[str]
    source_references: list[SourceReference]
    confidence_score: float
    human_review_required: bool
    review_status: ReviewStatus
    created_at: datetime
