from enum import Enum
from datetime import datetime
from pydantic import BaseModel


class DocType(str, Enum):
    CIRCULAR = "Circular"
    NOTIFICATION = "Notification"
    TARIFF_SCHEDULE = "Tariff Schedule"
    HSN_CLASSIFICATION = "HSN Classification"
    CASE_LAW = "Case Law"
    BIS_EXPORT_CONTROL = "BIS / Export Control"
    CUSTOMS_ACT = "Customs Act"
    TRADE_POLICY = "Trade Policy"
    OTHER = "Other"


class IndexingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"


class BaseResponse(BaseModel):
    success: bool
    message: str
