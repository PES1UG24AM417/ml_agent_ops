from datetime import date
from enum import StrEnum
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VehicleUse(StrEnum):
    PRIVATE = "private"
    COMMERCIAL = "commercial"
    FOOD_DELIVERY = "food_delivery"
    TAXI = "taxi"
    UNKNOWN = "unknown"


class ClaimType(StrEnum):
    OWN_DAMAGE = "own_damage"
    THIRD_PARTY = "third_party"
    THEFT = "theft"
    TOTAL_LOSS = "total_loss"
    UNKNOWN = "unknown"


class FnolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    policy_number: str = Field(min_length=4, max_length=32, pattern=r"^[A-Za-z0-9/-]+$")
    loss_date: date
    vehicle_registration: str = Field(min_length=8, max_length=15)
    description: str = Field(min_length=20, max_length=3000)
    claimed_amount: float = Field(gt=0, le=10_000_000)
    vehicle_use: VehicleUse

    @field_validator("vehicle_registration")
    @classmethod
    def normalize_registration(cls, value: str) -> str:
        normalized = re.sub(r"[\s-]", "", value).upper()
        if not re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", normalized):
            raise ValueError("must be a valid Indian vehicle registration")
        return normalized

    @model_validator(mode="after")
    def validate_loss_date(self) -> "FnolRequest":
        if self.loss_date > date.today():
            raise ValueError("loss_date cannot be in the future")
        return self


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    satisfactory: bool
    reason: str | None = Field(default=None, max_length=1000)


class Citation(BaseModel):
    clause_id: str
    source: str
    passage: str
    category: str = "unknown"


class AgentClassification(BaseModel):
    claim_type: ClaimType
    entities: dict[str, str]
    confidence: float = Field(ge=0, le=1)


class CoverageAssessment(BaseModel):
    coverage_position: str
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1)
    needs_human_review: bool
    reason: str | None = None
    retrieved_passages: list[Citation]
    tool_calls: list[dict[str, str]] = Field(default_factory=list)
