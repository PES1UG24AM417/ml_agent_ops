import asyncio
import re
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.models import AgentClassification, ClaimType
from app.telemetry import configure_tracing
from app.config import MODEL_STUB_DELAY_SECONDS
from app.logging_config import correlation_id
from opentelemetry import propagate

app = FastAPI(title="FNOL Intake Agent", version="1.0.0")
tracer = configure_tracing("intake-agent")


@app.middleware("http")
async def trace_request(request: Request, call_next):
    token = correlation_id.set(request.headers.get("X-Correlation-ID", "-"))
    parent_context = propagate.extract(dict(request.headers))
    try:
        with tracer.start_as_current_span("http.request", context=parent_context):
            return await call_next(request)
    finally:
        correlation_id.reset(token)


class IntakeInput(BaseModel):
    fnol: dict[str, Any]


def classify(fnol: dict[str, Any]) -> AgentClassification:
    text = fnol["description"].lower()
    if re.search(r"\b(stolen|theft|vehicle was taken|car was taken)\b", text):
        claim_type = ClaimType.THEFT
    elif re.search(r"\b(thief took|taken by a thief)\b", text):
        claim_type = ClaimType.THEFT
    elif re.search(r"\b(total loss|written off|beyond repair)\b", text):
        claim_type = ClaimType.TOTAL_LOSS
    elif re.search(
        r"\b(third[- ]party|other vehicle|another vehicle|injur(?:y|ies|ed)|pedestrian)\b",
        text,
    ):
        claim_type = ClaimType.THIRD_PARTY
    elif re.search(r"\b(damag|collision|crash|accident|dent|broken)\w*\b", text):
        claim_type = ClaimType.OWN_DAMAGE
    else:
        claim_type = ClaimType.UNKNOWN

    use_text = fnol["vehicle_use"]
    if re.search(r"\b(food delivery|delivery run|deliver(?:y|ies))\b", text):
        use_text = "food_delivery"
    elif re.search(r"\b(taxi|ride.?hail|passenger for hire)\b", text):
        use_text = "taxi"
    elif re.search(r"\b(commercial|business use|goods for hire)\b", text):
        use_text = "commercial"
    elif re.search(r"\b(rac(?:e|ing)|speed test(?:ing)?|hire or reward)\b", text):
        use_text = "restricted_activity"

    unsupported_circumstances = bool(
        re.search(r"\b(unclear|unknown event|unknown cause|not described in the policy)\b", text)
    )

    return AgentClassification(
        claim_type=claim_type,
        entities={
            "policy_number": fnol["policy_number"],
            "loss_date": fnol["loss_date"],
            "vehicle_registration": fnol["vehicle_registration"],
            "vehicle_use_reported": fnol["vehicle_use"],
            "vehicle_use_detected": use_text,
            "possible_unsupported_circumstances": str(unsupported_circumstances).lower(),
            "claimed_amount": str(fnol["claimed_amount"]),
        },
        confidence=0.9 if claim_type != ClaimType.UNKNOWN else 0.35,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "intake"}


@app.post("/classify", response_model=AgentClassification)
async def classify_request(payload: IntakeInput) -> AgentClassification:
    with tracer.start_as_current_span("intake.classify"):
        await asyncio.sleep(MODEL_STUB_DELAY_SECONDS)
        return classify(payload.fnol)
