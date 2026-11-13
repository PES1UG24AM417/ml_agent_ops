import json
import logging
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from redis.exceptions import RedisError
from redis.asyncio import Redis
from opentelemetry import propagate

from app.config import RATE_LIMIT_PER_MINUTE, REDIS_URL
from app.guardrails import inspect_untrusted_text
from app.logging_config import configure_logging, correlation_id
from app.models import FeedbackRequest, FnolRequest
from app.telemetry import configure_tracing, inject_trace_context

configure_logging()
logger = logging.getLogger(__name__)
tracer = configure_tracing("fnol-api")
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
SUBMISSIONS = Counter("fnol_submissions_total", "FNOL submissions", ["result"])
FEEDBACK = Counter("fnol_feedback_total", "Run feedback", ["satisfactory"])
QUARANTINED = Counter("fnol_quarantined_total", "Rejected requests quarantined", ["reason"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await redis_client.aclose()


app = FastAPI(
    title="Motor Insurance Claim Intake and Triage API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    token = correlation_id.set(request_id)
    started = time.perf_counter()
    try:
        parent_context = propagate.extract(dict(request.headers))
        with tracer.start_as_current_span("http.request", context=parent_context) as span:
            span.set_attribute("http.request.method", request.method)
            span.set_attribute("url.path", request.url.path)
            response = await call_next(request)
            span.set_attribute("http.response.status_code", response.status_code)
        response.headers["X-Correlation-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - started) * 1000, 2))
        return response
    finally:
        correlation_id.reset(token)


@app.exception_handler(RequestValidationError)
async def quarantine_invalid_request(request: Request, exc: RequestValidationError):
    body = exc.body
    errors = jsonable_encoder(exc.errors())
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "path": str(request.url.path),
        "client": request.client.host if request.client else "unknown",
        "body": body if isinstance(body, (dict, list, str, int, float, bool)) else repr(body),
        "errors": errors,
    }
    try:
        await redis_client.rpush("fnol:quarantine", json.dumps(record, default=str))
    except RedisError as error:
        logger.error("failed to quarantine invalid request: %s", error)
        return JSONResponse(
            status_code=503,
            content={"detail": "Invalid request could not be durably quarantined; retry later."},
        )
    SUBMISSIONS.labels(result="rejected").inc()
    QUARANTINED.labels(reason="schema").inc()
    return JSONResponse(status_code=422, content={"detail": errors, "quarantined": True})


@app.get("/health")
async def health() -> dict[str, str]:
    await redis_client.ping()
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/claims", status_code=202)
async def submit_claim(
    request: Request,
    fnol: FnolRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    client = request.client.host if request.client else "unknown"
    window = int(time.time() // 60)
    rate_key = f"fnol:rate:{client}:{window}"
    count = await redis_client.incr(rate_key)
    if count == 1:
        await redis_client.expire(rate_key, 120)
    if count > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Client rate limit exceeded")

    fnol_data = fnol.model_dump(mode="json")
    unsafe = inspect_untrusted_text(fnol.description)
    if unsafe:
        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
            "client": client,
            "body": fnol_data,
            "errors": unsafe,
        }
        await redis_client.rpush("fnol:quarantine", json.dumps(record))
        SUBMISSIONS.labels(result="rejected_guardrail").inc()
        QUARANTINED.labels(reason="guardrail").inc()
        raise HTTPException(status_code=422, detail={"errors": unsafe, "quarantined": True})

    idempotency_storage_key = None
    idempotency_lock_key = None
    if idempotency_key:
        if len(idempotency_key) > 128:
            raise HTTPException(status_code=400, detail="Idempotency-Key must be at most 128 characters")
        idempotency_storage_key = f"fnol:idempotency:{client}:{idempotency_key}"
        idempotency_lock_key = f"fnol:idempotency-lock:{client}:{idempotency_key}"
        cached_id = await redis_client.get(idempotency_storage_key)
        if cached_id:
            status_data = await redis_client.get(f"fnol:run:{cached_id}")
            status = json.loads(status_data)["status"] if status_data else "queued"
            return {"run_id": cached_id, "status": status, "status_url": f"/v1/runs/{cached_id}"}
        acquired = await redis_client.set(idempotency_lock_key, "1", ex=30, nx=True)
        if not acquired:
            for _ in range(20):
                await asyncio.sleep(0.05)
                cached_id = await redis_client.get(idempotency_storage_key)
                if cached_id:
                    return {
                        "run_id": cached_id,
                        "status": "queued",
                        "status_url": f"/v1/runs/{cached_id}",
                    }
            raise HTTPException(
                status_code=409,
                detail="An identical submission is in progress; retry with the same Idempotency-Key.",
            )

    run_id = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc).isoformat()
    run = {
        "run_id": run_id,
        "status": "queued",
        "submitted_at": submitted_at,
        "correlation_id": correlation_id.get(),
        "fnol": fnol_data,
        "trace": [],
    }
    trace_headers: dict[str, str] = {}
    inject_trace_context(trace_headers)
    run["traceparent"] = trace_headers.get("traceparent")
    pipeline = redis_client.pipeline(transaction=True)
    pipeline.set(f"fnol:run:{run_id}", json.dumps(run))
    pipeline.lpush("fnol:queue", run_id)
    if idempotency_storage_key and idempotency_lock_key:
        pipeline.set(idempotency_storage_key, run_id, ex=86400)
        pipeline.delete(idempotency_lock_key)
    await pipeline.execute()
    SUBMISSIONS.labels(result="accepted").inc()
    logger.info("FNOL accepted run_id=%s", run_id)
    return {"run_id": run_id, "status": "queued", "status_url": f"/v1/runs/{run_id}"}


@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str):
    raw = await redis_client.get(f"fnol:run:{run_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(raw)


@app.post("/v1/runs/{run_id}/feedback", status_code=202)
async def submit_feedback(run_id: str, feedback: FeedbackRequest):
    raw = await redis_client.get(f"fnol:run:{run_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="Run not found")
    entry = {
        "run_id": run_id,
        **feedback.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis_client.rpush("fnol:feedback", json.dumps(entry))
    FEEDBACK.labels(satisfactory=str(feedback.satisfactory).lower()).inc()
    return {"status": "recorded"}


@app.get("/v1/feedback/summary")
async def feedback_summary():
    entries = await redis_client.lrange("fnol:feedback", 0, -1)
    parsed = [json.loads(entry) for entry in entries]
    return {
        "total": len(parsed),
        "satisfactory": sum(item["satisfactory"] is True for item in parsed),
        "unsatisfactory": sum(item["satisfactory"] is False for item in parsed),
    }


app.mount(
    "/",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="web",
)
