import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, TypedDict
from uuid import uuid4

import httpx
from opentelemetry import propagate
from langgraph.graph import END, START, StateGraph
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from redis.asyncio import Redis

from app.config import (
    AGENT_HTTP_TIMEOUT_SECONDS,
    COVERAGE_AGENT_URL,
    INTAKE_AGENT_URL,
    MAX_AGENT_RETRIES,
    PROMPT_VERSION,
    REDIS_URL,
    RUN_TIMEOUT_SECONDS,
)
from app.guardrails import filter_recommendation
from app.logging_config import configure_logging, correlation_id
from app.models import AgentClassification, Citation, CoverageAssessment
from app.telemetry import configure_tracing, inject_trace_context

configure_logging()
logger = logging.getLogger(__name__)
tracer = configure_tracing("agent-worker")
RUNS = Counter("fnol_runs_total", "Runs by terminal status", ["status"])
RUN_LATENCY = Histogram("fnol_run_seconds", "End-to-end claim run latency")
WORKER_UP = Gauge("fnol_worker_up", "Worker process is available")
WORKER_START_TIME = Gauge("fnol_worker_process_start_time_seconds", "Worker process start time")
STEPS = Counter("fnol_agent_steps_total", "Agent steps by agent", ["agent", "result"])
STEP_LATENCY = Histogram(
    "fnol_agent_step_seconds", "Agent step latency", ["agent"]
)
RETRIES = Counter("fnol_agent_retries_total", "Agent HTTP retries", ["agent"])
TOOL_CALLS = Counter("fnol_tool_calls_total", "Coverage agent tool calls", ["tool"])
QUEUE_DEPTH = Gauge("fnol_queue_depth", "Queued claim runs")


class CircuitBreaker:
    def __init__(self, threshold: int = 3, reset_seconds: float = 30) -> None:
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.open_until = 0.0

    def before_call(self, service_name: str) -> None:
        if time.monotonic() < self.open_until:
            STEPS.labels(agent=service_name.replace(" ", "-"), result="failure").inc()
            raise RuntimeError(f"{service_name} circuit breaker is open")

    def success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.reset_seconds


CIRCUIT_BREAKERS = {
    "intake agent": CircuitBreaker(),
    "coverage agent": CircuitBreaker(),
}


class RunState(TypedDict, total=False):
    run: dict[str, Any]
    classification: dict[str, Any]
    assessment: dict[str, Any]
    recommendation: dict[str, Any]
    trace: list[dict[str, Any]]
    retry_count: int


def verify_assessment(assessment: CoverageAssessment) -> bool:
    retrieved = {
        (item.clause_id, item.source, item.passage)
        for item in assessment.retrieved_passages
    }
    return bool(assessment.citations) and all(
        (item.clause_id, item.source, item.passage) in retrieved
        for item in assessment.citations
    )


def choose_route(
    fnol: dict[str, Any],
    classification: AgentClassification,
    assessment: CoverageAssessment,
) -> tuple[str, str]:
    if assessment.needs_human_review or not verify_assessment(assessment):
        return "investigation", "Human review required because policy use, evidence, or verification needs attention."
    if (
        classification.claim_type.value in {"total_loss", "third_party"}
        or float(fnol["claimed_amount"]) >= 100_000
    ):
        return "surveyor_assignment", "Claim type or reported amount requires surveyor assessment."
    return "fast_track", "Demo triage criteria indicate a low-complexity claim for handler review."


async def post_with_retry(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], service_name: str
) -> dict[str, Any]:
    breaker = CIRCUIT_BREAKERS[service_name]
    breaker.before_call(service_name)
    delay = 0.2
    last_error: Exception | None = None
    for attempt in range(MAX_AGENT_RETRIES + 1):
        try:
            headers: dict[str, str] = {}
            inject_trace_context(headers)
            headers["X-Correlation-ID"] = correlation_id.get()
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            breaker.success()
            return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as error:
            last_error = error
            if attempt < MAX_AGENT_RETRIES:
                RETRIES.labels(agent=service_name.replace(" ", "-")).inc()
                await asyncio.sleep(delay)
                delay *= 2
    STEPS.labels(agent=service_name.replace(" ", "-"), result="failure").inc()
    breaker.failure()
    raise RuntimeError(f"{service_name} unavailable after bounded retries: {last_error}")


def create_graph(client: httpx.AsyncClient):
    async def classify_node(state: RunState) -> RunState:
        started = time.perf_counter()
        with tracer.start_as_current_span("agent.intake"):
            result = await post_with_retry(
                client,
                f"{INTAKE_AGENT_URL}/classify",
                {"fnol": state["run"]["fnol"]},
                "intake agent",
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        STEP_LATENCY.labels(agent="intake-agent").observe(elapsed_ms / 1000)
        state["classification"] = result
        state.setdefault("trace", []).append(
            {
                "step": "intake_classification",
                "agent": "intake-agent",
                "latency_ms": elapsed_ms,
                "result": result,
                "prompt_version": PROMPT_VERSION,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        )
        STEPS.labels(agent="intake-agent", result="success").inc()
        return state

    async def coverage_node(state: RunState) -> RunState:
        started = time.perf_counter()
        with tracer.start_as_current_span("agent.coverage"):
            result = await post_with_retry(
                client,
                f"{COVERAGE_AGENT_URL}/assess",
                {
                    "fnol": state["run"]["fnol"],
                    "classification": state["classification"],
                    "prompt_version": PROMPT_VERSION,
                },
                "coverage agent",
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        STEP_LATENCY.labels(agent="coverage-agent").observe(elapsed_ms / 1000)
        state["assessment"] = result
        tool_calls = result.get("tool_calls", [])
        state.setdefault("trace", []).append(
            {
                "step": "policy_retrieval_and_assessment",
                "agent": "coverage-agent",
                "latency_ms": elapsed_ms,
                "searches": [
                    "claim-type coverage",
                    "vehicle-use restriction and exclusion",
                ],
                "passages": result["retrieved_passages"],
                "tool_calls": tool_calls,
                "prompt_version": PROMPT_VERSION,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        )
        for item in tool_calls:
            TOOL_CALLS.labels(tool=item["tool"]).inc()
        STEPS.labels(agent="coverage-agent", result="success").inc()
        return state

    async def verify_and_route_node(state: RunState) -> RunState:
        fnol = state["run"]["fnol"]
        classification = AgentClassification.model_validate(state["classification"])
        assessment = CoverageAssessment.model_validate(state["assessment"])
        valid = verify_assessment(assessment)
        if valid:
            route, route_reason = choose_route(fnol, classification, assessment)
            position = filter_recommendation(assessment.coverage_position)
            citations = [item.model_dump() for item in assessment.citations]
            status = "escalated" if assessment.needs_human_review else "completed"
        else:
            route = "investigation"
            route_reason = "Citation verification failed; human review required."
            position = "No verified coverage position can be returned."
            citations = []
            status = "escalated"
        state["recommendation"] = {
            "status": status,
            "claim_type": classification.claim_type.value,
            "queue": route,
            "reason": route_reason,
            "coverage_position": position,
            "citations": citations,
            "escalation_reason": assessment.reason if status == "escalated" else None,
            "confidence": assessment.confidence if valid else 0.0,
            "prompt_version": PROMPT_VERSION,
        }
        state.setdefault("trace", []).append(
            {
                "step": "citation_verification_and_routing",
                "agent": "orchestrator",
                "latency_ms": 0,
                "verification_passed": valid,
                "citation_count": len(citations),
                "prompt_version": PROMPT_VERSION,
            }
        )
        return state

    graph = StateGraph(RunState)
    graph.add_node("classify", classify_node)
    graph.add_node("coverage", coverage_node)
    graph.add_node("verify_and_route", verify_and_route_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "coverage")
    graph.add_edge("coverage", "verify_and_route")
    graph.add_edge("verify_and_route", END)
    return graph.compile()


async def process_run(redis_client: Redis, run_id: str, client: httpx.AsyncClient) -> None:
    key = f"fnol:run:{run_id}"
    raw = await redis_client.get(key)
    if raw is None:
        logger.error("queued run missing run_id=%s", run_id)
        return
    run = json.loads(raw)
    run["status"] = "processing"
    run["started_at"] = datetime.now(timezone.utc).isoformat()
    run["worker_id"] = str(uuid4())
    await redis_client.set(key, json.dumps(run))
    correlation_id.set(run["correlation_id"])
    try:
        graph = create_graph(client)
        parent_context = propagate.extract({"traceparent": run.get("traceparent") or ""})
        with tracer.start_as_current_span(
            "claim.run",
            context=parent_context,
            attributes={"fnol.run_id": run_id, "fnol.prompt_version": PROMPT_VERSION},
        ):
            state = await asyncio.wait_for(
                graph.ainvoke({"run": run, "trace": []}),
                timeout=RUN_TIMEOUT_SECONDS,
            )
        run["status"] = state["recommendation"]["status"]
        run["recommendation"] = state["recommendation"]
        run["trace"] = state["trace"]
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUNS.labels(status=run["status"]).inc()
    except (TimeoutError, RuntimeError, httpx.HTTPError, ValueError) as error:
        run["status"] = "failed"
        run["error"] = str(error)
        run["recommendation"] = {
            "status": "failed",
            "queue": "investigation",
            "coverage_position": "The run did not complete; no coverage determination is available.",
            "citations": [],
            "confidence": 0,
            "prompt_version": PROMPT_VERSION,
        }
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUNS.labels(status="failed").inc()
        logger.exception("claim run failed run_id=%s", run_id)
    elapsed = (
        datetime.now(timezone.utc) - datetime.fromisoformat(run["submitted_at"])
    ).total_seconds()
    run["latency_ms"] = round(elapsed * 1000, 2)
    await redis_client.set(key, json.dumps(run))
    RUN_LATENCY.observe(elapsed)


async def recover_stale_runs(redis_client: Redis) -> None:
    now = datetime.now(timezone.utc)
    for key in redis_client.scan_iter(match="fnol:run:*"):
        raw = await redis_client.get(key)
        if raw is None:
            continue
        run = json.loads(raw)
        if run.get("status") != "processing":
            continue
        started_at = datetime.fromisoformat(run["started_at"])
        if (now - started_at).total_seconds() <= RUN_TIMEOUT_SECONDS + 30:
            continue
        run["status"] = "failed"
        run["error"] = "Worker stopped during execution; stale run recovered on startup."
        run["completed_at"] = now.isoformat()
        run["recommendation"] = {
            "status": "failed",
            "queue": "investigation",
            "coverage_position": "The run did not complete; no coverage determination is available.",
            "citations": [],
            "confidence": 0,
            "prompt_version": PROMPT_VERSION,
        }
        await redis_client.set(key, json.dumps(run))
        RUNS.labels(status="failed").inc()
        logger.error("stale claim run recovered run_id=%s", run["run_id"])


async def main() -> None:
    WORKER_UP.set(1)
    WORKER_START_TIME.set(time.time())
    start_http_server(8001, addr="0.0.0.0")
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    await recover_stale_runs(redis_client)
    async with httpx.AsyncClient(timeout=AGENT_HTTP_TIMEOUT_SECONDS) as client:
        while True:
            QUEUE_DEPTH.set(await redis_client.llen("fnol:queue"))
            item = await redis_client.brpop("fnol:queue", timeout=5)
            if item is None:
                continue
            _, run_id = item
            await process_run(redis_client, run_id, client)


if __name__ == "__main__":
    asyncio.run(main())
