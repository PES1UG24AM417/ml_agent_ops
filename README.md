# Motor Insurance Claim Intake and Triage

An advisory FNOL triage service built for the UE24AM342AA5 mini project. It
validates claims before queueing them, hands work to separate intake and
coverage agent services, retrieves policy passages, verifies citations, and
escalates uncertain cases to a human. It never approves or rejects a claim.

> **Corpus notice:** `data/corpus/demo_policy.json` is illustrative demo data,
> not an insurer's policy wording. Replace it with a complete, licensed/public
> private-car policy wording and claims procedure before making coverage claims
> or presenting results as policy-grounded. The corpus ingestion command is
> ready for that workflow.

## Run locally with Docker

Prerequisites: Docker Desktop with Compose v2.

```powershell
Copy-Item .env.example .env
# Edit .env and set GRAFANA_ADMIN_USER and GRAFANA_ADMIN_PASSWORD first.
docker compose up --build
```

Services:

| Service | URL | Purpose |
|---|---|---|
| FNOL API | http://localhost:8000/docs | Submit and poll claim runs |
| Intake agent | internal port 8101 | Validate/enrich and classify FNOL |
| Coverage agent | internal port 8102 | Retrieve clauses and form a grounded position |
| Prometheus | http://localhost:9090 | Service metrics |
| Grafana | http://localhost:3000 | Local dashboard (set credentials in `.env`) |
| Jaeger | http://localhost:16686 | Trace UI |

Submit an example (the demo policy number is `DEMO-1001`):

```powershell
$body = @{
  policy_number = "DEMO-1001"
  loss_date = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
  vehicle_registration = "KA01AB1234"
  description = "My car was damaged in a collision while I was driving for personal use."
  claimed_amount = 25000
  vehicle_use = "private"
} | ConvertTo-Json
$run = Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/claims `
  -ContentType "application/json" -Body $body
$run
Invoke-RestMethod "http://localhost:8000/v1/runs/$($run.run_id)"
```

Submission returns immediately with a run ID. Poll `GET /v1/runs/{run_id}` until
the status is `completed`, `escalated`, or `failed`. Pass an `Idempotency-Key`
header to safely retry a submission. Rejected requests return field-level
errors and are quarantined; they never enter the run queue.

## API contract

- `POST /v1/claims`: validated FNOL, `202 Accepted`, `run_id`, `status_url`.
- `GET /v1/runs/{run_id}`: status, recommendation, citations, escalation reason,
  prompt version, and run metadata.
- `POST /v1/runs/{run_id}/feedback`: `satisfactory`, optional reason.
- `GET /health`, `GET /metrics`: health and Prometheus metrics.

The Pydantic request model is the ingestion contract. Its field, range,
category, and cross-field checks run before a job can be queued. Input text is
checked for prompt-injection patterns and a small set of abusive terms; an
unsafe request is rejected rather than sent to an agent.

## Architecture and failure behavior

```text
Client -> API -> Redis queue -> scalable agent-worker (LangGraph)
                                     |-> intake-agent (classification)
                                     `-> coverage-agent (LangChain + Chroma)
Redis stores run state, queue items, idempotency results, feedback and rejected
requests. API, worker, and the two agent roles are separate Compose services:
the worker can scale independently, and an agent failure is isolated from API
request handling. The worker uses bounded HTTP timeouts and retries; if a
service remains unavailable, the run is marked failed/escalated and is never
returned as a confident recommendation.

The worker's bounded graph hands normalized fields to the intake agent, then
classification and usage context to the coverage agent. The coverage agent
searches for the apparent cover and separately searches for restrictions and
exclusions (including vehicle-use limitations). The worker checks that each
citation resolves to a passage returned in that run. Missing/weak evidence or
failed verification means human escalation.
```

### Corpus

Place text-layer policy and claims-procedure PDFs under `data/corpus/source/`.
Run the idempotent parser/indexer in the coverage-agent container:

```powershell
docker compose run --rm coverage-agent python -m app.ingest
```

The current demo JSON corpus allows a clean-clone smoke test without external
credentials or model downloads. The indexer extracts text-layer PDFs, chunks
them into overlapping page passages, and assigns stable IDs tied to the source
file. Scanned PDFs require OCR before ingestion. Each passage carries source
metadata. Chroma is persisted in the `corpus-index` volume. Embeddings use a
deterministic local hashing embedder; this keeps the demo offline and
re-embedding repeatable.

## Evaluation and load testing

```powershell
docker compose run --rm agent-worker python -m app.evaluate
docker compose run --rm agent-worker locust -f load/locustfile.py --headless `
  -u 10 -r 2 -t 2m --host http://api:8000
# In a separate terminal, scale only the run workers:
docker compose up --scale agent-worker=3
```

`evaluation/scenarios.json` contains 20 deterministic scenarios, including the
food-delivery-use challenge and unanswerable cases. The CLI reports context
relevance, citation faithfulness, groundedness, and hallucination rate from
retrieved-passage/citation checks. The prompt variants are versioned in
`app/prompts.py`; evaluation reports the same metrics for both. The API's
deterministic agent endpoints act as fixed-response local stubs with a
configurable delay (`MODEL_STUB_DELAY_SECONDS`); provider-backed model latency
is not represented by this offline demo. The predeclared end-to-end p95 target
is 10 seconds at 10 concurrent stubbed requests; measure it with the Compose
deployment before claiming it was met.

Latest run against the **illustrative fixture only**: v1 → v2 context relevance
improved from 0.60 to 1.00; groundedness stayed at 0.90, faithfulness at 0.90,
and hallucination rate at 0.00. Expected triage outcome accuracy was 1.00 for
both variants. These are deterministic fixture checks, not insurer-policy
quality results.

## Configuration and security

Copy `.env.example` to `.env`, set unique local Grafana credentials, and do not
commit `.env`. The demo has no external LLM provider and requires no API key. Configuration is read at runtime. The API
has a per-client rate limit and the worker limits each run's retries and
execution time. Structured JSON logs include a correlation ID. Prometheus,
Grafana, and Jaeger are available in Compose.

## Known gaps before final submission

- Replace the illustrative corpus with the selected insurer's complete policy
  wording and claims process; verify every clause against the original PDFs.
- The deterministic demo agents are not a substitute for provider-backed
  generative reasoning. Integrate and evaluate a provider behind a controlled
  interface before claiming LLM quality.
- Run Locust against the declared SLO and record actual results, scaling effect,
  cost, and a genuine development incident. Do not invent benchmark or incident
  results.
- Add resource-utilization panels and automatic live-run quality sampling after
  the target deployment is selected.
- Review the deployment and security configuration before any non-local use.

See [docs/runbook.md](docs/runbook.md) for operational recovery steps.
