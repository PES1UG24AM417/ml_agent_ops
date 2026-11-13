# Architecture and design decisions

The API owns the public data contract and queues only validated requests.
Redis provides durable queue/run state, rate-limit counters, idempotency,
quarantine, and feedback. The LangGraph worker is horizontally scalable and
hands off to two independently deployed agent services: intake classification
and policy coverage. Chroma persists in its own named volume; Prometheus,
Grafana, and Jaeger are operational support services.

The intake role classifies the claim and derives use entities. The coverage role
searches for apparent cover, then performs a second retrieval for modifying
restrictions or exclusions. The worker verifies every citation against the
retrieved text before routing. Missing evidence, an unrecognized claim, a
possible use restriction, or invalid citation results in human review. Model
training, settlement calculation, and automated approve/deny decisions are
intentionally out of scope.

The demo uses a deterministic local hashing embedder to avoid credentials,
network downloads, and provider cost. It is not a semantic-quality substitute
for an evaluated sentence-transformer. The synthetic corpus is only for
end-to-end testing; no policy conclusion should be drawn from it.

## Predeclared performance target

Target complete-run p95: **10 seconds at 10 concurrent stubbed requests**.
The run starts at submission and ends at a terminal result. This is a local
development target, set before measurement, not an observed result. The demo
agents are deterministic rather than a latency-realistic external LLM stub, so
the benchmark must be rerun and documented with a delay stub before submission.
