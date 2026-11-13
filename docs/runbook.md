# Runbook

## 1. Redis unavailable

- **Symptom:** API `/health` fails, submissions return service errors, or queued
  run status stops changing.
- **Confirm:** `docker compose ps` shows Redis unhealthy; check
  `docker compose logs redis`.
- **Recover:** `docker compose restart redis`, wait for its health check, then
  restart API and workers with `docker compose restart api agent-worker`.
  Redis persistence uses the named `redis-data` volume; do not remove it during
  recovery.

## 2. Agent unavailable or slow

- **Symptom:** runs remain `processing` briefly and then become `failed`, with
  no confident recommendation.
- **Confirm:** inspect the run's `error` and `trace`, then check
  `docker compose ps` and `docker compose logs intake-agent coverage-agent
  agent-worker`.
- **Recover:** restore the unhealthy agent and wait for its health check, then
  resubmit the FNOL with the original idempotency key if one was used. A failed
  run is not silently converted to a coverage answer.

## Incident log

No genuine development incident has been recorded yet. Add a dated entry only
after reproducing and resolving a real failure; do not populate this log with a
fabricated incident.
