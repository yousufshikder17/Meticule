# Runtime observability and operations

PostgreSQL remains the evidence source. `GET /runs/:id/trace` assembles canonical run, step, model-attempt, tool-attempt, approval, reconciliation, context, orchestration, usage, and audit records; logs and metrics do not replace those ledgers.

The API assigns a new correlation UUID to every request, returns it as `x-correlation-id`, and emits a completion event through the structured logger. Workers emit claimed/released run events with run, tenant, worker, and elapsed-time fields. The shared logger serializes JSON, bounds nested values, redacts sensitive key names, and removes bearer values. It must not receive prompts, raw provider responses, credentials, or unrestricted tool payloads.

`system_operator` is required for operational endpoints:

- `GET /operations/metrics` returns tenant-scoped queue depth, lease health, 24-hour usage/cost, bounded failure categories, and persisted worker-heartbeat health.
- `GET /operations/providers/health` probes only providers configured in that API process and returns redacted status and latency. A healthy transport does not establish model quality or authorization.

Workers upsert a process heartbeat at startup and each scheduler pass, and mark themselves draining during graceful shutdown. A heartbeat older than 90 seconds is reported unhealthy. PostgreSQL readiness remains `GET /health/ready`; process liveness remains `GET /health/live`.

Operational rows contain only instance identity, kind, timing, drain state, and bounded non-secret metadata. Run telemetry remains tenant-scoped under repository predicates and API RLS. Operators should export structured logs and aggregate metrics to a deployment-selected backend; this repository intentionally has no vendor-specific collector.

Retention must be selected according to data classification. Keep audit, approval, and idempotency evidence longer than diagnostic logs; never delete unresolved tool outcomes or active lifecycle evidence. Before purging completed traces, account for tenant policy, investigations, evaluation provenance, backup retention, and referential constraints. Automated retention and purge jobs are not implemented in Stage 13.
