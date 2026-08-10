# Implemented API

Except for health checks, endpoints require `Authorization: Bearer <JWT>`. JWT issuer, audience, algorithm, expiry, subject, token ID, tenant, identity type, persisted membership, revocation, and effective roles are validated. A signed tenant claim cannot access another tenant's rows. Explicit non-production header mode exists for local tests only.

The public `GET /`, `/app.css`, and `/app.js` routes contain the same-origin product shell and no tenant data. `GET /me`, `GET /providers`, `GET /agents`, `GET /runs`, `GET /approvals`, `GET /usage`, `GET /audit`, and membership list/create/revoke support that shell. Usage requires `usage_viewer` or `tenant_admin`; audit requires `audit_viewer` or `tenant_admin`; membership administration requires `tenant_admin`, applies RLS, and prevents self-revocation. Existing resource-specific authorization remains authoritative.

- `POST /agents`; `GET /agents/:id`; `PATCH /agents/:id`
- `POST /agents/:id/runs`; `GET /runs/:id`
- `POST /runs/:id/cancel`; `POST /runs/:id/resume`
- `GET /runs/:id/steps`; `GET /runs/:id/trace`; `GET /runs/:id/tree`; `GET /runs/:id/approvals`
- `GET /approvals/:id`; `POST /approvals/:id/approve`; `POST /approvals/:id/reject`
- `GET /memories`; `POST /memories`; `DELETE /memories/:id`
- `GET /memories/:id/provenance`; `POST /memories/:id/corrections`
- `GET /documents`; `POST /documents`; `DELETE /documents/:id`
- `GET /documents/:id/versions`; `POST /documents/:id/versions`
- `POST /retrieval/search`
- `GET /connectors`; `POST /connectors`; `GET /connectors/:id`
- `GET /connectors/:id/tools`; `POST /connectors/:id/discover`; `POST /connectors/:id/health`; `POST /connectors/:id/revoke`
- `GET /skills`; `POST /skills`; `GET /skills/:id/versions`
- `POST /skills/:id/versions`; `POST /skills/:id/revoke`
- `GET|POST /evaluation-suites`; `GET /evaluation-suites/:id`; `POST /evaluation-suites/:id/versions`
- `POST /evaluation-suites/:id/executions`; `GET /evaluations`; `GET /evaluations/:id`; `GET /evaluations/:id/compare/:otherId`

Approval decision bodies accept an optional `comment`. The caller must have the persisted required role, match the approval tenant, and differ from the requester when separation of duties is enabled. Repeating the same decision is idempotent; a conflicting decision returns `409`. Decision endpoints only persist and requeue—they never invoke tools inline. `POST /runs/:id/resume` applies only to ordinary paused runs and returns `409` while an approval, child-run wait, or unknown external tool effect remains unresolved. Manual reconciliation remains unresolved until an explicit final outcome is persisted.

Memory creation is deliberate. Ordinary users may create/list/delete/correct their own user-scoped records. Agent/tenant-scoped creation and management require `memory_manager`; list queries accept `agentId`, `memoryType`, `q`, and bounded `limit`. Provenance inspection includes correction ancestry. Deletion is soft and deleted/expired records are not retrieved.

Document bodies accept bounded plain text, Markdown, or JSON content, explicit private/tenant visibility, an optional source URI, and bounded scalar metadata. Requests above 2.2 MB are rejected. Tenant-visible document creation requires `document_manager`; private documents remain owner-only. Version creation makes a new immutable source version current only after every embedding and chunk persists successfully. Deletion soft-deletes all versions and chunks. Search accepts a query, exact metadata filters, result/token limits, and minimum score, then returns citation-bearing chunks. Document and search endpoints return `503 retrieval_not_configured` when no real embedding provider is configured.

Connector management requires `connector_manager`; inspection accepts `connector_manager` or `connector_auditor`. Registration accepts an exact allowlisted endpoint, environment credential-reference name, tool allowlist, execution roles, and per-minute rate. Discovery and health perform network policy checks; discovered schemas and metadata are inspection data, not trusted policy. Revocation disables the persisted catalog. Connector endpoints return `503 connector_not_configured` when no process egress allowlist exists. They never expose resolved secret values or execute a discovered tool.

Skill creation, revision, and revocation require `skill_manager`. Each version includes explicit provenance and a content hash. Listing returns only current non-revoked versions; version history remains inspectable. Skills do not expand agent or connector authorization.

`GET /runs/:id/tree` returns the tenant-scoped root hierarchy and delegation edges without exposing shared-context snapshots. Agents, not API callers, create child runs through the governed `delegate_run` tool. A parent waiting on children cannot be manually resumed; cancellation propagates to its non-terminal descendants.

Evaluation mutation requires `evaluation_manager`; inspection also accepts `evaluation_viewer`. Suites are immutable versions. Starting an execution snapshots the selected agent and queues one ordinary run per case; it never invokes models inline. `GET /evaluations` accepts optional `suiteId`, `agentId`, `mode`, and bounded `limit` filters. Candidate comparison requires two completed executions from the same tenant and exact suite version. The production API rejects test-only deterministic mode.

`GET /tools` returns non-handler tool metadata, including `delegate_run`, approval-only `memory_store`, configured read-only `knowledge_search`, and configured approval-only `mcp_call`. `GET /runs/:id/trace` includes the run, steps, redacted model attempts, tool executions and attempts, connector invocation metadata, approval metadata, plan revisions, summaries, checkpoints, direct delegations/waits, context-build provenance (including selected memory, document-chunk, skill, connector-tool, and child-run IDs), and audit events. `GET /health/live` is process liveness; `GET /health/ready` checks PostgreSQL, applied migrations, and API drain state, returning `503` before migrations or during graceful shutdown. Provider configuration, reconciliation, orchestration scheduling, planning execution, and run execution remain worker-owned, not inline API behavior.

`GET /operations/metrics` and `GET /operations/providers/health` require `system_operator`. Metrics are tenant-scoped for runs, leases, usage, cost, and failures while worker process health is deployment-wide. Provider health checks adapters configured in the API process. Every response includes a generated `x-correlation-id`; callers cannot inject the logged identifier.
