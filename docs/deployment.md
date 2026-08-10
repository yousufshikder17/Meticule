# Database and process deployment

## Host processes with Compose PostgreSQL

With Node.js 24 LTS installed, run `docker compose up -d postgres`, point host processes at `postgres://agent:agent@localhost:5432/agent_platform`, apply migrations, then run the API and workers with npm scripts. The Compose service uses a PostgreSQL 16 image with pgvector because the initial schema requires the `vector` extension.

## Host processes with non-Docker PostgreSQL

Provision PostgreSQL 15+ on the host, another server, or a managed service with compatible `pgcrypto` and `vector` extensions. The migration role must initially be able to create or use those extensions plus schemas, tables, types, functions, triggers, and restricted roles. Some managed services require an administrator to enable pgvector first. Set the service-provided `DATABASE_URL` (including TLS options) and run the same `npm run db:migrate`. A reduced-privilege migration/runtime role split is recommended but not automated.

## Containerized processes

`docker compose --profile app up --build` starts PostgreSQL, runs the one-shot migration container, then starts API and worker containers after migration success. For external PostgreSQL, build the image and run migration/API/worker containers with the external `DATABASE_URL`; no application code or migration changes are required. Compose's default `postgres` hostname is only a deployment default passed through `DATABASE_URL`, never hard-coded by the application.

Kubernetes is not required. If introduced later, deployments scale worker pools; they do not represent logical agents. PostgreSQL remains lifecycle authority.

Workers persist heartbeats and mark drain intent on graceful shutdown. Operational checks should alert on stale worker heartbeats, expired leases, growing queue depth, failure-category changes, and provider-health degradation. Export JSON logs without raw prompts or credentials and apply the retention guidance in `observability.md`.

Retrieval additionally requires an Ollama embedding endpoint reachable from both API and worker processes. For host processes the default is `http://localhost:11434`; the Compose application profile uses `http://host.docker.internal:11434` unless overridden. Pull an embedding model separately, set its exact dimension, and treat any remote embedding endpoint as a data processor because document/query text leaves the application process. Only connection-string portability was verified outside Compose; a second non-Docker pgvector installation was not live-tested.

MCP connector processes are optional and external to the platform. Configure identical exact origins in `CONNECTOR_EGRESS_ALLOWLIST` for API discovery/management and worker execution. Compose does not inject connector credentials: provide referenced `CONNECTOR_SECRET_*` values through the deployment secret mechanism. Private-network destinations are blocked unless explicitly enabled. Container deployments must use an origin reachable from the container; `localhost` inside a container is not the host. No third-party MCP server was manually verified in Stage 10.

## Graceful shutdown and pressure

The API changes readiness to `503`, stops accepting connections, closes idle connections, and waits up to `SHUTDOWN_GRACE_MS` before forcing remaining HTTP connections closed. A worker marks its heartbeat draining on signal, finishes its current bounded scheduler phase, and does not enter another. If the container deadline kills active external work, lease recovery and unknown-effect reconciliation apply; JavaScript shutdown cannot revoke an issued side effect.

Configure `DB_POOL_MAX` per replica. API pools, worker pools, migrations, administration, and headroom must fit the PostgreSQL connection limit. Scale workers only while queue latency and database/provider/tool capacity justify it. Stage 14 tested concurrent claims over 12 queued runs but did not establish production throughput.

## Migration and rollback policy

The initial release supports bootstrap from an empty PostgreSQL/pgvector database through `001_initial.sql`. Run migrations separately with a migration-capable identity; application replicas should use runtime identities. `npm run db:check` must pass before admission. Future migrations are forward-only and should remain additive while old replicas can run. Do not automatically reverse schema after application rollback. Roll code back only when compatible with the applied schema; otherwise deploy a forward corrective migration. Destructive or large data changes require separate review and a verified backup.

## Backup, restore, and PITR

Use physical backups and continuous WAL archiving for point-in-time recovery, plus logical exports when portability is needed. Encrypt backups, isolate credentials, define recovery/retention objectives, and restore periodically into a separate database. A restore drill must run migrations/readiness and tenant-isolation smoke tests before promotion. The Compose volume is development persistence, not backup. This repository does not automate backup, WAL archiving, PITR, or restore drills.

## CI and cloud-neutral operation

CI provisions pgvector PostgreSQL, installs from the lockfile, migrates and checks drift, builds, runs the complete deterministic suite, audits production dependencies, inspects package contents, builds the image, and scans working files plus history with pinned Gitleaks. Paid providers are excluded. Registries, signing/attestation, cloud vulnerability scanners, and optional Kubernetes worker pools may be added without changing the control plane.
