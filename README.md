# Durable Agent Platform

A TypeScript modular monolith whose only control plane is a PostgreSQL-persisted run state machine processed by leased workers. Providers and the native execution engine can propose actions; only the canonical runtime changes lifecycle state, invokes tools, accounts usage, or completes runs.

## Requirements

- Node.js 24 LTS and its bundled npm
- Docker with Compose for the verified local PostgreSQL 16/pgvector path

## Demonstrated scope

Stages 0–15 are implemented and tested with PostgreSQL 16. The native engine, provider-neutral contract, attempt ledgers, explicit retry/fallback, budgets, typed tools, durable approvals, exact resumption, crash classification, leased reconciliation, signed JWT authentication, tenant membership, API-path RLS, validated plans, budgeted context, durable summaries/checkpoints, deliberate tenant-scoped memory, tenant-isolated vector retrieval, governed skills, controlled stateless HTTP MCP connectors, persisted supervisor/child-run orchestration, durable evaluation history, redacted structured logs, correlation IDs, operational health/metrics, migration-gated startup, bounded process draining, and a real API-backed product shell are automated. Real `llama3.1:8b` model runs were manually verified through Ollama; Stage 9 also used real `all-minilm` embeddings and persisted pgvector search. Stages 10–15 use deterministic automated integrations rather than third-party MCP, real-model multi-agent, paid model-evaluation verification, an external telemetry backend, a production load environment, or browser automation. Anthropic, Gemini, and OpenAI-compatible adapters compile and have raw-response schema tests but are not credential-backed or manually verified here. There is no external framework engine, Kubernetes deployment, built-in identity issuer, or production-readiness claim.

## PostgreSQL deployment

The application uses only `DATABASE_URL`; it has no Docker-specific database code. Supported configurations are host processes with Compose PostgreSQL, host processes with host/remote/managed PostgreSQL, or containerized processes with containerized/external PostgreSQL. PostgreSQL 16 is verified; 15+ is the compatibility target. Migrations require `pgcrypto` and, from Stage 9, the `vector` extension. The supplied Compose image includes pgvector; external services must make a compatible extension available.

### Compose database

```powershell
npm ci
docker compose up -d postgres
$env:DATABASE_URL='postgres://agent:agent@localhost:5432/agent_platform'
npm run db:migrate
```

### Non-Docker database

Create an empty PostgreSQL database and a migration-capable role, set `DATABASE_URL` to that server (including TLS parameters when required), then run `npm run db:migrate`. This connection-string portability is implemented; a second non-Docker instance was not manually verified in this gate.

## Local Ollama

Install Ollama, pull an allowed model, copy `.env.example`, and explicitly configure the model’s capabilities. Do not claim tool calling for a model unless verified.

```powershell
ollama pull llama3.1:8b
$env:OLLAMA_MODELS='llama3.1:8b'
$env:OLLAMA_MODEL_CAPABILITIES_JSON='{"llama3.1:8b":{"toolCalling":true,"structuredOutput":true,"contextWindow":131072}}'
npm run verify:ollama
```

For retrieval, pull and explicitly configure a real embedding model and its output dimension. Retrieval remains disabled when `OLLAMA_EMBEDDING_MODEL` is empty; there is no fake fallback.

```powershell
ollama pull all-minilm
$env:OLLAMA_EMBEDDING_MODEL='all-minilm'
$env:OLLAMA_EMBEDDING_DIMENSIONS='384'
npm run verify:rag
```

## Optional classical ML

ML V1 supports versioned datasets/pipelines, durable training and evaluation, reproducible experiment records, explicit registration/promotion, version-pinned inference and regression comparison. It reuses the leased runtime and stores bounded artifacts in PostgreSQL. ML is disabled by default; see [ML setup and APIs](docs/ml.md) for the isolated scikit-learn backend.

## Controlled connectors

MCP connectors are disabled unless `CONNECTOR_EGRESS_ALLOWLIST` contains exact HTTP(S) origins. Registrations store only environment-variable credential references, never secret values. Discovery validates bounded JSON Schema 2020-12 tool schemas, while descriptions, server instructions, and annotations remain untrusted. Agents opt in to connector IDs and the canonical `mcp_call` tool; every call requires human approval and resumes through the existing durable `ToolExecutor`.

See `docs/connectors-skills.md` before enabling private-network access. This stage was tested against a local synthetic current-protocol server, not manually against a third-party MCP deployment.

## Multi-agent orchestration

An explicitly configured native supervisor may propose the canonical `delegate_run` tool. The tool creates ordinary tenant-scoped child runs with roles, context scope, and reserved token/cost budgets. Parents pause durably while children proceed through the same queue, lease, approval, cancellation, tool, and completion paths. Required failed or empty children prevent parent success. See `docs/orchestration.md`.

## Evaluation

Versioned evaluation suites queue ordinary canonical runs and asynchronously score persisted terminal evidence for task success, tool/argument behavior, retrieval/citations, approvals, safety markers, recovery evidence, latency, tokens, cost, and steps. Agent configurations are frozen per run so later edits cannot alter candidate provenance. Deterministic CI mode is test-harness-only; model-dependent runs use explicitly configured providers and are not part of unattended CI. See `docs/evaluation.md`.

## Run

The API uses the real JWT authentication path by default. The repository includes an explicit local-only helper that provisions a development membership in loopback PostgreSQL and signs an eight-hour token with secret material you generate. It refuses `NODE_ENV=production` and non-loopback database URLs.

In a PowerShell terminal, start PostgreSQL, configure the process environment, migrate, and mint a local token:

```powershell
npm ci
docker compose up -d postgres
$env:DATABASE_URL='postgres://agent:agent@localhost:5432/agent_platform'
$env:AUTH_MODE='jwt'
$env:JWT_SECRET = node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"
npm run db:migrate
$env:DEV_JWT = npm run --silent dev:identity
$env:DEV_JWT
npm run dev:api
```

Copy the displayed token before starting the API. In a second terminal, start the worker:

```powershell
$env:DATABASE_URL='postgres://agent:agent@localhost:5432/agent_platform'
npm run dev:worker
```

In a third terminal, paste the token and make an authenticated request:

```powershell
$env:DEV_JWT='<paste the token emitted by dev:identity>'
Invoke-RestMethod http://localhost:3000/agents -Headers @{ Authorization = "Bearer $env:DEV_JWT" }
```

Open `http://localhost:3000/` and paste the same token into the control surface. The helper defaults to tenant `11111111-1111-4111-8111-111111111111` and user `22222222-2222-4222-8222-222222222222`; override them with `DEV_TENANT_ID` and `DEV_USER_ID` if needed. This workflow provisions local development identity only—it does not add a login or token-issuance endpoint.

The verified cross-stage status, security and performance limits, commit map, and interview-safe description are recorded in `docs/final-assessment.md`.

The API defaults to HS256 JWT authentication. Tokens must have the configured issuer and audience, a valid expiry, `sub`, `jti`, `tenant_id`, `identity_type`, and role claims. The subject must also have an active database membership; effective roles are the intersection of signed and persisted roles. `.env.example` deliberately leaves `JWT_SECRET` empty, and startup fails until valid secret material is supplied. Development headers are available only with explicit `AUTH_MODE=development_headers` and are rejected when `NODE_ENV=production`; the JWT workflow above exercises the production authentication model instead.

## Verify

```powershell
npm run build
npm run test:unit
npm run test:integration
npm test
```

See `docs/architecture.md`, `docs/providers.md`, `docs/execution-engines.md`, `docs/evaluation.md`, `docs/configuration.md`, `docs/stage-gates.md`, and `docs/limitations.md`.
