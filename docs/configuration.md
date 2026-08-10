# Configuration

Node.js 24 LTS is the supported host runtime. `DATABASE_URL` is required. API, worker, and migrations use it directly and never infer a Docker hostname.

Core settings: `PORT`, `WORKER_ID`, `LEASE_SECONDS`, `WORKER_POLL_MS`, and comma-separated `WORKER_ROLES`.

API authentication defaults to `AUTH_MODE=jwt`. `JWT_SECRET` is required and must contain at least 32 characters; `.env.example` intentionally leaves it empty so an example value cannot satisfy startup validation. For local development, generate independent secret material with `node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"`. `JWT_ISSUER` and `JWT_AUDIENCE` default to `durable-agent-platform` and `durable-agent-api`. Tokens use HS256 and require expiry, subject, token ID, tenant, identity type, and roles. `AUTH_MODE=development_headers` accepts UUID development headers only when `NODE_ENV` is not `production`; it is intended for tests and isolated local debugging, never deployment.

Tenant membership is stored in `tenant_memberships`. Revocation inserts the token issuer/JTI and its expiry into `revoked_tokens`; expired rows may be removed by routine database maintenance. The platform validates tokens but does not provide user/password login or a production token issuer. `npm run dev:identity` is an explicit development-only bootstrap: it refuses production mode and non-loopback PostgreSQL, upserts one local user membership, and emits an eight-hour JWT using the configured secret, issuer, and audience. Its default UUIDs and role set are documented in README. Operators remain responsible for controlled production identity provisioning, token issuance, secret storage, and rotation; rotating the HMAC secret invalidates existing tokens.

Ollama settings: `OLLAMA_ENABLED`, `OLLAMA_BASE_URL`, comma-separated `OLLAMA_MODELS`, and `OLLAMA_MODEL_CAPABILITIES_JSON`. An empty model list registers no Ollama provider. Capability JSON is keyed by exact model ID.

Container deployments pass the same provider enablement, model, capability, endpoint, and credential-reference environment to both API and worker processes. This lets the API expose the configured selection surface while workers resolve the identical provider IDs. A process with a different provider configuration is a deployment error; PostgreSQL does not store provider secrets.

Optional OpenAI-compatible settings: `OPENAI_COMPAT_ENABLED`, `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODELS`, `OPENAI_COMPAT_API_KEY`, and `OPENAI_COMPAT_MODEL_CAPABILITIES_JSON`.

Optional cloud settings: `ANTHROPIC_ENABLED`, `ANTHROPIC_MODELS`, `ANTHROPIC_MODEL_CAPABILITIES_JSON`, `ANTHROPIC_API_KEY`; and equivalent `GEMINI_*` values. Enabling a cloud provider without its key is a clear configuration error. Keys must come from the process environment or a future secret resolver; never place them in committed files.

Agent model targets persist provider/model IDs, output-token/timeout limits, and integer micro-USD-per-million-token rates. Zero rates are valid for explicitly configured local models. Incorrect cloud rates make cost estimates incorrect, so cloud deployment must supply reviewed pricing metadata in the agent definition.

Model `retryPolicy` sets bounded `maxAttempts`, `baseDelayMs`, and `maxDelayMs`. Retries occur only for normalized retryable provider failures. `fallbacks` is an ordered, explicit list of complete provider/model/cost targets; absent configuration means no fallback. Every fallback must be registered and satisfy the current turn's required capabilities.

Agent `approvalPolicy` is keyed by tool name. `true` requires the default `approval_reviewer` role with requester/approver separation. An object may set `requiredApproverRole`, `separationOfDuties`, and a public-safe `riskExplanation`. A false or absent entry does not override a tool whose definition always requires approval. Workers must retain the tool's execution role in `WORKER_ROLES`; human approvers receive only approval roles.

Agent `contextPolicy` controls `maxInputTokens`, `recentSteps`, and `summaryTargetTokens`. The runtime also honors a known provider model context window and reserves the largest configured candidate output allowance. These values use a documented deterministic estimate rather than provider-specific tokenization. Set composition `planner` to `native` to allow validated plan revisions; it remains `disabled` by default.

Agent `memoryPolicy` defaults fully disabled. `writeEnabled` requires at least one `allowedScopes` value (`user` or `agent`) and a positive `maxWritesPerRun`; model writes still require `memory_store`, the `memory_write` worker role, and human approval. `retrievalEnabled`, `maxContextItems`, and `maxContextTokens` bound memory context selection. Any enabled memory behavior requires composition `memory: native`. Tenant-scoped memory is intentionally restricted to authenticated `memory_manager` API writes rather than model writes.

Document retrieval is disabled unless `OLLAMA_EMBEDDING_MODEL` is non-empty. `OLLAMA_EMBEDDING_BASE_URL` defaults to the local Ollama address, `OLLAMA_EMBEDDING_DIMENSIONS` is mandatory when enabled, and `OLLAMA_EMBEDDING_TIMEOUT_MS` bounds requests. The configured dimension must exactly match the model output; mismatches fail ingestion/search. There is no fake embedding or fallback provider. API and worker processes must receive identical retrieval configuration.

Agent `retrievalPolicy` defaults disabled. Enabling it requires composition `retriever: native` and configures `maxContextChunks`, `maxContextTokens`, and `minimumScore`; a native retriever selected without an enabled policy is rejected. Agents using explicit retrieval calls must also allowlist `knowledge_search`. Tenant-visible document management requires the persisted `document_manager` role.

Connectors remain disabled when `CONNECTOR_EGRESS_ALLOWLIST` is empty. When enabled, it is a comma-separated set of exact origins such as `https://mcp.example.invalid`; paths belong to connector registrations, not the origin allowlist. `CONNECTOR_ALLOW_PRIVATE_NETWORK=false` is the safe default. Set it to `true` only for deliberately reviewed local/private MCP servers. `CONNECTOR_TIMEOUT_MS` and `CONNECTOR_MAX_RESPONSE_BYTES` bound transport calls and responses. Both API and worker processes need the same allowlist and transport settings.

Connector `credentialRef` values must match `CONNECTOR_SECRET_[A-Z0-9_]+`. The referenced environment variable contains the actual bearer credential. Only the reference name is persisted. Supply each secret explicitly to the API and worker secret environment; do not add values to Compose files or committed configuration.

Agent `connectorPolicy` defaults disabled. Enabling it requires composition `connectors: native`, at least one tenant connector ID, and `mcp_call` in `allowedTools`. `maxContextTools` bounds the untrusted discovered catalog. Workers executing connector proposals need every connector's configured roles, normally including `connector_execute`; human decisions still require the approval role.

Agent `skillPolicy` defaults disabled. Enabling it requires composition `skills: native`, explicit current skill-version IDs, and a separate context-token budget. Skills cannot name a tool absent from the agent allowlist. Management requires `skill_manager`; connector registration/discovery/revocation requires `connector_manager`, inspection may use `connector_auditor`, and execution uses configured connector roles.

Agent `orchestrationPolicy` defaults disabled. Enabling it requires composition `orchestrator: native`, `delegate_run` in `allowedTools`, explicit child-agent and role allowlists, positive child/fan-out/depth limits, and per-child token/cost ceilings. `allowSharedContext` is an explicit opt-in; otherwise children receive no parent-step snapshot. A delegated budget cannot exceed the target agent budget, orchestration ceiling, or remaining parent run budget. `maximumParallel` bounds active direct children rather than worker process count.

Evaluation suite mutation and execution require `evaluation_manager`; inspection accepts `evaluation_manager` or `evaluation_viewer`. No provider credential is stored in evaluation records. Production/API construction supports only `model_dependent` execution. `deterministic_ci` is an explicit test-harness mode and cannot be enabled through environment configuration or used as a production fallback.

Operational inspection requires the persisted and signed `system_operator` role. Worker IDs should be stable per process replica and must not contain secrets. Provider-health results reflect only providers configured in the API process. Stage 13 does not require a vendor telemetry endpoint or credential.

`DB_POOL_MAX`, `DB_POOL_IDLE_TIMEOUT_MS`, and `DB_CONNECT_TIMEOUT_MS` bound each process pool and connection wait. Size aggregate connections across API and worker replicas below the database limit with migration/operations headroom. `SHUTDOWN_GRACE_MS` bounds API HTTP draining. Readiness fails when a repository migration is absent from `schema_migrations`; `npm run db:check` performs the same read-only deployment check.

The product shell needs no separate configuration. It submits a bearer JWT to the same origin and discovers configured provider IDs and tool metadata from the API. `tenant_admin` manages memberships; `usage_viewer` and `audit_viewer` grant inspection without management authority. Initial administrator and token issuance must be provisioned externally.
