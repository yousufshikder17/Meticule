# Controlled MCP connectors and governed skills

## Security boundary

Connectors and skills are bounded components of the native runtime. They do not own runs, leases, completion, approvals, budgets, or cancellation. An MCP server cannot register a direct executable handler: all external tool proposals use the built-in `mcp_call` definition and therefore pass through canonical agent allowlists, connector policy, worker roles, human approval, exact frozen resumption, `ToolExecutor`, idempotency, and audit evidence.

External server instructions, tool descriptions, annotations, and results are untrusted data. In particular, `readOnlyHint`, `destructiveHint`, and `idempotentHint` never reduce the platform's high-risk, approval-always, non-retryable classification. Unknown post-dispatch outcomes pause for manual reconciliation and are not automatically called again.

## Protocol support

Stage 10 implements the current stateless MCP HTTP request model used by protocol version `2026-07-28`: `server/discover`, paginated `tools/list`, and `tools/call` over JSON-RPC responses in JSON or bounded SSE. Requests send explicit protocol/method/name headers and per-request client metadata. Discovery requires the server to advertise tool capability and the configured protocol version.

Tool input/output definitions are treated as untrusted JSON Schema 2020-12. The implementation bounds serialized size, depth, node count, tools, and pages; rejects `$ref`/`$dynamicRef`; requires object input; and compiles schemas locally. Tools requiring the unsupported tasks extension cannot be enabled.

Legacy stateful/session transports, stdio servers, resources, prompts, server-initiated sampling or elicitation, OAuth negotiation, task execution, streaming progress, and third-party SDK hosting are unsupported. A configured connector that needs one of these features fails explicitly.

## Registration and execution

1. An operator configures exact origins in `CONNECTOR_EGRESS_ALLOWLIST`. Empty configuration disables connector APIs and the runtime tool.
2. A `connector_manager` registers a tenant endpoint, a `CONNECTOR_SECRET_*` reference, explicit tool names, execution roles, and a per-minute rate.
3. Discovery validates the endpoint and schemas, then atomically replaces the tenant catalog and audit evidence. A missing allowlisted tool fails discovery.
4. An agent explicitly enables native connectors, selects connector IDs, and allowlists `mcp_call`.
5. Context contains only a bounded catalog labeled as untrusted. The model may propose `mcp_call` with connector ID, tool name, and arguments.
6. The runtime validates and authorizes the proposal, freezes it for role-separated approval, and releases the lease.
7. A later worker rechecks connector state, tenant, roles, agent policy, tool allowlist, discovered schema, frozen hash, idempotency, and cancellation before the canonical executor dispatches the request.

Connector decisions never invoke the server inline from the HTTP API. Revocation disables catalog entries and blocks future/resuming authorization. Durable PostgreSQL counters enforce the configured connector rate across processes. Credential values are resolved only at discovery/health/call time and are not stored in connector, invocation, tool, context, or audit rows.

## Egress limitations

The network policy requires HTTP(S), rejects userinfo and fragments, uses exact origins, refuses redirects, screens resolved addresses for private/special ranges by default, and bounds time and response bytes. Private networks require explicit opt-in for local development.

This is not a network sandbox. The current transport does not pin its preflight DNS result into the fetch socket, so a DNS answer can theoretically change between policy check and connection. Use only reviewed operator-controlled origins and place production egress behind a network proxy/firewall that independently enforces DNS/IP policy. Stage 10 was automated against a local synthetic server; interoperability with a third-party MCP deployment was not manually verified.

## Skills

Skills are versioned operator-authored instruction records, not downloaded executables. Each includes tenant ownership, provenance source, creation reason, optional public source URI, author, content hash, version ancestry, and revocation evidence. `skill_manager` controls creation/revision/revocation. Agents select exact current versions with a token budget. A selected skill may reference only tools already in the agent allowlist and cannot bypass runtime policy.

Skill text is intentionally inserted as operator-authorized system guidance. Memory, retrieved documents, and MCP metadata remain separately labeled untrusted data. No remote skill installation, executable code, package loading, filesystem access, or automatic model-authored skill persistence is implemented.
