# Execution engines and composition

The platform implements only the `native` bounded-turn engine. It receives canonical context/tool definitions, invokes one configured provider, and returns one validated canonical action: `call_tool`, `final_answer`, `request_clarification`, `update_plan`, or `pause`. At Stage 7, `update_plan` is enabled only when the agent explicitly selects the native planner; durable validation and persistence remain outside the engine.

Composition is persisted with each agent and currently accepts only:

```json
{
  "executionEngine": "native",
  "contextBuilder": "native",
  "outputParser": "native",
  "planner": "disabled",
  "retriever": "disabled",
  "memory": "disabled",
  "connectors": "disabled",
  "skills": "disabled",
  "orchestrator": "disabled"
}
```

`memory` may be set to `native` when a validated memory policy enables writes or retrieval. The memory service remains a bounded component: model writes pass through approval and the canonical tool executor, while context selection only reads authorized persisted records.

`retriever` may be set to `native` only with an enabled validated retrieval policy and a configured real embedding provider. The native context builder uses the same tenant/visibility-aware retrieval service as the canonical `knowledge_search` tool, persists selected chunk IDs, and treats returned documents as untrusted evidence. It remains a component inside the native engine path, not an execution engine or lifecycle owner.

`orchestrator` may be set to `native` only with a validated orchestration policy and the canonical `delegate_run` tool. The engine still performs one bounded model turn; delegation, child creation, waiting, requeueing, cancellation propagation, budgets, and completion checks remain runtime/database responsibilities. There is no framework-owned supervisor engine.

Unsupported engines or hybrid components fail validation. No LangChain or other framework dependency/fallback is installed. Future adapters may provide bounded planning, retrieval, context, memory ranking, summarization, parsing, or model turns, but never lifecycle, leases, transition policy, approval, tool authorization/execution, idempotency, cancellation, budgets, audit, tenancy, or completion.
