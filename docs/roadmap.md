# Staged implementation plan

Stages follow the active implementation order. A later stage begins only after its predecessor has an executable GO record in `stage-gates.md`.

0. Architecture and repository foundation (implemented).
1. PostgreSQL lifecycle, leases, cancellation, and recovery (implemented).
2. Typed deterministic tools and idempotent execution (implemented).
3. Provider-neutral structured model loop with native engine (implemented).
4. Durable role-separated approval and exact resumption (implemented).
5. External-effect reconciliation, retry safety, and provider-attempt orchestration (implemented).
6. Authentication, authorization, tenant security, RLS, and secrets (implemented).
7. Context management, planning, summarization, and checkpoints (implemented).
8. Tenant-scoped durable memory (implemented).
9. Tenant-isolated retrieval and RAG (implemented).
10. Controlled MCP, skills, connectors, and external capabilities (implemented).
11. Durable multi-agent orchestration through child runs (implemented).
12. Evaluation and regression framework (implemented).
13. Observability and operational controls (implemented).
14. Production deployment and scale hardening (implemented).
15. Product, API, and UI completion (implemented).

The README and gate record, not this roadmap, state verified current implementation status.

The optional ML extension preserves its Phase 1–5 lifecycle and adds the bounded Phase 6 Public Core
adapter path: local LoRA and single-GPU NF4/FP4 QLoRA, safe checkpoints and version-pinned adapter inference.
See [ML scope and limits](ml.md#phase-6-public-core-local-adapters) and the [verification report](ml-port-report.md).
Distillation, RL/DPO, Ray orchestration and broader managed/enterprise ML capabilities are not included.
This selection does not change the staged agent roadmap above.
