# Durable multi-agent orchestration

Stage 11 adds supervisor/worker behavior without adding another runtime. A supervisor model can only propose `delegate_run`; the existing canonical `ToolExecutor` validates, authorizes, records, invokes, and reconciles that keyed side effect. Each child is an ordinary `runs` row with the same state machine, PostgreSQL lease, steps, approvals, cancellation, budgets, trace, and terminal rules as a root run.

## Configuration and authority

Orchestration is disabled by default. Enabling it requires composition `orchestrator: native`, `delegate_run` in the agent tool allowlist, explicit child-agent and role allowlists, maximum children, maximum parallel active children, maximum depth, per-child token/cost ceilings, and an explicit shared-context decision. Delegation also checks the target agent's own budget and the parent run's remaining budget. A target already present in the ancestor agent chain is rejected.

`delegate_run` input freezes the target agent, bounded child goal, role, required flag, private/shared context scope, and child budgets. Its idempotency identity is bound by the canonical tool ledger; a completed duplicate returns the persisted child identifier, while changed content conflicts. The tool is reconcilable: a crash after child creation but before tool-result persistence resolves by looking up the persisted delegation rather than creating another child.

## Waiting and resumption

After creating immediately ready children, a supervisor returns `pause` with the exact reason `wait_for_children`. The runtime atomically persists an `orchestration_waits` row, transitions the parent to `paused`, releases its lease, and leaves children claimable by ordinary workers. A scheduler path uses row locking to resolve waits only after every direct child is terminal, then requeues the parent for a normal lease. It never executes a child or a parent inline.

On the resumed turn, required children must be completed with non-empty output. Failed, cancelled, or empty required output fails the parent before another provider call. Application checks and a PostgreSQL completion trigger prevent premature success. Optional child failures remain visible to the supervisor for explicit synthesis policy.

Private child context includes only the delegated goal and relationship metadata. Shared context additionally receives a bounded immutable snapshot of the parent's goal and recent persisted terminal steps. Parent context receives bounded persisted child states/results; selected child IDs are recorded on each context build. Context is data, not lifecycle authority.

Cancellation of a parent propagates durably through all non-terminal descendants and intervenes in descendant approvals. A cancelled child is never executed later. Approval remains local to the child run and uses the existing role separation and exact frozen resumption path. Parent/child trace hierarchy is exposed at `GET /runs/:id/tree`; ordinary run traces include direct delegations and orchestration waits.

## Deliberate limitations

This stage provides bounded child-run orchestration, not a distributed workflow language. It has no dynamic worker placement, affinity, work stealing, cross-tenant delegation, child result streaming, Kubernetes-per-agent deployment, framework-owned supervisor, or real-model multi-agent manual verification. Parallelism means multiple independently leased child runs may be processed by multiple worker processes or containers; a single worker loop remains sequential.
