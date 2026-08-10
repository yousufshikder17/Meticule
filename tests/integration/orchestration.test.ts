import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { OrchestrationService } from "../../src/orchestration/orchestration-service.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";
import { ModelProviderError } from "../../src/models/model-errors.js";
import { ToolExecutor } from "../../src/tools/executor.js";
import { ApprovalService } from "../../src/approvals/approval-service.js";
import { withTenantSession } from "../../src/db/tenant-session.js";
import { ReconciliationService } from "../../src/tools/reconciliation-service.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const principal: Principal = { tenantId: "a1000000-0000-4000-8000-000000000001", userId: "a1000000-0000-4000-8000-000000000002", roles: [] };

beforeEach(async () => pool.query("TRUNCATE structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

function base(provider: string, name: string) {
  return {
    name, systemInstructions: "Return one canonical action based only on persisted state.",
    model: { provider, model: "deterministic", maxOutputTokens: 100, timeoutMs: 1000, inputCostMicrousdPerMillion: 0, outputCostMicrousdPerMillion: 0, cachedCostMicrousdPerMillion: 0 },
    allowedTools: [] as string[], maximumSteps: 20, tokenBudget: 10_000, costBudgetMicrousd: 10_000, approvalPolicy: {}, outputSchema: null,
  };
}

async function runtime(providers: DeterministicTestProvider[], roles: string[] = principal.roles) {
  const registry = new ProviderRegistry(); providers.forEach((provider) => registry.register(provider));
  const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
  const tools = createToolRegistry(pool); const loop = new AgentLoop(pool, registry, engines, tools);
  return new LifecycleWorker(pool, { workerId: "orchestration-worker", leaseSeconds: 30 }, { execute: (id, workerId) => loop.execute(id, workerId, roles) });
}

describe("durable multi-agent orchestration", () => {
  it("delegates ordinary child runs, waits durably, and synthesizes only after persisted child results", async () => {
    const childOneProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: { finding: "one" } } }], {}, "child-one-provider");
    const childTwoProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: { finding: "two" } } }], {}, "child-two-provider");
    const agents = new AgentRepository(pool);
    const childOne = await agents.create(principal, base(childOneProvider.id, "Child One"));
    const childTwo = await agents.create(principal, base(childTwoProvider.id, "Child Two"));
    const supervisorProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "delegate-one", name: "delegate_run", arguments: { targetAgentId: childOne.id, goal: "produce finding one", roleName: "researcher", required: true, contextScope: "shared", tokenBudget: 1000, costBudgetMicrousd: 100 } }] },
      { toolCalls: [{ id: "delegate-two", name: "delegate_run", arguments: { targetAgentId: childTwo.id, goal: "produce finding two", roleName: "reviewer", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 100 } }] },
      { body: { type: "pause", reason: "wait_for_children" } },
      { body: { type: "final_answer", output: { synthesis: ["one", "two"] } } },
    ], {}, "supervisor-provider");
    const supervisor = await agents.create(principal, {
      ...base(supervisorProvider.id, "Supervisor"), allowedTools: ["delegate_run"],
      composition: { orchestrator: "native" },
      orchestrationPolicy: { enabled: true, allowedAgentIds: [childOne.id, childTwo.id], allowedRoles: ["researcher", "reviewer"], maximumChildren: 2, maximumParallel: 2, maximumDepth: 2, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: true },
    });
    const run = await new RunRepository(pool).create(principal, supervisor.id, "combine two independent findings");
    const worker = await runtime([supervisorProvider, childOneProvider, childTwoProvider]);
    expect(await worker.tick()).toBe(true);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("paused");
    expect(supervisorProvider.calls).toBe(3);
    expect((await pool.query("SELECT count(*)::int AS count FROM run_delegations WHERE parent_run_id=$1", [run.id])).rows[0].count).toBe(2);
    expect(await worker.tick()).toBe(true);
    expect(await worker.tick()).toBe(true);
    expect(supervisorProvider.calls).toBe(3);
    expect(await new OrchestrationService(pool).advanceNext("orchestration-scheduler")).toBe(true);
    expect(await worker.tick()).toBe(true);
    const completed = await new RunRepository(pool).get(principal.tenantId, run.id);
    expect(completed.status).toBe("completed"); expect(completed.finalOutput).toEqual({ synthesis: ["one", "two"] });
    expect(supervisorProvider.calls).toBe(4);
    expect(completed.reservedChildTokens).toBe(2000);
    const children = await pool.query("SELECT status,final_output,context_scope FROM runs WHERE parent_run_id=$1 ORDER BY delegation_role", [run.id]);
    expect(children.rows.every((child) => child.status === "completed")).toBe(true);
    const shared = await pool.query("SELECT shared_context FROM run_delegations WHERE parent_run_id=$1 AND context_scope='shared'", [run.id]);
    expect(shared.rows[0].shared_context.parentGoal).toBe("combine two independent findings");
    const trace = await new RunRepository(pool).trace(principal.tenantId, run.id);
    expect(trace.delegations).toHaveLength(2); expect(trace.orchestrationWaits[0].status).toBe("resolved");
    expect(trace.contextBuilds.at(-1)?.selected_child_run_ids).toHaveLength(2);
    await withTenantSession(pool, principal.tenantId, async (database) => {
      expect(Number((await database.query("SELECT count(*) AS count FROM run_delegations")).rows[0].count)).toBe(2);
      expect(Number((await database.query("SELECT count(*) AS count FROM orchestration_waits")).rows[0].count)).toBe(1);
    });
    await withTenantSession(pool, "a1000000-0000-4000-8000-000000000099", async (database) => {
      expect(Number((await database.query("SELECT count(*) AS count FROM run_delegations")).rows[0].count)).toBe(0);
      expect(Number((await database.query("SELECT count(*) AS count FROM orchestration_waits")).rows[0].count)).toBe(0);
    });
  });

  it("fails the parent instead of reporting success when a required child fails", async () => {
    const childProvider = new DeterministicTestProvider([{ error: new ModelProviderError("unavailable", "child unavailable", false) }], {}, "failed-child-provider");
    const agents = new AgentRepository(pool); const child = await agents.create(principal, base(childProvider.id, "Failing Child"));
    const supervisorProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "delegate-required", name: "delegate_run", arguments: { targetAgentId: child.id, goal: "required work", roleName: "worker", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 100 } }] },
      { body: { type: "pause", reason: "wait_for_children" } },
      { body: { type: "final_answer", output: "must not complete" } },
    ], {}, "failed-supervisor-provider");
    const supervisor = await agents.create(principal, { ...base(supervisorProvider.id, "Failure Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [child.id], allowedRoles: ["worker"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false } });
    const run = await new RunRepository(pool).create(principal, supervisor.id, "required child must succeed"); const worker = await runtime([supervisorProvider, childProvider]);
    expect(await worker.tick()).toBe(true);
    await expect(worker.tick()).rejects.toThrow("child unavailable");
    expect(await new OrchestrationService(pool).advanceNext("scheduler")).toBe(true);
    await expect(worker.tick()).rejects.toThrow(/Required child run/);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
    expect(supervisorProvider.calls).toBe(2);
  });

  it("rejects an empty required child output before parent synthesis", async () => {
    const childProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: {} } }], {}, "empty-child-provider");
    const agents = new AgentRepository(pool); const child = await agents.create(principal, base(childProvider.id, "Empty Child"));
    const supervisorProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "delegate-empty", name: "delegate_run", arguments: { targetAgentId: child.id, goal: "return useful evidence", roleName: "worker", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 100 } }] },
      { body: { type: "pause", reason: "wait_for_children" } },
      { body: { type: "final_answer", output: "must not complete" } },
    ], {}, "empty-supervisor-provider");
    const supervisor = await agents.create(principal, { ...base(supervisorProvider.id, "Empty Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [child.id], allowedRoles: ["worker"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false } });
    const run = await new RunRepository(pool).create(principal, supervisor.id, "empty children are invalid"); const worker = await runtime([supervisorProvider, childProvider]);
    await worker.tick(); await worker.tick(); expect(await new OrchestrationService(pool).advanceNext("scheduler")).toBe(true);
    await expect(worker.tick()).rejects.toThrow(/non-empty result/);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
    expect(supervisorProvider.calls).toBe(2);
  });

  it("propagates cancellation through descendants and prevents later execution", async () => {
    const childProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: "never" } }], {}, "cancel-child-provider");
    const agents = new AgentRepository(pool); const child = await agents.create(principal, base(childProvider.id, "Cancel Child"));
    const supervisorProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "delegate-cancel", name: "delegate_run", arguments: { targetAgentId: child.id, goal: "do not run", roleName: "worker", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 0 } }] },
      { body: { type: "pause", reason: "wait_for_children" } },
    ], {}, "cancel-supervisor-provider");
    const supervisor = await agents.create(principal, { ...base(supervisorProvider.id, "Cancel Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [child.id], allowedRoles: ["worker"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 0, allowSharedContext: false } });
    const run = await new RunRepository(pool).create(principal, supervisor.id, "cancel hierarchy"); const worker = await runtime([supervisorProvider, childProvider]);
    await worker.tick(); await new RunRepository(pool).requestCancellation(principal, run.id);
    await worker.tick(); await worker.tick();
    const hierarchy = await pool.query("SELECT status FROM runs WHERE root_run_id=$1 ORDER BY delegation_depth", [run.id]);
    expect(hierarchy.rows.map((row) => row.status)).toEqual(["cancelled", "cancelled"]);
    expect(childProvider.calls).toBe(0);
    expect((await pool.query("SELECT status FROM orchestration_waits WHERE parent_run_id=$1", [run.id])).rows[0].status).toBe("cancelled");
  });

  it("enforces allowlists, concurrency, ancestry-loop prevention, tenant isolation, and keyed replay", async () => {
    const agents = new AgentRepository(pool);
    const target = await agents.create(principal, base("unused-target", "Target"));
    const denied = await agents.create(principal, base("unused-denied", "Denied"));
    let supervisor = await agents.create(principal, { ...base("unused-supervisor", "Policy Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [target.id], allowedRoles: ["worker"], maximumChildren: 2, maximumParallel: 1, maximumDepth: 2, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false } });
    const runs = new RunRepository(pool); const run = await runs.create(principal, supervisor.id, "policy checks");
    await runs.claimNext("policy-worker", 30); await runs.workerTransition(run.id, "policy-worker", "running");
    const registry = createToolRegistry(pool); const executor = new ToolExecutor(pool, registry);
    const input = { targetAgentId: target.id, goal: "one child", roleName: "worker", required: true, contextScope: "private" as const, tokenBudget: 1000, costBudgetMicrousd: 100 };
    const command = { principal, runId: run.id, workerId: "policy-worker", toolName: "delegate_run", arguments: input, idempotencyKey: "delegate-policy-one" };
    const first = await executor.execute(command) as { childRunId: string };
    expect((await executor.execute(command) as { childRunId: string }).childRunId).toBe(first.childRunId);
    await expect(executor.execute({ ...command, idempotencyKey: "delegate-policy-two", arguments: { ...input, goal: "parallel child" } })).rejects.toThrow(/parallel child count/);
    await expect(executor.execute({ ...command, idempotencyKey: "delegate-denied", arguments: { ...input, targetAgentId: denied.id } })).rejects.toThrow(/not allowed/);
    const otherTenant: Principal = { tenantId: "a1000000-0000-4000-8000-000000000099", userId: "a1000000-0000-4000-8000-000000000098", roles: [] };
    await expect(new OrchestrationService(pool).tree(otherTenant, run.id)).rejects.toThrow("Run not found");

    await runs.requestCancellation(principal, first.childRunId); await runs.finalizeUnleasedCancellations();
    supervisor = await agents.patch(principal, supervisor.id, supervisor.version, { orchestrationPolicy: { ...supervisor.orchestrationPolicy, allowedAgentIds: [supervisor.id] } });
    const loopRun = await runs.create(principal, supervisor.id, "loop check");
    await runs.claimNext("loop-worker", 30); await runs.workerTransition(loopRun.id, "loop-worker", "running");
    await expect(executor.execute({ principal, runId: loopRun.id, workerId: "loop-worker", toolName: "delegate_run", arguments: { ...input, targetAgentId: supervisor.id }, idempotencyKey: "delegate-loop" })).rejects.toThrow(/ancestry loop/);
  });

  it("keeps a sensitive child tool behind its own durable approval before the parent resumes", async () => {
    const agents = new AgentRepository(pool);
    const childProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "child-note", name: "structured_note_storage", arguments: { title: "Child result", body: "approved evidence", tags: ["delegated"] } }] },
      { body: { type: "final_answer", output: { noteStored: true } } },
    ], {}, "approval-child-provider");
    const child = await agents.create(principal, { ...base(childProvider.id, "Approval Child"), allowedTools: ["structured_note_storage"], approvalPolicy: { structured_note_storage: { requiredApproverRole: "approval_reviewer", separationOfDuties: true } } });
    const supervisorProvider = new DeterministicTestProvider([
      { toolCalls: [{ id: "delegate-approval-child", name: "delegate_run", arguments: { targetAgentId: child.id, goal: "store approved evidence", roleName: "operator", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 100 } }] },
      { body: { type: "pause", reason: "wait_for_children" } },
      { body: { type: "final_answer", output: { childApproved: true } } },
    ], {}, "approval-supervisor-provider");
    const supervisor = await agents.create(principal, { ...base(supervisorProvider.id, "Approval Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [child.id], allowedRoles: ["operator"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false } });
    const run = await new RunRepository(pool).create(principal, supervisor.id, "approval stays canonical");
    const tools = createToolRegistry(pool); const worker = await runtime([supervisorProvider, childProvider], ["note_writer"]);
    await worker.tick(); await worker.tick();
    const childRun = (await pool.query("SELECT id,status FROM runs WHERE parent_run_id=$1", [run.id])).rows[0];
    expect(childRun.status).toBe("waiting_for_approval");
    expect((await pool.query("SELECT count(*)::int AS count FROM structured_notes")).rows[0].count).toBe(0);
    expect(await new OrchestrationService(pool).advanceNext("scheduler")).toBe(false);
    const approval = (await pool.query("SELECT id FROM approvals WHERE run_id=$1", [childRun.id])).rows[0];
    const approver: Principal = { tenantId: principal.tenantId, userId: "a1000000-0000-4000-8000-000000000003", roles: ["approval_reviewer"] };
    await new ApprovalService(pool, tools).decide(approver, approval.id, "approved");
    await worker.tick();
    expect((await pool.query("SELECT count(*)::int AS count FROM structured_notes")).rows[0].count).toBe(1);
    expect(await new OrchestrationService(pool).advanceNext("scheduler")).toBe(true);
    await worker.tick();
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("completed");
  });

  it("reconciles a crash after child creation without creating a duplicate child", async () => {
    const agents = new AgentRepository(pool); const target = await agents.create(principal, base("unused-recovery-child", "Recovery Child"));
    const supervisor = await agents.create(principal, { ...base("unused-recovery-parent", "Recovery Supervisor"), allowedTools: ["delegate_run"], composition: { orchestrator: "native" }, orchestrationPolicy: { enabled: true, allowedAgentIds: [target.id], allowedRoles: ["worker"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false } });
    const runs = new RunRepository(pool); const run = await runs.create(principal, supervisor.id, "recover delegation");
    await runs.claimNext("recovery-worker", 30); await runs.workerTransition(run.id, "recovery-worker", "running");
    const registry = createToolRegistry(pool); const executor = new ToolExecutor(pool, registry);
    await executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "delegate_run", arguments: { targetAgentId: target.id, goal: "persist once", roleName: "worker", required: true, contextScope: "private", tokenBudget: 1000, costBudgetMicrousd: 100 }, idempotencyKey: "delegate-crash-boundary" });
    const execution = (await pool.query("SELECT id,step_id FROM tool_executions WHERE run_id=$1 AND idempotency_key='delegate-crash-boundary'", [run.id])).rows[0];
    await pool.query("UPDATE tool_executions SET status='unknown',output=NULL,reconciliation_status='pending' WHERE id=$1", [execution.id]);
    await pool.query("UPDATE steps SET status='unknown',output=NULL WHERE id=$1", [execution.step_id]);
    await runs.workerTransition(run.id, "recovery-worker", "paused", { code: "simulated_crash_after_child_creation" });
    expect(await new ReconciliationService(pool, registry).reconcileNext("delegation-reconciler", 30)).toBe(true);
    expect((await runs.get(principal.tenantId, run.id)).status).toBe("queued");
    expect((await pool.query("SELECT count(*)::int AS count FROM runs WHERE parent_run_id=$1", [run.id])).rows[0].count).toBe(1);
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE id=$1", [execution.id])).rows[0]).toEqual({ status: "succeeded", reconciliation_status: "resolved" });
  });
});
