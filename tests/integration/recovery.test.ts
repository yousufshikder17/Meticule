import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { ToolExecutor } from "../../src/tools/executor.js";
import { ReconciliationService } from "../../src/tools/reconciliation-service.js";
import { ToolRegistry } from "../../src/tools/types.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const principal: Principal = { tenantId: "c1111111-1111-4111-8111-111111111111", userId: "c2222222-2222-4222-8222-222222222222", roles: [] };

beforeEach(async () => pool.query("TRUNCATE tool_execution_attempts,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

async function runningRun(toolNames: string[]) {
  const agent = await new AgentRepository(pool).create(principal, { name: `recovery-${crypto.randomUUID()}`, systemInstructions: "test", model: { provider: "unused", model: "unused" }, allowedTools: toolNames, maximumSteps: 20, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null });
  const runs = new RunRepository(pool); const run = await runs.create(principal, agent.id, "recovery test");
  const claimed = await runs.claimNext("recovery-worker", 30); await runs.workerTransition(claimed!.id, "recovery-worker", "running"); return run;
}

describe("Stage 5 tool retry and reconciliation", () => {
  it("classifies a worker crash during an external effect as unknown without replay", async () => {
    const run = await runningRun([]);
    const step = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input,started_at) VALUES($1,$2,1,'tool','running','crash-tool',1,'{}',now()) RETURNING id", [principal.tenantId, run.id]);
    await pool.query("INSERT INTO tool_executions(tenant_id,run_id,step_id,tool_name,idempotency_key,status,validated_arguments,operation_hash,retry_safety,attempt_count) VALUES($1,$2,$3,'external','crash-tool','running','{}',$4,'reconcilable',1)", [principal.tenantId, run.id, step.rows[0].id, "a".repeat(64)]);
    await pool.query("UPDATE runs SET current_step=1,lease_expires_at=now()-interval '1 second' WHERE id=$1", [run.id]);
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("paused");
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE run_id=$1", [run.id])).rows[0]).toEqual({ status: "unknown", reconciliation_status: "pending" });
  });

  it("turns a crash before tool invocation into a deterministic non-effect", async () => {
    const run = await runningRun([]);
    const step = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input) VALUES($1,$2,1,'tool','pending','pre-invoke',0,'{}') RETURNING id", [principal.tenantId, run.id]);
    await pool.query("INSERT INTO tool_executions(tenant_id,run_id,step_id,tool_name,idempotency_key,status,validated_arguments,operation_hash,retry_safety) VALUES($1,$2,$3,'external','pre-invoke','pending','{}',$4,'non_retryable')", [principal.tenantId, run.id, step.rows[0].id, "b".repeat(64)]);
    await pool.query("UPDATE runs SET current_step=1,lease_expires_at=now()-interval '1 second' WHERE id=$1", [run.id]);
    await new RunRepository(pool).recoverExpired();
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
    expect((await pool.query("SELECT status FROM tool_executions WHERE run_id=$1", [run.id])).rows[0].status).toBe("failed");
  });

  it("fails closed when a worker dies during a provider invocation", async () => {
    const run = await runningRun([]);
    const step = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input,started_at) VALUES($1,$2,1,'model','running','model-crash',1,'{}',now()) RETURNING id", [principal.tenantId, run.id]);
    await pool.query("INSERT INTO model_attempts(tenant_id,run_id,step_id,request_id,provider_id,model_id,attempt_number,status,started_at) VALUES($1,$2,$3,$4,'provider','model',1,'running',now())", [principal.tenantId, run.id, step.rows[0].id, crypto.randomUUID()]);
    await pool.query("UPDATE runs SET current_step=1,lease_expires_at=now()-interval '1 second' WHERE id=$1", [run.id]);
    await new RunRepository(pool).recoverExpired();
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
    expect((await pool.query("SELECT status,normalized_error_code FROM model_attempts WHERE run_id=$1", [run.id])).rows[0]).toEqual({ status: "unknown", normalized_error_code: "unknown_outcome" });
  });

  it("binds an idempotency key to immutable operation content", async () => {
    const registry = new ToolRegistry(); registry.register({ name: "echo", description: "echo", inputSchema: z.object({ value: z.string() }), outputSchema: z.object({ value: z.string() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 100, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "pure", retrySafety: "pure", async execute(input) { return input; } });
    const run = await runningRun(["echo"]); const executor = new ToolExecutor(pool, registry);
    await executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "echo", arguments: { value: "one" }, idempotencyKey: "bound-key" });
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "echo", arguments: { value: "two" }, idempotencyKey: "bound-key" })).rejects.toThrow(/different operation/);
  });

  it("retries a replay-safe tool with bounded persisted attempts", async () => {
    let calls = 0; const registry = new ToolRegistry(); registry.register({ name: "flaky", description: "flaky", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 100, retryPolicy: { maxAttempts: 2, retryableErrors: ["40001"] }, idempotency: "pure", retrySafety: "pure", async execute() { calls += 1; if (calls === 1) throw Object.assign(new Error("serialization"), { code: "40001" }); return { ok: true }; } });
    const run = await runningRun(["flaky"]); const executor = new ToolExecutor(pool, registry);
    expect(await executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "flaky", arguments: {}, idempotencyKey: "flaky-key" })).toEqual({ ok: true });
    expect(calls).toBe(2); expect(Number((await pool.query("SELECT count(*) AS count FROM tool_execution_attempts")).rows[0].count)).toBe(2);
  });

  it("stops at the configured retry budget", async () => {
    let calls = 0; const registry = new ToolRegistry(); registry.register({ name: "always_flaky", description: "flaky", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 100, retryPolicy: { maxAttempts: 2, retryableErrors: ["40001"] }, idempotency: "pure", retrySafety: "pure", async execute() { calls += 1; throw Object.assign(new Error("serialization"), { code: "40001" }); } });
    const run = await runningRun(["always_flaky"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "always_flaky", arguments: {}, idempotencyKey: "retry-budget" })).rejects.toThrow(/serialization/);
    expect(calls).toBe(2); expect(Number((await pool.query("SELECT count(*) AS count FROM tool_execution_attempts")).rows[0].count)).toBe(2);
  });

  it("stops retrying when cancellation arrives between attempts", async () => {
    let calls = 0; const registry = new ToolRegistry(); registry.register({ name: "cancel_retry", description: "cancel", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5_000, retryPolicy: { maxAttempts: 3, retryableErrors: ["40001"] }, idempotency: "pure", retrySafety: "pure", async execute(_input, context) { calls += 1; await new RunRepository(pool).requestCancellation(context.principal, context.runId); throw Object.assign(new Error("serialization"), { code: "40001" }); } });
    const run = await runningRun(["cancel_retry"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "cancel_retry", arguments: {}, idempotencyKey: "cancel-retry" })).rejects.toThrow(/Cancellation/);
    expect(calls).toBe(1); expect((await pool.query("SELECT status FROM tool_executions WHERE idempotency_key='cancel-retry'")).rows[0].status).toBe("cancelled");
  });

  it("never replays an unknown effect and resolves it through one leased reconciliation", async () => {
    let effects = 0; const external = new Map<string, { ok: boolean }>(); const registry = new ToolRegistry(); registry.register({ name: "reconcilable", description: "reconcilable", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "high", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5, retryPolicy: { maxAttempts: 3, retryableErrors: ["timeout"] }, idempotency: "keyed_side_effect", retrySafety: "reconcilable", async execute(_input, context) { effects += 1; external.set(context.idempotencyKey, { ok: true }); await new Promise((resolve) => setTimeout(resolve, 30)); return { ok: true }; }, async reconcile(_input, context) { await new Promise((resolve) => setTimeout(resolve, 20)); const output = external.get(context.idempotencyKey); return output ? { status: "succeeded", output } : { status: "pending" }; } });
    const run = await runningRun(["reconcilable"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "reconcilable", arguments: {}, idempotencyKey: "external-key" })).rejects.toThrow(/unknown/);
    await new RunRepository(pool).workerTransition(run.id, "recovery-worker", "paused", { code: "reconciliation_required" });
    await pool.query("UPDATE tool_executions SET reconciliation_status='running',reconciliation_owner='dead-reconciler',reconciliation_expires_at=now()-interval '1 second' WHERE idempotency_key='external-key'");
    const reconciler = new ReconciliationService(pool, registry);
    const results = await Promise.all([reconciler.reconcileNext("reconciler-a"), reconciler.reconcileNext("reconciler-b")]);
    expect(results.filter(Boolean)).toHaveLength(1); expect(effects).toBe(1);
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE idempotency_key='external-key'")).rows[0]).toEqual({ status: "succeeded", reconciliation_status: "resolved" });
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("queued");
  });

  it("persists failed and still-pending reconciliation outcomes without replay", async () => {
    let effects = 0; const registry = new ToolRegistry(); registry.register({ name: "status_lookup", description: "lookup", inputSchema: z.object({ outcome: z.enum(["failed", "pending"]) }), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "high", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "keyed_side_effect", retrySafety: "reconcilable", async execute() { effects += 1; await new Promise((resolve) => setTimeout(resolve, 30)); return { ok: true }; }, async reconcile(input) { return (input as { outcome: string }).outcome === "failed" ? { status: "failed", error: { code: "external_rejected" } } : { status: "pending" }; } });
    const run = await runningRun(["status_lookup"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "status_lookup", arguments: { outcome: "failed" }, idempotencyKey: "failed-reconcile" })).rejects.toThrow(/unknown/);
    await new RunRepository(pool).workerTransition(run.id, "recovery-worker", "paused", { code: "reconciliation_required" });
    expect(await new ReconciliationService(pool, registry).reconcileNext("reconciler")).toBe(true);
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE idempotency_key='failed-reconcile'")).rows[0]).toEqual({ status: "failed", reconciliation_status: "resolved" });
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("queued");
    const runs = new RunRepository(pool); const claimed = await runs.claimNext("recovery-worker-2", 30); await runs.workerTransition(claimed!.id, "recovery-worker-2", "running");
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker-2", toolName: "status_lookup", arguments: { outcome: "pending" }, idempotencyKey: "pending-reconcile" })).rejects.toThrow(/unknown/);
    await runs.workerTransition(run.id, "recovery-worker-2", "paused", { code: "reconciliation_required" });
    expect(await new ReconciliationService(pool, registry).reconcileNext("reconciler-2")).toBe(true);
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE idempotency_key='pending-reconcile'")).rows[0]).toEqual({ status: "unknown", reconciliation_status: "pending" });
    expect((await runs.get(principal.tenantId, run.id)).status).toBe("paused"); expect(effects).toBe(2);
  });

  it("leaves unsupported or unresolved reconciliation fail-closed", async () => {
    const registry = new ToolRegistry(); registry.register({ name: "manual", description: "manual", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "high", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "keyed_side_effect", retrySafety: "non_retryable", async execute() { await new Promise((resolve) => setTimeout(resolve, 30)); return { ok: true }; } });
    const run = await runningRun(["manual"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal, runId: run.id, workerId: "recovery-worker", toolName: "manual", arguments: {}, idempotencyKey: "manual-key" })).rejects.toThrow(/unknown/);
    expect((await pool.query("SELECT status,reconciliation_status FROM tool_executions WHERE idempotency_key='manual-key'")).rows[0]).toEqual({ status: "unknown", reconciliation_status: "manual" });
    expect(await new ReconciliationService(pool, registry).reconcileNext("reconciler")).toBe(false);
  });
});
