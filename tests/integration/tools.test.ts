import pg from "pg";
import { beforeEach, afterAll, describe, expect, it } from "vitest";
import { z } from "zod";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { ToolExecutor } from "../../src/tools/executor.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { ToolRegistry } from "../../src/tools/types.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const base: Principal = { tenantId: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", userId: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", roles: [] };

async function runningRun(tools: string[], principal = base) {
  const agent = await new AgentRepository(pool).create(principal, {
    name: `tools-${crypto.randomUUID()}`, systemInstructions: "test", model: { provider: "test-only", model: "none" },
    allowedTools: tools, maximumSteps: 10, tokenBudget: 100, costBudgetMicrousd: 100, approvalPolicy: {}, outputSchema: null,
  });
  const repository = new RunRepository(pool); const run = await repository.create(principal, agent.id, "tool test");
  const claimed = await repository.claimNext("tool-worker", 30); await repository.workerTransition(claimed!.id, "tool-worker", "running");
  return run;
}

beforeEach(async () => pool.query("TRUNCATE structured_notes,audit_events,usage_records,approvals,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("typed tool executor", () => {
  it("rejects invalid input before persisting execution", async () => {
    const run = await runningRun(["calculator"]); const executor = new ToolExecutor(pool, createToolRegistry(pool));
    await expect(executor.execute({ principal: base, runId: run.id, workerId: "tool-worker", toolName: "calculator", arguments: { expression: "2 + process.exit()" }, idempotencyKey: "bad" })).rejects.toThrow();
    expect(Number((await pool.query("SELECT count(*) AS count FROM tool_executions")).rows[0].count)).toBe(0);
  });

  it("rejects tools absent from the agent allowlist", async () => {
    const run = await runningRun([]); const executor = new ToolExecutor(pool, createToolRegistry(pool));
    await expect(executor.execute({ principal: base, runId: run.id, workerId: "tool-worker", toolName: "calculator", arguments: { expression: "2+2" }, idempotencyKey: "unauthorized" })).rejects.toThrow(/not allowed/);
  });

  it("persists validated results and reuses a completed idempotency key", async () => {
    const principal = { ...base, roles: ["note_writer"] }; const run = await runningRun(["structured_note_storage"], principal);
    const executor = new ToolExecutor(pool, createToolRegistry(pool));
    const command = { principal, runId: run.id, workerId: "tool-worker", toolName: "structured_note_storage", arguments: { title: "fact", body: "durable", tags: ["test"] }, idempotencyKey: "note-once" };
    const first = await executor.execute(command); const second = await executor.execute(command);
    expect(second).toEqual(first);
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(1);
    expect((await pool.query("SELECT status,output FROM tool_executions WHERE idempotency_key='note-once'")).rows[0].status).toBe("succeeded");
  });

  it("runs read-only SQL under the restricted demo role", async () => {
    const principal = { ...base, roles: ["demo_reader"] }; const run = await runningRun(["demo_readonly_sql"], principal);
    const executor = new ToolExecutor(pool, createToolRegistry(pool));
    const output = await executor.execute({ principal, runId: run.id, workerId: "tool-worker", toolName: "demo_readonly_sql", arguments: { query: "SELECT name, price_cents FROM demo_products ORDER BY id" }, idempotencyKey: "read-1" }) as { rows: unknown[] };
    expect(output.rows).toHaveLength(3);
    await expect(executor.execute({ principal, runId: run.id, workerId: "tool-worker", toolName: "demo_readonly_sql", arguments: { query: "DELETE FROM demo_products" }, idempotencyKey: "write-1" })).rejects.toThrow(/SELECT/);
  });

  it("records a clean timeout and checks cancellation before execution", async () => {
    const registry = new ToolRegistry();
    registry.register({ name: "slow", description: "test only", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "pure", retrySafety: "pure", async execute() { await new Promise((resolve) => setTimeout(resolve, 30)); return { ok: true }; } });
    const run = await runningRun(["slow"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal: base, runId: run.id, workerId: "tool-worker", toolName: "slow", arguments: {}, idempotencyKey: "slow-1" })).rejects.toThrow(/timed out/);
    expect((await pool.query("SELECT status FROM tool_executions WHERE idempotency_key='slow-1'")).rows[0].status).toBe("failed");
    const repository = new RunRepository(pool); await repository.requestCancellation(base, run.id);
    await expect(executor.execute({ principal: base, runId: run.id, workerId: "tool-worker", toolName: "slow", arguments: {}, idempotencyKey: "slow-2" })).rejects.toThrow(/Cancellation/);
  });

  it("persists invalid output as a failed execution", async () => {
    const registry = new ToolRegistry();
    registry.register({ name: "bad_output", description: "test only", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "low", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 100, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "pure", retrySafety: "pure", async execute() { return { wrong: true }; } });
    const run = await runningRun(["bad_output"]); const executor = new ToolExecutor(pool, registry);
    await expect(executor.execute({ principal: base, runId: run.id, workerId: "tool-worker", toolName: "bad_output", arguments: {}, idempotencyKey: "bad-output-1" })).rejects.toThrow();
    expect((await pool.query("SELECT status FROM tool_executions WHERE idempotency_key='bad-output-1'")).rows[0].status).toBe("failed");
  });

  it("fails closed with unknown status for a timed-out keyed side effect", async () => {
    const registry = new ToolRegistry();
    registry.register({ name: "uncertain_write", description: "test only", inputSchema: z.object({}), outputSchema: z.object({ ok: z.boolean() }), riskLevel: "medium", authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 5, retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "keyed_side_effect", retrySafety: "non_retryable", async execute() { await new Promise((resolve) => setTimeout(resolve, 30)); return { ok: true }; } });
    const run = await runningRun(["uncertain_write"]); const executor = new ToolExecutor(pool, registry);
    const command = { principal: base, runId: run.id, workerId: "tool-worker", toolName: "uncertain_write", arguments: {}, idempotencyKey: "uncertain-1" };
    await expect(executor.execute(command)).rejects.toThrow(/unknown.*reconciled/i);
    expect((await pool.query("SELECT status FROM tool_executions WHERE idempotency_key='uncertain-1'")).rows[0].status).toBe("unknown");
    await expect(executor.execute(command)).rejects.toThrow(/already unknown/);
  });

  it("enforces tenant isolation in the canonical executor", async () => {
    const run = await runningRun(["calculator"]); const executor = new ToolExecutor(pool, createToolRegistry(pool));
    const other = { ...base, tenantId: "cccccccc-cccc-4ccc-8ccc-cccccccccccc" };
    await expect(executor.execute({ principal: other, runId: run.id, workerId: "tool-worker", toolName: "calculator", arguments: { expression: "1+1" }, idempotencyKey: "cross-tenant" })).rejects.toThrow(/Run not found/);
    expect(Number((await pool.query("SELECT count(*) AS count FROM tool_executions WHERE idempotency_key='cross-tenant'")).rows[0].count)).toBe(0);
  });
});
