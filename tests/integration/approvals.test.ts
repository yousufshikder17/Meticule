import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { ApprovalService } from "../../src/approvals/approval-service.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { canonicalJsonHash } from "../../src/domain/canonical-json.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const requester: Principal = { tenantId: "a1111111-1111-4111-8111-111111111111", userId: "a2222222-2222-4222-8222-222222222222", roles: ["note_writer"] };
const approver: Principal = { tenantId: requester.tenantId, userId: "a3333333-3333-4333-8333-333333333333", roles: ["approval_reviewer"] };
const otherTenant: Principal = { tenantId: "b1111111-1111-4111-8111-111111111111", userId: "b2222222-2222-4222-8222-222222222222", roles: ["approval_reviewer"] };

beforeEach(async () => pool.query("TRUNCATE structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

async function setup() {
  const provider = new DeterministicTestProvider([
    { body: { type: "call_tool", toolName: "structured_note_storage", arguments: { title: "  frozen title  ", body: "exact body", tags: ["approval"] }, idempotencyKey: "approved-note-once" } },
    { body: { type: "final_answer", output: { done: true } } },
  ]);
  const providers = new ProviderRegistry(); providers.register(provider);
  const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
  const tools = createToolRegistry(pool);
  const loop = new AgentLoop(pool, providers, engines, tools);
  const agent = await new AgentRepository(pool).create(requester, {
    name: `approval-${crypto.randomUUID()}`, systemInstructions: "Use the note tool once.",
    model: { provider: provider.id, model: "deterministic", maxOutputTokens: 100, timeoutMs: 1000,
      inputCostMicrousdPerMillion: 0, outputCostMicrousdPerMillion: 0, cachedCostMicrousdPerMillion: 0 },
    allowedTools: ["structured_note_storage"], maximumSteps: 10, tokenBudget: 1000, costBudgetMicrousd: 1000,
    approvalPolicy: { structured_note_storage: { requiredApproverRole: "approval_reviewer", separationOfDuties: true, riskExplanation: "A durable note will be created" } }, outputSchema: null,
  });
  const run = await new RunRepository(pool).create(requester, agent.id, "store the approved note");
  const worker = (id: string) => new LifecycleWorker(pool, { workerId: id, leaseSeconds: 30 }, { execute: (runId, workerId) => loop.execute(runId, workerId, requester.roles) });
  return { provider, run, worker, approvals: new ApprovalService(pool, tools) };
}

async function pendingApproval(runId: string) {
  return (await pool.query("SELECT * FROM approvals WHERE run_id=$1", [runId])).rows[0];
}

describe("Stage 4 durable approvals", () => {
  it("atomically freezes a validated action, releases the lease, and does not execute", async () => {
    const { provider, run, worker } = await setup();
    expect(await worker("proposal-worker").tick()).toBe(true);
    const stored = await new RunRepository(pool).get(requester.tenantId, run.id);
    expect(stored.status).toBe("waiting_for_approval");
    expect(stored.leaseOwner).toBeNull();
    expect(Number((await pool.query("SELECT count(*) AS count FROM worker_leases WHERE run_id=$1", [run.id])).rows[0].count)).toBe(0);
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
    const approval = await pendingApproval(run.id);
    expect(approval.validated_arguments).toEqual({ title: "frozen title", body: "exact body", tags: ["approval"] });
    expect(approval.canonical_argument_hash).toBe(canonicalJsonHash(approval.validated_arguments));
    expect(approval.idempotency_key).toBe("approved-note-once");
    await expect(pool.query("UPDATE approvals SET validated_arguments=$1 WHERE id=$2", [JSON.stringify({ title: "tampered", body: "x", tags: [] }), approval.id])).rejects.toThrow(/immutable/);
    expect(provider.calls).toBe(1);
  });

  it("enforces role, separation of duties, tenant ownership, and records denied decisions", async () => {
    const { run, worker, approvals } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id);
    await expect(approvals.decide({ ...approver, roles: [] }, approval.id, "approved")).rejects.toThrow(/role/);
    await expect(approvals.decide({ ...requester, roles: ["note_writer", "approval_reviewer"] }, approval.id, "approved")).rejects.toThrow(/Requester/);
    await expect(approvals.decide(otherTenant, approval.id, "approved")).rejects.toThrow(/not found/);
    await expect(approvals.get({ ...approver, userId: "a4444444-4444-4444-8444-444444444444", roles: [] }, approval.id)).rejects.toThrow(/inspection/);
    expect(await approvals.listForRun({ ...approver, userId: "a4444444-4444-4444-8444-444444444444", roles: [] }, run.id)).toEqual([]);
    expect(Number((await pool.query("SELECT count(*) AS count FROM audit_events WHERE run_id=$1 AND event_type='approval.decision_denied'", [run.id])).rows[0].count)).toBe(2);
    expect((await pendingApproval(run.id)).decision).toBe("pending");
  });

  it("makes decisions idempotent, rejects conflicts, and requeues without inline execution", async () => {
    const { run, worker, approvals } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id);
    expect((await approvals.decide(approver, approval.id, "approved")).decision).toBe("approved");
    expect((await approvals.decide(approver, approval.id, "approved")).decision).toBe("approved");
    await expect(approvals.decide(approver, approval.id, "rejected")).rejects.toThrow(/already approved/);
    expect((await new RunRepository(pool).get(requester.tenantId, run.id)).status).toBe("queued");
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
  });

  it("records approval through the HTTP API without executing the tool inline", async () => {
    const { run, worker } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id);
    const app = createApp(pool, new DevelopmentHeaderAuthenticator());
    const headers = { "content-type": "application/json", "x-tenant-id": approver.tenantId, "x-user-id": approver.userId, "x-roles": "approval_reviewer" };
    expect((await app.request(`/approvals/${approval.id}`, { headers })).status).toBe(200);
    const response = await app.request(`/approvals/${approval.id}/approve`, { method: "POST", headers, body: JSON.stringify({ comment: "approved in API" }) });
    expect(response.status).toBe(200);
    expect((await response.json() as { decision: string }).decision).toBe("approved");
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
  });

  it("rejects durably and lets the model continue from a deterministic rejection result", async () => {
    const { run, worker, approvals, provider } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id);
    await approvals.decide(approver, approval.id, "rejected", "not appropriate");
    expect((await pool.query("SELECT status FROM steps WHERE id=$1", [approval.step_id])).rows[0].status).toBe("failed");
    expect(await worker("replan-worker").tick()).toBe(true);
    expect((await new RunRepository(pool).get(requester.tenantId, run.id)).status).toBe("completed");
    expect(provider.calls).toBe(2);
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
  });

  it("resumes the exact frozen action after worker restart without model regeneration and executes once", async () => {
    const { run, worker, approvals, provider } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id); await approvals.decide(approver, approval.id, "approved");
    const [first, second] = await Promise.all([worker("restart-worker-a").tick(), worker("restart-worker-b").tick()]);
    expect([first, second].filter(Boolean)).toHaveLength(1);
    expect((await new RunRepository(pool).get(requester.tenantId, run.id)).status).toBe("completed");
    expect(provider.calls).toBe(2);
    const notes = (await pool.query("SELECT title,body,tags FROM structured_notes")).rows;
    expect(notes).toEqual([{ title: "frozen title", body: "exact body", tags: ["approval"] }]);
    const persisted = await pendingApproval(run.id);
    expect(persisted.consumed_at).not.toBeNull();
    expect(Number((await pool.query("SELECT count(*) AS count FROM tool_executions WHERE idempotency_key='approved-note-once' AND status='succeeded'")).rows[0].count)).toBe(1);
  });

  it("prevents execution when cancellation arrives while waiting", async () => {
    const { run, worker } = await setup(); await worker("proposal-worker").tick();
    await new RunRepository(pool).requestCancellation(requester, run.id);
    expect(await worker("cancel-worker").tick()).toBe(false);
    expect((await new RunRepository(pool).get(requester.tenantId, run.id)).status).toBe("cancelled");
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
    expect(Number((await pool.query("SELECT count(*) AS count FROM audit_events WHERE run_id=$1 AND event_type='approval.cancellation_intervened'", [run.id])).rows[0].count)).toBe(1);
  });

  it("prevents execution when cancellation arrives after approval but before reclaim", async () => {
    const { run, worker, approvals } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id); await approvals.decide(approver, approval.id, "approved");
    await new RunRepository(pool).requestCancellation(requester, run.id);
    expect(await worker("cancel-worker").tick()).toBe(false);
    expect((await new RunRepository(pool).get(requester.tenantId, run.id)).status).toBe("cancelled");
    expect(Number((await pool.query("SELECT count(*) AS count FROM structured_notes")).rows[0].count)).toBe(0);
  });

  it("records the complete successful approval audit trail", async () => {
    const { run, worker, approvals } = await setup(); await worker("proposal-worker").tick();
    const approval = await pendingApproval(run.id); await approvals.decide(approver, approval.id, "approved"); await worker("resume-worker").tick();
    const events = (await pool.query("SELECT event_type FROM audit_events WHERE run_id=$1", [run.id])).rows.map((row) => row.event_type);
    for (const expected of ["approval.requested", "approval.granted", "run.requeued", "approval.consumed", "approval.execution_started", "approval.execution_succeeded", "run.completed"]) expect(events).toContain(expected);
  });
});
