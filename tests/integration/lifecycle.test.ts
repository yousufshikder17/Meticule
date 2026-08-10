import pg from "pg";
import { beforeAll, beforeEach, afterAll, describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import type { Principal } from "../../src/db/types.js";

const { Pool } = pg;
const connectionString = process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform";
const pool = new Pool({ connectionString, max: 10 });
const principal: Principal = {
  tenantId: "11111111-1111-4111-8111-111111111111",
  userId: "22222222-2222-4222-8222-222222222222",
  roles: [],
};

async function createRun() {
  const agent = await new AgentRepository(pool).create(principal, {
    name: `agent-${crypto.randomUUID()}`, systemInstructions: "test", model: { provider: "test-only", model: "none" },
    allowedTools: [], maximumSteps: 10, tokenBudget: 100, costBudgetMicrousd: 100, approvalPolicy: {}, outputSchema: null,
  });
  return new RunRepository(pool).create(principal, agent.id, "verify durable lifecycle");
}

beforeAll(async () => {
  const exists = await pool.query<{ exists: boolean }>("SELECT to_regclass('public.runs') IS NOT NULL AS exists");
  if (!exists.rows[0]?.exists) await pool.query(await readFile(resolve(process.cwd(), "migrations/001_initial.sql"), "utf8"));
});
beforeEach(async () => pool.query("TRUNCATE audit_events,usage_records,approvals,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("durable lifecycle with PostgreSQL", () => {
  it("creates a durable queued run and rejects invalid transitions in both layers", async () => {
    const repository = new RunRepository(pool); const run = await createRun();
    expect((await repository.get(principal.tenantId, run.id)).status).toBe("queued");
    await expect(repository.transition(principal, run.id, run.version, "completed")).rejects.toThrow();
    await expect(pool.query("UPDATE runs SET status='completed' WHERE id=$1", [run.id])).rejects.toThrow(/invalid run transition/);
  });

  it("allows exactly one competing worker to claim a run", async () => {
    const repository = new RunRepository(pool); const run = await createRun();
    const claims = await Promise.all([repository.claimNext("worker-a", 30), repository.claimNext("worker-b", 30)]);
    expect(claims.filter(Boolean)).toHaveLength(1);
    const stored = await repository.get(principal.tenantId, run.id);
    expect(stored.status).toBe("claimed");
    expect(["worker-a", "worker-b"]).toContain(stored.leaseOwner);
  });

  it("rejects heartbeat by a competing worker", async () => {
    const repository = new RunRepository(pool); await createRun();
    const claimed = await repository.claimNext("owner", 30); expect(claimed).not.toBeNull();
    expect(await repository.heartbeat(claimed!.id, "intruder", 30)).toBe(false);
    expect(await repository.heartbeat(claimed!.id, "owner", 30)).toBe(true);
  });

  it("recovers an expired lease and preserves the run across repository restart", async () => {
    const first = new RunRepository(pool); const run = await createRun(); await first.claimNext("dead-worker", 30);
    await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [run.id]);
    const restarted = new RunRepository(pool);
    expect(await restarted.recoverExpired()).toBe(1);
    const recovered = await restarted.get(principal.tenantId, run.id);
    expect(recovered.status).toBe("queued"); expect(recovered.leaseOwner).toBeNull();
  });

  it("makes cancellation durable and prevents later completion", async () => {
    const repository = new RunRepository(pool); const run = await createRun();
    const cancelling = await repository.requestCancellation(principal, run.id);
    expect(cancelling.status).toBe("cancelling"); expect(cancelling.cancellationRequestedAt).toBeInstanceOf(Date);
    expect(await repository.finalizeUnleasedCancellations()).toBe(1);
    const cancelled = await repository.get(principal.tenantId, run.id);
    expect(cancelled.status).toBe("cancelled");
    await expect(repository.transition(principal, run.id, cancelled.version, "completed")).rejects.toThrow();
  });

  it("does not expose another tenant's run", async () => {
    const repository = new RunRepository(pool); const run = await createRun();
    await expect(repository.get("33333333-3333-4333-8333-333333333333", run.id)).rejects.toThrow("Run not found");
  });

  it("rejects stale optimistic versions and records transition audit evidence", async () => {
    const repository = new RunRepository(pool); const run = await createRun();
    const cancelling = await repository.requestCancellation(principal, run.id);
    await expect(repository.transition(principal, run.id, run.version, "cancelled")).rejects.toThrow(/version conflict/);
    expect(cancelling.version).toBeGreaterThan(run.version);
    const events = await pool.query("SELECT event_type FROM audit_events WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at", [principal.tenantId, run.id]);
    expect(events.rows.map((row) => row.event_type)).toEqual(["run.created", "run.cancellation_requested"]);
  });

  it("heartbeats the lease while a bounded processor is active", async () => {
    const run = await createRun(); const repository = new RunRepository(pool); let observedExpiry: Date | null = null;
    const worker = new LifecycleWorker(pool, { workerId: "heartbeat-worker", leaseSeconds: 3 }, { execute: async (runId) => {
      await new Promise((resolve) => setTimeout(resolve, 4000)); observedExpiry = (await repository.get(principal.tenantId, runId)).leaseExpiresAt;
    } });
    await worker.tick();
    expect(observedExpiry).toBeInstanceOf(Date); expect(observedExpiry!.getTime()).toBeGreaterThan(Date.now());
    await repository.workerTransition(run.id, "heartbeat-worker", "paused", { code: "test_cleanup" });
  });
});
