import pg from "pg";
import { beforeEach, afterAll, describe, expect, it } from "vitest";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const app = createApp(pool, new DevelopmentHeaderAuthenticator());
const headers = { "content-type": "application/json", "x-tenant-id": "44444444-4444-4444-8444-444444444444", "x-user-id": "55555555-5555-4555-8555-555555555555" };
const principal: Principal = { tenantId: headers["x-tenant-id"], userId: headers["x-user-id"], roles: [] };

beforeEach(async () => pool.query("TRUNCATE structured_notes,audit_events,usage_records,approvals,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("Stage 1/2 HTTP API", () => {
  it("rejects missing identity and exposes real tool metadata", async () => {
    expect((await app.request("/tools")).status).toBe(401);
    const response = await app.request("/tools", { headers });
    expect(response.status).toBe(200);
    expect((await response.json()) as unknown[]).toHaveLength(5);
  });

  it("creates an agent and run, reads it, and durably requests cancellation", async () => {
    const agentResponse = await app.request("/agents", { method: "POST", headers, body: JSON.stringify({
      name: "api-agent", systemInstructions: "Operate safely", model: { provider: "disabled", model: "stage-3" },
      allowedTools: ["calculator"], maximumSteps: 5, tokenBudget: 1000, costBudgetMicrousd: 1000,
      approvalPolicy: {}, outputSchema: null,
    }) });
    expect(agentResponse.status).toBe(201);
    const agent = await agentResponse.json() as { id: string };
    const runResponse = await app.request(`/agents/${agent.id}/runs`, { method: "POST", headers, body: JSON.stringify({ goal: "calculate safely" }) });
    expect(runResponse.status).toBe(201);
    const run = await runResponse.json() as { id: string };
    expect((await app.request(`/runs/${run.id}`, { headers })).status).toBe(200);
    const cancelled = await app.request(`/runs/${run.id}/cancel`, { method: "POST", headers });
    expect(cancelled.status).toBe(200);
    expect((await cancelled.json() as { status: string }).status).toBe("cancelling");
  });

  it.each(["pending", "running", "manual"] as const)("rejects manual resume while %s tool reconciliation remains unresolved", async (reconciliationStatus) => {
    const agent = await new AgentRepository(pool).create(principal, {
      name: `resume-guard-${reconciliationStatus}`, systemInstructions: "Operate safely", model: { provider: "disabled", model: "test" },
      allowedTools: [], maximumSteps: 5, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null,
    });
    const runs = new RunRepository(pool);
    const run = await runs.create(principal, agent.id, "do not replay an uncertain effect");
    await runs.claimNext("resume-guard-worker", 30);
    await runs.workerTransition(run.id, "resume-guard-worker", "running");
    const step = await pool.query(
      "INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input,started_at,finished_at) VALUES($1,$2,1,'tool','unknown',$3,1,'{}',now(),now()) RETURNING id",
      [principal.tenantId, run.id, `resume-${reconciliationStatus}`],
    );
    await pool.query(
      `INSERT INTO tool_executions(
         tenant_id,run_id,step_id,tool_name,idempotency_key,status,validated_arguments,operation_hash,retry_safety,attempt_count,
         reconciliation_status,reconciliation_owner,reconciliation_expires_at
       ) VALUES($1,$2,$3,'external_side_effect',$4,'unknown','{}',$5,'reconcilable',1,$6,$7,$8)`,
      [principal.tenantId, run.id, step.rows[0].id, `resume-${reconciliationStatus}`, "a".repeat(64), reconciliationStatus,
        reconciliationStatus === "running" ? "active-reconciler" : null, reconciliationStatus === "running" ? new Date(Date.now() + 30_000) : null],
    );
    await runs.workerTransition(run.id, "resume-guard-worker", "paused", { code: "reconciliation_required" });
    const paused = await runs.get(principal.tenantId, run.id);

    const response = await app.request(`/runs/${run.id}/resume`, {
      method: "POST", headers, body: JSON.stringify({ expectedVersion: paused.version }),
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({ error: "conflict", message: "Run cannot resume while external tool-effect reconciliation remains unresolved" });
    expect((await runs.get(principal.tenantId, run.id)).status).toBe("paused");
    const executions = await pool.query("SELECT status,reconciliation_status,reconciliation_owner FROM tool_executions WHERE tenant_id=$1 AND run_id=$2", [principal.tenantId, run.id]);
    expect(executions.rows).toEqual([{ status: "unknown", reconciliation_status: reconciliationStatus, reconciliation_owner: reconciliationStatus === "running" ? "active-reconciler" : null }]);
    expect(Number((await pool.query("SELECT count(*) AS count FROM tool_executions WHERE tenant_id=$1 AND run_id=$2", [principal.tenantId, run.id])).rows[0].count)).toBe(1);
  });

  it("still resumes an ordinary paused run with no unresolved dependency", async () => {
    const agent = await new AgentRepository(pool).create(principal, {
      name: "ordinary-resume", systemInstructions: "Wait for clarification", model: { provider: "disabled", model: "test" },
      allowedTools: [], maximumSteps: 5, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null,
    });
    const runs = new RunRepository(pool);
    const run = await runs.create(principal, agent.id, "pause normally");
    await runs.claimNext("ordinary-resume-worker", 30);
    await runs.workerTransition(run.id, "ordinary-resume-worker", "running");
    const paused = await runs.workerTransition(run.id, "ordinary-resume-worker", "paused", { code: "clarification_requested" });

    const response = await app.request(`/runs/${run.id}/resume`, {
      method: "POST", headers, body: JSON.stringify({ expectedVersion: paused.version }),
    });

    expect(response.status).toBe(200);
    expect((await response.json() as { status: string }).status).toBe("queued");
    expect((await runs.get(principal.tenantId, run.id)).status).toBe("queued");
  });
});
