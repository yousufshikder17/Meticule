import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";
import { migrationReadiness } from "../../src/db/migration-state.js";
import type { Principal } from "../../src/db/types.js";

const { Pool } = pg; const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 20 });
const principal: Principal = { tenantId: "fa000000-0000-4000-8000-000000000001", userId: "fa000000-0000-4000-8000-000000000002", roles: [] };
const agentInput = { name: "pressure-agent", systemInstructions: "bounded", model: { provider: "disabled", model: "none" }, allowedTools: [] as string[], maximumSteps: 3, tokenBudget: 100, costBudgetMicrousd: 100, approvalPolicy: {}, outputSchema: null };
beforeEach(async () => pool.query("TRUNCATE operational_instances,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("Stage 14 deployment safety", () => {
  it("claims queued pressure concurrently without duplicate ownership", async () => {
    const agent = await new AgentRepository(pool).create(principal, agentInput); for (let index = 0; index < 12; index += 1) await new RunRepository(pool).create(principal, agent.id, `pressure-${index}`);
    const claimed: string[] = []; let round = 0;
    while (true) { const results = await Promise.all(Array.from({ length: 6 }, (_, index) => new RunRepository(pool).claimNext(`pressure-${round}-${index}`, 30))); const ids = results.filter((run): run is NonNullable<typeof run> => run !== null).map((run) => run.id); claimed.push(...ids); if (!ids.length) break; round += 1; }
    expect(claimed).toHaveLength(12); expect(new Set(claimed)).toHaveLength(12); expect(Number((await pool.query("SELECT count(*) AS count FROM worker_leases")).rows[0].count)).toBe(12);
  });

  it("makes readiness fail closed while draining or when migrations are pending", async () => {
    const auth = new DevelopmentHeaderAuthenticator();
    const pending = createApp(pool, auth, null, null, null, undefined, { isDraining: () => false, checkMigrations: async () => ({ ready: false, missing: ["future"] }) });
    expect((await pending.request("/health/ready")).status).toBe(503);
    const draining = createApp(pool, auth, null, null, null, undefined, { isDraining: () => true, checkMigrations: async () => ({ ready: true, missing: [] }) });
    expect((await draining.request("/health/ready")).status).toBe(503);
    expect((await migrationReadiness(pool)).ready).toBe(true);
  });
});
