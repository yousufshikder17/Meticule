import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import { OperationalService } from "../../src/observability/operational-service.js";
import type { Principal } from "../../src/db/types.js";
import type { StructuredLogger } from "../../src/observability/logger.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import type { ModelProvider } from "../../src/models/model-provider.js";

const { Pool } = pg; const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const operator: Principal = { tenantId: "f1000000-0000-4000-8000-000000000001", userId: "f1000000-0000-4000-8000-000000000002", roles: ["system_operator"] };
const other: Principal = { tenantId: "f2000000-0000-4000-8000-000000000001", userId: "f2000000-0000-4000-8000-000000000002", roles: ["system_operator"] };
const headers = (principal: Principal) => ({ "x-tenant-id": principal.tenantId, "x-user-id": principal.userId, "x-roles": principal.roles.join(",") });
const agentInput = { name: "observable-agent", systemInstructions: "Return a bounded action", model: { provider: "disabled", model: "none" }, allowedTools: ["calculator"], maximumSteps: 3, tokenBudget: 100, costBudgetMicrousd: 100, approvalPolicy: {}, outputSchema: null };

beforeEach(async () => pool.query("TRUNCATE operational_instances,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("Stage 13 operational controls", () => {
  it("reports tenant-scoped queue, lease, usage, failures, and persisted worker health", async () => {
    const a1 = await new AgentRepository(pool).create(operator, agentInput); const a2 = await new AgentRepository(pool).create(other, agentInput);
    const run = await new RunRepository(pool).create(operator, a1.id, "observe this run"); await new RunRepository(pool).create(other, a2.id, "hidden run");
    await pool.query("INSERT INTO usage_records(tenant_id,run_id,provider,model,input_tokens,output_tokens,cost_microusd) VALUES($1,$2,'test','model',5,2,9)", [operator.tenantId, run.id]);
    const service = new OperationalService(pool); await service.heartbeat("worker-observe", "worker", { pool: "default" });
    const result = await service.metrics(operator) as any;
    expect(result.queueDepth.queued).toBe(1); expect(result.usageLast24Hours.cost_microusd).toBe("9"); expect(result.workers[0].healthy).toBe(true); expect(result.workers[0].metadata).toEqual({ pool: "default" });
  });

  it("requires system_operator and emits a fresh correlation ID without logging credentials", async () => {
    const records: Array<Record<string, unknown>> = []; const capture: StructuredLogger = { log: (level, event, fields = {}) => records.push({ level, event, ...fields }) };
    const app = createApp(pool, new DevelopmentHeaderAuthenticator(), null, null, null, capture);
    const denied = await app.request("/operations/metrics", { headers: headers({ ...operator, roles: [] }) }); expect(denied.status).toBe(403);
    const accepted = await app.request("/operations/metrics", { headers: { ...headers(operator), authorization: "Bearer must-not-be-logged" } });
    expect(accepted.status).toBe(200); expect(accepted.headers.get("x-correlation-id")).toMatch(/^[0-9a-f-]{36}$/); expect(JSON.stringify(records)).not.toContain("must-not-be-logged");
  });

  it("reports configured provider health and redacts credentials from details", async () => {
    const provider: ModelProvider = { id: "health-test", capabilities: () => ({ toolCalling: false, structuredOutput: false, streaming: false, vision: false, tokenUsage: false, contextWindow: null, nativeIdempotency: false, execution: "local" }), invoke: async () => { throw new Error("unused"); }, healthCheck: async () => ({ healthy: false, details: "Bearer synthetic-health-secret" }) };
    const registry = new ProviderRegistry(); registry.register(provider);
    const app = createApp(pool, new DevelopmentHeaderAuthenticator(), null, null, registry);
    const response = await app.request("/operations/providers/health", { headers: headers(operator) }); const body = await response.json() as Array<Record<string, unknown>>;
    expect(response.status).toBe(200); expect(body[0]).toMatchObject({ providerId: "health-test", healthy: false, details: "Bearer [REDACTED]" }); expect(JSON.stringify(body)).not.toContain("synthetic-health-secret");
  });
});
