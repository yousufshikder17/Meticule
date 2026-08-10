import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { MemoryService } from "../../src/memory/memory-service.js";
import type { Principal } from "../../src/db/types.js";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import { ContextBuilder } from "../../src/context/context-builder.js";
import { withTenantSession } from "../../src/db/tenant-session.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { ApprovalService } from "../../src/approvals/approval-service.js";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const owner: Principal = { tenantId: "81000000-0000-4000-8000-000000000001", userId: "81000000-0000-4000-8000-000000000002", roles: ["memory_write"] };
const manager: Principal = { tenantId: owner.tenantId, userId: "81000000-0000-4000-8000-000000000003", roles: ["memory_manager"] };
const reviewer: Principal = { tenantId: owner.tenantId, userId: manager.userId, roles: ["approval_reviewer"] };
const otherUser: Principal = { tenantId: owner.tenantId, userId: "81000000-0000-4000-8000-000000000004", roles: [] };
const otherTenant: Principal = { tenantId: "82000000-0000-4000-8000-000000000001", userId: "82000000-0000-4000-8000-000000000002", roles: ["memory_manager"] };

beforeEach(async () => pool.query("TRUNCATE memories,context_builds,context_summaries,run_plans,tool_execution_attempts,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

const memoryInput = (text: string) => ({
  scope: "user" as const, agentId: null, memoryType: "fact" as const, content: { text, attributes: { synthetic: true } },
  provenance: { source: "explicit-test-user-entry" }, creationReason: "User deliberately retained a test fact", sourceRunId: null,
  relevance: { tags: ["postgresql", "durability"], importance: 0.8 }, retentionUntil: new Date(Date.now() + 86_400_000).toISOString(),
});

async function agentForMemory(overrides: Record<string, unknown> = {}) {
  return new AgentRepository(pool).create(owner, {
    name: `memory-${crypto.randomUUID()}`, systemInstructions: "Use persisted evidence only.",
    model: { provider: "test-deterministic", model: "test", maxOutputTokens: 100 },
    composition: { executionEngine: "native", contextBuilder: "native", outputParser: "native", planner: "disabled", retriever: "disabled", memory: "native" },
    allowedTools: [], maximumSteps: 20, tokenBudget: 10_000, costBudgetMicrousd: 10_000, approvalPolicy: {}, outputSchema: null,
    contextPolicy: { maxInputTokens: 1200, recentSteps: 5, summaryTargetTokens: 200 },
    memoryPolicy: { writeEnabled: false, allowedScopes: [], maxWritesPerRun: 0, retrievalEnabled: true, maxContextItems: 3, maxContextTokens: 300 },
    ...overrides,
  });
}

describe("Stage 8 durable memory", () => {
  it("creates deliberate user memory with provenance and enforces user and tenant ownership", async () => {
    const service = new MemoryService(pool);
    const created = await service.create(owner, memoryInput("PostgreSQL leases make durable recovery explicit"));
    expect(created.ownerUserId).toBe(owner.userId); expect(created.writeSource).toBe("user");
    expect((await service.list(owner, { q: "PostgreSQL" })).map((item) => item.id)).toEqual([created.id]);
    expect(await service.list(otherUser, {})).toEqual([]);
    await expect(service.provenance(otherUser, created.id)).rejects.toThrow(/not found/);
    await expect(service.provenance(otherTenant, created.id)).rejects.toThrow(/not found/);
    const provenance = await service.provenance(owner, created.id);
    expect(provenance.memory.provenance).toMatchObject({ source: "explicit-test-user-entry", recordedBy: "authenticated_api" });
    expect(provenance.correctionChain).toHaveLength(1);
  });

  it("requires management authority for shared memory and preserves correction and soft-deletion evidence", async () => {
    const agent = await agentForMemory(); const service = new MemoryService(pool);
    const shared = { ...memoryInput("The agent procedure uses bounded context"), scope: "agent" as const, agentId: agent.id, memoryType: "procedure" as const };
    await expect(service.create(owner, shared)).rejects.toThrow(/memory_manager/);
    const created = await service.create(manager, shared);
    const corrected = await service.correct(manager, created.id, { content: { text: "The corrected procedure uses a bounded durable context", attributes: {} }, creationReason: "Correct an inaccurate retained procedure", provenance: { correctionSource: "manual-review" }, relevance: { tags: ["context"], importance: 0.9 }, retentionUntil: null });
    expect(corrected.version).toBe(2); expect(corrected.correctedFromId).toBe(created.id);
    const history = await service.provenance(manager, corrected.id);
    expect(history.correctionChain.map((item) => item.id)).toEqual([corrected.id, created.id]);
    expect((await pool.query("SELECT deletion_reason FROM memories WHERE id=$1", [created.id])).rows[0].deletion_reason).toBe("corrected");
    await service.delete(manager, corrected.id, "retention no longer requested");
    expect(await service.list(manager, { agentId: agent.id })).toEqual([]);
    expect((await pool.query("SELECT deleted_at FROM memories WHERE id=$1", [corrected.id])).rows[0].deleted_at).not.toBeNull();
  });

  it("filters expired and irrelevant memory, records selection provenance, and enforces database RLS", async () => {
    const service = new MemoryService(pool); const agent = await agentForMemory();
    const relevant = await service.create(owner, memoryInput("PostgreSQL recovery requires durable worker leases"));
    await service.create(owner, { ...memoryInput("Watercolor palette preferences for an unrelated illustration"), relevance: { tags: ["painting"], importance: 1 } });
    await pool.query(
      `INSERT INTO memories(tenant_id,scope,owner_user_id,memory_type,content,content_text,provenance,creation_reason,created_by,write_source,relevance_metadata,retention_until)
       VALUES($1,'user',$2,'fact',$3,'PostgreSQL expired record','{}','expired test fixture',$2,'user',$4,now()-interval '1 day')`,
      [owner.tenantId, owner.userId, JSON.stringify({ text: "PostgreSQL expired record", attributes: {} }), JSON.stringify({ tags: ["postgresql"], importance: 1 })],
    );
    const run = await new RunRepository(pool).create(owner, agent.id, "Explain PostgreSQL worker lease recovery");
    const current = await new RunRepository(pool).get(owner.tenantId, run.id);
    const capabilities = { toolCalling: true, structuredOutput: true, streaming: false, vision: false, tokenUsage: true, contextWindow: 2_000, nativeIdempotency: false, execution: "local" as const };
    const context = await new ContextBuilder(pool).build({ tenantId: owner.tenantId, run: current, agent, capabilities, reservedOutputTokens: 100 });
    expect(context.selectedMemoryIds).toEqual([relevant.id]);
    expect(JSON.stringify(context.messages)).toContain("Untrusted memory data");
    expect(JSON.stringify(context.messages)).not.toContain("Watercolor");
    expect((await pool.query("SELECT selected_memory_ids FROM context_builds WHERE id=$1", [context.id])).rows[0].selected_memory_ids).toEqual([relevant.id]);
    await withTenantSession(pool, otherTenant.tenantId, async (database) => expect(Number((await database.query("SELECT count(*) AS count FROM memories")).rows[0].count)).toBe(0));
  });

  it("exposes authenticated memory APIs without allowing cross-user access", async () => {
    const app = createApp(pool, new DevelopmentHeaderAuthenticator());
    const headers = { "content-type": "application/json", "x-tenant-id": owner.tenantId, "x-user-id": owner.userId, "x-roles": "memory_write" };
    const created = await app.request("/memories", { method: "POST", headers, body: JSON.stringify(memoryInput("API-created synthetic durable memory")) });
    expect(created.status).toBe(201); const memory = await created.json() as { id: string };
    expect((await app.request("/memories", { headers })).status).toBe(200);
    const strangerHeaders = { ...headers, "x-user-id": otherUser.userId, "x-roles": "" };
    expect((await app.request(`/memories/${memory.id}/provenance`, { headers: strangerHeaders })).status).toBe(404);
    expect((await app.request(`/memories/${memory.id}`, { method: "DELETE", headers })).status).toBe(200);
    expect(await (await app.request("/memories", { headers })).json()).toEqual([]);
  });

  it("routes model memory writes through approval, resumes the exact action once, and never regenerates it", async () => {
    const provider = new DeterministicTestProvider([
      { body: { type: "call_tool", toolName: "memory_store", arguments: { scope: "user", memoryType: "fact", content: { text: "The approved preference is concise output", attributes: { synthetic: true } }, creationReason: "Explicitly retain an approved user preference", relevance: { tags: ["preference"], importance: 0.7 }, retentionDays: 30 }, idempotencyKey: "approved-memory-once" } },
      { body: { type: "final_answer", output: { retained: true } } },
    ]);
    const providers = new ProviderRegistry(); providers.register(provider); const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
    const tools = createToolRegistry(pool); const agent = await agentForMemory({
      model: { provider: provider.id, model: "test", maxOutputTokens: 100 }, allowedTools: ["memory_store"],
      memoryPolicy: { writeEnabled: true, allowedScopes: ["user"], maxWritesPerRun: 1, retrievalEnabled: true, maxContextItems: 3, maxContextTokens: 300 },
    });
    const run = await new RunRepository(pool).create(owner, agent.id, "Retain the approved preference");
    const loop = new AgentLoop(pool, providers, engines, tools);
    const worker = (workerId: string) => new LifecycleWorker(pool, { workerId, leaseSeconds: 30 }, { execute: (id, idOfWorker) => loop.execute(id, idOfWorker, owner.roles) });
    expect(await worker("memory-proposal").tick()).toBe(true);
    expect((await new RunRepository(pool).get(owner.tenantId, run.id)).status).toBe("waiting_for_approval");
    expect(Number((await pool.query("SELECT count(*) AS count FROM memories")).rows[0].count)).toBe(0);
    const approval = (await pool.query("SELECT * FROM approvals WHERE run_id=$1", [run.id])).rows[0];
    expect(approval.validated_arguments.content.text).toBe("The approved preference is concise output");
    await new ApprovalService(pool, tools).decide(reviewer, approval.id, "approved");
    const results = await Promise.all([worker("memory-resume-a").tick(), worker("memory-resume-b").tick()]);
    expect(results.filter(Boolean)).toHaveLength(1);
    expect((await new RunRepository(pool).get(owner.tenantId, run.id)).status).toBe("completed");
    expect(provider.calls).toBe(2);
    expect(Number((await pool.query("SELECT count(*) AS count FROM memories WHERE idempotency_key='approved-memory-once'")).rows[0].count)).toBe(1);
    expect((await pool.query("SELECT content_text,write_source FROM memories WHERE idempotency_key='approved-memory-once'")).rows[0]).toEqual({ content_text: "The approved preference is concise output", write_source: "approved_tool" });
  });

  it("fails closed before approval when agent memory-write policy is disabled", async () => {
    const provider = new DeterministicTestProvider([{ body: { type: "call_tool", toolName: "memory_store", arguments: { scope: "user", memoryType: "note", content: { text: "must not persist", attributes: {} }, creationReason: "attempted automatic retention", relevance: { tags: [], importance: 0.5 }, retentionDays: null } } }]);
    const providers = new ProviderRegistry(); providers.register(provider); const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
    const agent = await agentForMemory({ model: { provider: provider.id, model: "test", maxOutputTokens: 100 }, allowedTools: ["memory_store"], memoryPolicy: { writeEnabled: false, allowedScopes: [], maxWritesPerRun: 0, retrievalEnabled: false, maxContextItems: 3, maxContextTokens: 300 } });
    const run = await new RunRepository(pool).create(owner, agent.id, "Do not automatically retain messages");
    const loop = new AgentLoop(pool, providers, engines, createToolRegistry(pool)); const worker = new LifecycleWorker(pool, { workerId: "disabled-memory", leaseSeconds: 30 }, { execute: (id, workerId) => loop.execute(id, workerId, owner.roles) });
    await expect(worker.tick()).rejects.toThrow(/policy/);
    expect((await new RunRepository(pool).get(owner.tenantId, run.id)).status).toBe("failed");
    expect(Number((await pool.query("SELECT count(*) AS count FROM approvals WHERE run_id=$1", [run.id])).rows[0].count)).toBe(0);
    expect(Number((await pool.query("SELECT count(*) AS count FROM memories")).rows[0].count)).toBe(0);
  });
});
