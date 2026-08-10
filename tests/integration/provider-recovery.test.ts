import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { ModelProviderError } from "../../src/models/model-errors.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const principal: Principal = { tenantId: "d1111111-1111-4111-8111-111111111111", userId: "d2222222-2222-4222-8222-222222222222", roles: [] };
beforeEach(async () => pool.query("TRUNCATE tool_execution_attempts,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

async function setup(primary: DeterministicTestProvider, fallbacks: DeterministicTestProvider[] = [], options: { retryAttempts?: number; allowedTools?: string[] } = {}) {
  const providers = new ProviderRegistry(); providers.register(primary); for (const fallback of fallbacks) providers.register(fallback);
  const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
  const loop = new AgentLoop(pool, providers, engines, createToolRegistry(pool));
  const target = (provider: DeterministicTestProvider) => ({ provider: provider.id, model: "deterministic", maxOutputTokens: 100, timeoutMs: 1000, inputCostMicrousdPerMillion: 1_000_000, outputCostMicrousdPerMillion: 1_000_000, cachedCostMicrousdPerMillion: 0 });
  const agent = await new AgentRepository(pool).create(principal, { name: `provider-recovery-${crypto.randomUUID()}`, systemInstructions: "Return one action.", model: { ...target(primary), retryPolicy: { maxAttempts: options.retryAttempts ?? 1, baseDelayMs: 1, maxDelayMs: 2 }, fallbacks: fallbacks.map(target) }, allowedTools: options.allowedTools ?? [], maximumSteps: 20, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null });
  const run = await new RunRepository(pool).create(principal, agent.id, "provider recovery");
  const worker = new LifecycleWorker(pool, { workerId: "provider-worker", leaseSeconds: 30 }, { execute: (runId, workerId) => loop.execute(runId, workerId, principal.roles) });
  return { run, worker };
}

describe("Stage 5 provider attempt orchestration", () => {
  it("retries transient errors and persists every attempt", async () => {
    const provider = new DeterministicTestProvider([{ error: new ModelProviderError("unavailable", "temporary", true) }, { body: { type: "final_answer", output: "recovered" } }], {}, "retry-provider");
    const { run, worker } = await setup(provider, [], { retryAttempts: 2 }); expect(await worker.tick()).toBe(true);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("completed");
    expect(provider.calls).toBe(2);
    expect((await pool.query("SELECT attempt_number,status FROM model_attempts WHERE run_id=$1 ORDER BY attempt_number", [run.id])).rows).toEqual([{ attempt_number: 1, status: "failed" }, { attempt_number: 2, status: "succeeded" }]);
  });

  it("does not retry permanent provider errors", async () => {
    const provider = new DeterministicTestProvider([{ error: new ModelProviderError("authentication", "invalid credential", false) }], {}, "permanent-provider");
    const { run, worker } = await setup(provider, [], { retryAttempts: 3 }); await expect(worker.tick()).rejects.toThrow(/invalid credential/);
    expect(provider.calls).toBe(1); expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
  });

  it("uses only an explicit capable fallback and accounts cumulative billable usage", async () => {
    const primaryError = new ModelProviderError("rate_limited", "limited", true, {}, { inputTokens: 10, outputTokens: 0, cachedTokens: 0 });
    const primary = new DeterministicTestProvider([{ error: primaryError }], {}, "primary-provider");
    const fallback = new DeterministicTestProvider([{ body: { type: "final_answer", output: "fallback" }, usage: { inputTokens: 10, outputTokens: 5 } }], {}, "fallback-provider");
    const { run, worker } = await setup(primary, [fallback]); expect(await worker.tick()).toBe(true);
    const stored = await new RunRepository(pool).get(principal.tenantId, run.id);
    expect(stored.inputTokens).toBe(20); expect(stored.outputTokens).toBe(5); expect(stored.costMicrousd).toBe(25);
    expect((await pool.query("SELECT provider_id,status FROM model_attempts WHERE run_id=$1 ORDER BY attempt_number", [run.id])).rows).toEqual([{ provider_id: "primary-provider", status: "failed" }, { provider_id: "fallback-provider", status: "succeeded" }]);
  });

  it("rejects a capability-incompatible fallback before provider invocation", async () => {
    const primary = new DeterministicTestProvider([{ error: new ModelProviderError("unavailable", "offline", true) }], {}, "tool-primary");
    const fallback = new DeterministicTestProvider([{ body: { type: "final_answer", output: "invalid" } }], { toolCalling: false }, "non-tool-fallback");
    const { run, worker } = await setup(primary, [fallback], { allowedTools: ["calculator"] }); await expect(worker.tick()).rejects.toThrow(/lacks required tool-calling/);
    expect(primary.calls).toBe(0); expect(fallback.calls).toBe(0); expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("failed");
  });

  it("honors cancellation between provider attempts", async () => {
    let runId = ""; const provider = new DeterministicTestProvider([{ beforeReturn: async () => { await new RunRepository(pool).requestCancellation(principal, runId); }, error: new ModelProviderError("unavailable", "temporary", true) }, { body: { type: "final_answer", output: "must not run" } }], {}, "cancel-retry-provider");
    const setupResult = await setup(provider, [], { retryAttempts: 2 }); runId = setupResult.run.id;
    await expect(setupResult.worker.tick()).rejects.toThrow(/Cancellation/); expect(provider.calls).toBe(1);
    expect((await new RunRepository(pool).get(principal.tenantId, runId)).status).toBe("cancelled");
    expect((await pool.query("SELECT status FROM steps WHERE run_id=$1 AND kind='model'", [runId])).rows[0].status).toBe("cancelled");
  });
});
