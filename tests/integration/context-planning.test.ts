import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { AgentRepository, RunRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { PlanService } from "../../src/planning/plan-service.js";
import { ContextBuilder } from "../../src/context/context-builder.js";
import { CheckpointService } from "../../src/context/checkpoint-service.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";
import { withTenantSession } from "../../src/db/tenant-session.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const principal: Principal = { tenantId: "77777777-7777-4777-8777-777777777777", userId: "88888888-8888-4888-8888-888888888888", roles: [] };

async function createAgent(options: { planner?: "disabled" | "native"; maxInputTokens?: number; recentSteps?: number; allowedTools?: string[] } = {}) {
  return new AgentRepository(pool).create(principal, {
    name: `context-${crypto.randomUUID()}`, systemInstructions: "Preserve every active constraint.",
    model: { provider: "test-deterministic", model: "test", maxOutputTokens: 100 },
    composition: { executionEngine: "native", contextBuilder: "native", outputParser: "native", planner: options.planner ?? "native", retriever: "disabled", memory: "disabled" },
    allowedTools: options.allowedTools ?? [], maximumSteps: 20, tokenBudget: 10_000, costBudgetMicrousd: 10_000, approvalPolicy: {}, outputSchema: null,
    contextPolicy: { maxInputTokens: options.maxInputTokens ?? 800, recentSteps: options.recentSteps ?? 2, summaryTargetTokens: 120 },
  });
}

async function runningRun(workerId = "planning-worker") {
  const agent = await createAgent(); const runs = new RunRepository(pool);
  const run = await runs.create(principal, agent.id, "Complete the durable planning objective without losing constraints");
  expect((await runs.claimNext(workerId, 30))?.id).toBe(run.id);
  await runs.workerTransition(run.id, workerId, "running");
  return { agent, run: await runs.get(principal.tenantId, run.id), runs, workerId };
}

beforeEach(async () => pool.query("TRUNCATE context_builds,context_summaries,run_plans,tool_execution_attempts,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"));
afterAll(async () => pool.end());

describe("Stage 7 durable planning and context", () => {
  it("persists validated revisions, evidence, dependency state, and restart-safe active state", async () => {
    const { run, workerId } = await runningRun();
    const evidence = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,output,finished_at) VALUES($1,$2,1,'tool','succeeded','task-evidence',$3,now()) RETURNING id", [principal.tenantId, run.id, JSON.stringify({ ok: true })]);
    const source = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,output,finished_at) VALUES($1,$2,2,'model','succeeded','plan-source',$3,now()) RETURNING id", [principal.tenantId, run.id, JSON.stringify({ type: "update_plan" })]);
    await pool.query("UPDATE runs SET current_step=2 WHERE id=$1", [run.id]);
    const plans = new PlanService(pool);
    const first = await plans.revise({ principal, runId: run.id, workerId, stepId: source.rows[0].id, plan: { objective: "ship", tasks: [{ taskId: "a", objective: "first" }, { taskId: "b", objective: "second", dependencies: ["a"] }] } });
    expect(first.plan.tasks.map((task) => task.status)).toEqual(["ready", "pending"]);
    const second = await plans.revise({ principal, runId: run.id, workerId, stepId: source.rows[0].id, plan: { objective: "ship", tasks: [{ taskId: "a", objective: "first", status: "completed", result: { ok: true }, evidenceStepId: evidence.rows[0].id }, { taskId: "b", objective: "second", dependencies: ["a"] }] } });
    expect(second.version).toBe(2); expect(second.plan.tasks[1].status).toBe("ready");
    expect((await new PlanService(pool).active(principal.tenantId, run.id))?.plan).toEqual(second.plan);
    expect((await pool.query("SELECT status FROM run_plans WHERE run_id=$1 ORDER BY version", [run.id])).rows).toEqual([{ status: "superseded" }, { status: "active" }]);
    await expect(plans.revise({ principal, runId: run.id, workerId, stepId: source.rows[0].id, plan: { objective: "ship", tasks: [{ taskId: "a", objective: "redefined" }] } })).rejects.toThrow(/Completed task/);
  });

  it("constructs bounded summarized context, retains trace provenance, and restores equivalently after restart", async () => {
    const agent = await createAgent({ maxInputTokens: 700, recentSteps: 2 }); const runs = new RunRepository(pool);
    const created = await runs.create(principal, agent.id, "Investigate alpha failures while preserving the safety constraint");
    for (let sequence = 1; sequence <= 8; sequence += 1) await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,output,finished_at) VALUES($1,$2,$3,'tool','succeeded',$4,$5,now())", [principal.tenantId, created.id, sequence, `history-${sequence}`, JSON.stringify({ observation: `alpha-${sequence}-${"x".repeat(180)}` })]);
    await pool.query("UPDATE runs SET current_step=8 WHERE id=$1", [created.id]);
    const run = await runs.get(principal.tenantId, created.id);
    const capabilities = { toolCalling: true, structuredOutput: true, streaming: false, vision: false, tokenUsage: true, contextWindow: 1_000, nativeIdempotency: false, execution: "local" as const };
    const first = await new ContextBuilder(pool).build({ tenantId: principal.tenantId, run, agent, capabilities, reservedOutputTokens: 100 });
    const second = await new ContextBuilder(pool).build({ tenantId: principal.tenantId, run, agent, capabilities, reservedOutputTokens: 100 });
    expect(first.tokenEstimate).toBeLessThanOrEqual(first.tokenBudget); expect(first.summaryId).not.toBeNull(); expect(first.omittedStepSequences.length).toBeGreaterThan(0);
    expect(first.messages).toEqual(second.messages);
    expect(JSON.stringify(first.messages)).toContain("Unresolved goal"); expect(JSON.stringify(first.messages)).toContain("Durable summary");
    expect(Number((await pool.query("SELECT count(*) AS count FROM steps WHERE run_id=$1", [run.id])).rows[0].count)).toBe(8);
    expect((await pool.query("SELECT provenance FROM context_summaries WHERE id=$1", [first.summaryId])).rows[0].provenance.originalTraceRetained).toBe(true);
    expect((await pool.query("SELECT provenance FROM context_builds WHERE id=$1", [second.id])).rows[0].provenance.baseCheckpointId).toBe(first.checkpointId);
    expect(Number((await pool.query("SELECT count(*) AS count FROM audit_events WHERE run_id=$1 AND event_type='context.summarized'", [run.id])).rows[0].count)).toBe(1);
    const restored = await new CheckpointService(pool).restoreLatest(principal.tenantId, run.id); expect(restored?.state.currentStep).toBe(8);
    await pool.query("UPDATE checkpoints SET state=jsonb_set(state,'{currentStep}','99') WHERE id=$1", [restored!.id]);
    await expect(new CheckpointService(pool).restoreLatest(principal.tenantId, run.id)).rejects.toThrow(/checksum/);
    await withTenantSession(pool, "99999999-9999-4999-8999-999999999999", async (database) => {
      for (const table of ["context_summaries", "context_builds", "run_plans"]) expect(Number((await database.query(`SELECT count(*) AS count FROM ${table}`)).rows[0].count)).toBe(0);
    });
  });

  it("runs a native plan revision and feeds it into the next model turn before completion", async () => {
    let runId = "";
    const provider = new DeterministicTestProvider([
      { body: { type: "update_plan", plan: { objective: "calculate", tasks: [{ taskId: "decide", objective: "decide the answer" }] } } },
      { toolCalls: [{ id: "planned-calculation", name: "calculator", arguments: { expression: "6*7" } }] },
      { body: async () => {
        const evidence = await pool.query("SELECT id FROM steps WHERE run_id=$1 AND kind='tool' AND status='succeeded' ORDER BY sequence DESC LIMIT 1", [runId]);
        return { type: "update_plan", plan: { objective: "calculate", tasks: [{ taskId: "decide", objective: "decide the answer", status: "completed", result: { answer: 42 }, evidenceStepId: evidence.rows[0].id }] } };
      } },
      { body: { type: "final_answer", output: { answer: 42 } } },
    ]);
    const providers = new ProviderRegistry(); providers.register(provider);
    const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
    const tools = createToolRegistry(pool); const agent = await createAgent({ allowedTools: ["calculator"] });
    const run = await new RunRepository(pool).create(principal, agent.id, "Produce 42 through a persisted plan");
    runId = run.id;
    const loop = new AgentLoop(pool, providers, engines, tools);
    const worker = new LifecycleWorker(pool, { workerId: "plan-loop", leaseSeconds: 30 }, { execute: (id, workerId) => loop.execute(id, workerId) });
    expect(await worker.tick()).toBe(true);
    expect((await new RunRepository(pool).get(principal.tenantId, run.id)).status).toBe("completed");
    expect(provider.calls).toBe(4); expect(JSON.stringify(provider.requests[1].messages)).toContain("Active persisted plan revision 1");
    expect(Number((await pool.query("SELECT count(*) AS count FROM run_plans WHERE run_id=$1 AND status='active'", [run.id])).rows[0].count)).toBe(1);
    expect(Number((await pool.query("SELECT count(*) AS count FROM context_builds WHERE run_id=$1", [run.id])).rows[0].count)).toBe(4);
  });

  it("never completes a run while an active plan still has required work", async () => {
    const provider = new DeterministicTestProvider([{ body: { type: "update_plan", plan: { objective: "unfinished", tasks: [{ taskId: "required", objective: "must finish" }] } } }, { body: { type: "final_answer", output: "premature" } }]);
    const providers = new ProviderRegistry(); providers.register(provider); const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
    const agent = await createAgent(); const run = await new RunRepository(pool).create(principal, agent.id, "Do not report premature success");
    const loop = new AgentLoop(pool, providers, engines, createToolRegistry(pool)); const worker = new LifecycleWorker(pool, { workerId: "incomplete-plan", leaseSeconds: 30 }, { execute: (id, workerId) => loop.execute(id, workerId) });
    await expect(worker.tick()).rejects.toThrow(/remain incomplete/);
    const stored = await new RunRepository(pool).get(principal.tenantId, run.id); expect(stored.status).toBe("failed"); expect(stored.finalOutput).toBeNull();
  });

  it("refuses plan revision after durable cancellation", async () => {
    const { run, runs, workerId } = await runningRun("cancel-plan");
    const step = await pool.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,output,finished_at) VALUES($1,$2,1,'model','succeeded','cancel-source','{}',now()) RETURNING id", [principal.tenantId, run.id]);
    await runs.requestCancellation(principal, run.id);
    await expect(new PlanService(pool).revise({ principal, runId: run.id, workerId, stepId: step.rows[0].id, plan: { objective: "x", tasks: [{ taskId: "a", objective: "a" }] } })).rejects.toThrow(/Cancellation/);
    expect(Number((await pool.query("SELECT count(*) AS count FROM run_plans WHERE run_id=$1", [run.id])).rows[0].count)).toBe(0);
  });

  it("fails closed instead of silently dropping oversized active constraints", async () => {
    const initialAgent = await createAgent({ maxInputTokens: 256 });
    await pool.query("UPDATE agents SET system_instructions=$1 WHERE id=$2", ["critical-constraint ".repeat(400), initialAgent.id]);
    const agent = await new AgentRepository(pool).get(principal.tenantId, initialAgent.id);
    const run = await new RunRepository(pool).create(principal, agent.id, "retain this goal");
    const capabilities = { toolCalling: true, structuredOutput: true, streaming: false, vision: false, tokenUsage: true, contextWindow: 1_000, nativeIdempotency: false, execution: "local" as const };
    await expect(new ContextBuilder(pool).build({ tenantId: principal.tenantId, run, agent, capabilities, reservedOutputTokens: 100 })).rejects.toThrow(/exceed the context budget/);
    expect(Number((await pool.query("SELECT count(*) AS count FROM context_builds WHERE run_id=$1", [run.id])).rows[0].count)).toBe(0);
  });
});
