import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { AgentRepository } from "../../src/db/repositories.js";
import type { Principal } from "../../src/db/types.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { ExecutionEngineRegistry } from "../../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../../src/execution/native-engine.js";
import { createToolRegistry } from "../../src/tools/registry.js";
import { AgentLoop } from "../../src/execution/agent-loop.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { EvaluationService } from "../../src/evaluation/evaluation-service.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";
import { createApp } from "../../src/api/app.js";
import { DeterministicTestProvider, type DeterministicReply } from "../support/deterministic-provider.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const manager: Principal = {
  tenantId: "12121212-1212-4212-8212-121212121212",
  userId: "13131313-1313-4313-8313-131313131313",
  roles: ["evaluation_manager", "evaluation_viewer"],
};
const otherTenant: Principal = {
  tenantId: "14141414-1414-4414-8414-141414141414",
  userId: "15151515-1515-4515-8515-151515151515",
  roles: ["evaluation_manager", "evaluation_viewer"],
};

beforeEach(async () => pool.query(
  "TRUNCATE evaluation_measurements,evaluation_case_runs,evaluation_executions,evaluation_cases,evaluation_suites,orchestration_waits,run_delegations,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_execution_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE",
));
afterAll(async () => pool.end());

function agentInput(provider: string, systemInstructions = "Return one canonical action.") {
  return {
    name: `evaluation-${crypto.randomUUID()}`,
    systemInstructions,
    model: {
      provider,
      model: "deterministic",
      maxOutputTokens: 100,
      timeoutMs: 1000,
      inputCostMicrousdPerMillion: 0,
      outputCostMicrousdPerMillion: 0,
      cachedCostMicrousdPerMillion: 0,
    },
    allowedTools: ["calculator"],
    maximumSteps: 10,
    tokenBudget: 1000,
    costBudgetMicrousd: 1000,
    approvalPolicy: {},
    outputSchema: null,
  };
}

function runtime(providers: DeterministicTestProvider[]) {
  const registry = new ProviderRegistry();
  for (const provider of providers) registry.register(provider);
  const engines = new ExecutionEngineRegistry();
  engines.register(new NativeExecutionEngine());
  const loop = new AgentLoop(pool, registry, engines, createToolRegistry(pool));
  return new LifecycleWorker(pool, { workerId: "evaluation-runtime", leaseSeconds: 30 }, {
    execute: (runId, workerId) => loop.execute(runId, workerId, manager.roles),
  });
}

function suiteInput(outputs: unknown[]) {
  return {
    name: "Deterministic regression suite",
    description: "Synthetic cases executed through the canonical durable run lifecycle.",
    thresholds: {},
    cases: outputs.map((output, index) => ({
      name: `case-${index + 1}`,
      goal: `Produce deterministic result ${index + 1}`,
      expectations: {
        expectedStatus: "completed" as const,
        expectedFinalOutput: output,
        maximumTokens: 100,
        maximumCostMicrousd: 100,
        maximumSteps: 2,
      },
      tags: ["deterministic", `case-${index + 1}`],
    })),
  };
}

describe("durable evaluation framework", () => {
  it("runs deterministic regression cases as ordinary persisted runs and records release-gate history", async () => {
    const replies: DeterministicReply[] = [
      { body: { type: "final_answer", output: { answer: 14 } }, usage: { inputTokens: 8, outputTokens: 3 } },
      { body: { type: "final_answer", output: { answer: 14 } }, usage: { inputTokens: 7, outputTokens: 3 } },
    ];
    const provider = new DeterministicTestProvider(replies, {}, "evaluation-test-provider");
    const agent = await new AgentRepository(pool).create(manager, agentInput(provider.id));
    const evaluations = new EvaluationService(pool, { allowDeterministicCi: true });
    const suite = await evaluations.createSuite(manager, suiteInput([{ answer: 14 }, { answer: 14 }]));
    const execution = await evaluations.createExecution(manager, suite.id as string, {
      agentId: agent.id,
      mode: "deterministic_ci",
      candidateLabel: "candidate-a",
    });

    const worker = runtime([provider]);
    expect(await worker.tick()).toBe(true);
    expect(await worker.tick()).toBe(true);
    const competingScorer = new EvaluationService(pool, { allowDeterministicCi: true });
    expect(await Promise.all([
      evaluations.advanceNext("evaluation-scorer-a"),
      competingScorer.advanceNext("evaluation-scorer-b"),
    ])).toEqual([true, true]);
    expect(await evaluations.advanceNext("evaluation-scorer")).toBe(false);

    const stored = await evaluations.getExecution(manager, execution.id as string);
    expect(stored.execution).toMatchObject({ status: "completed", passed_count: 2, failed_count: 0, gate_passed: true });
    expect(stored.caseRuns).toHaveLength(2);
    expect(stored.caseRuns.every((item) => item.status === "passed" && item.run_status === "completed")).toBe(true);
    expect(stored.measurements).toHaveLength(24);
    expect(Number((await pool.query("SELECT count(*) AS count FROM model_attempts WHERE run_id=ANY($1::uuid[])", [stored.caseRuns.map((item) => item.run_id)])).rows[0].count)).toBe(2);
    expect(Number((await pool.query("SELECT count(*) AS count FROM audit_events WHERE event_type='evaluation.case_scored'")).rows[0].count)).toBe(2);
  });

  it("keeps immutable candidate snapshots and compares provider, model, prompt, and metric history", async () => {
    const firstProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: "answer" } }], {}, "evaluation-first");
    const secondProvider = new DeterministicTestProvider([{ body: { type: "final_answer", output: "answer" } }], {}, "evaluation-second");
    const agents = new AgentRepository(pool);
    const agent = await agents.create(manager, agentInput(firstProvider.id, "First prompt."));
    const evaluations = new EvaluationService(pool, { allowDeterministicCi: true });
    const firstSuite = await evaluations.createSuite(manager, suiteInput(["answer"]));
    const first = await evaluations.createExecution(manager, firstSuite.id as string, { agentId: agent.id, mode: "deterministic_ci", candidateLabel: "first" });
    const revisedAgent = await agents.patch(manager, agent.id, agent.version, { systemInstructions: "Second prompt.", model: { ...agent.model, provider: secondProvider.id, model: "deterministic-v2" } });
    const second = await evaluations.createExecution(manager, firstSuite.id as string, { agentId: revisedAgent.id, mode: "deterministic_ci", candidateLabel: "second" });

    const worker = runtime([firstProvider, secondProvider]);
    expect(await worker.tick()).toBe(true);
    expect(await evaluations.advanceNext("evaluation-scorer")).toBe(true);
    expect(await worker.tick()).toBe(true);
    expect(await evaluations.advanceNext("evaluation-scorer")).toBe(true);

    expect(JSON.stringify(firstProvider.requests[0]?.messages[0]?.content)).toContain("First prompt.");
    expect(JSON.stringify(secondProvider.requests[0]?.messages[0]?.content)).toContain("Second prompt.");
    const comparison = await evaluations.compare(manager, first.id as string, second.id as string);
    expect(comparison.left).toMatchObject({ provider: firstProvider.id, model: "deterministic", gatePassed: true });
    expect(comparison.right).toMatchObject({ provider: secondProvider.id, model: "deterministic-v2", gatePassed: true });
    expect((comparison.left as Record<string, unknown>).promptHash).not.toBe((comparison.right as Record<string, unknown>).promptHash);
    expect(await evaluations.listExecutions(manager, { suiteId: firstSuite.id, limit: 10 })).toHaveLength(2);
  });

  it("records a completed but failed release gate without reporting the evaluated run as successful", async () => {
    const provider = new DeterministicTestProvider([{ body: { type: "final_answer", output: "unsafe marker" } }], {}, "evaluation-failing");
    const agent = await new AgentRepository(pool).create(manager, agentInput(provider.id));
    const evaluations = new EvaluationService(pool, { allowDeterministicCi: true });
    const suite = await evaluations.createSuite(manager, {
      ...suiteInput(["expected"]),
      cases: [{ name: "failure", goal: "Return a safe value", expectations: { expectedFinalOutput: "expected", forbiddenOutputText: ["unsafe marker"] }, tags: ["safety"] }],
    });
    const execution = await evaluations.createExecution(manager, suite.id as string, { agentId: agent.id, mode: "deterministic_ci", candidateLabel: "failing" });
    expect(await runtime([provider]).tick()).toBe(true);
    expect(await evaluations.advanceNext("evaluation-scorer")).toBe(true);
    const stored = await evaluations.getExecution(manager, execution.id as string);
    expect(stored.execution).toMatchObject({ status: "completed", passed_count: 0, failed_count: 1, gate_passed: false });
    expect(stored.caseRuns[0]).toMatchObject({ status: "failed", run_status: "completed" });
  });

  it("versions datasets without allowing historical expectations or thresholds to be rewritten", async () => {
    const evaluations = new EvaluationService(pool, { allowDeterministicCi: true });
    const original = await evaluations.createSuite(manager, suiteInput(["answer"]));
    const revised = await evaluations.reviseSuite(manager, original.id as string, {
      ...suiteInput(["answer"]),
      name: "Deterministic regression suite revision",
    });
    expect(revised).toMatchObject({ logical_id: original.logical_id, version: 2, supersedes_id: original.id, status: "active" });
    expect((await evaluations.getSuite(manager, original.id as string)).suite.status).toBe("archived");
    await expect(evaluations.createExecution(manager, original.id as string, {
      agentId: "16161616-1616-4616-8616-161616161616",
      mode: "deterministic_ci",
      candidateLabel: "invalid-old-suite",
    })).rejects.toThrow("Active evaluation suite not found");
    await expect(pool.query("UPDATE evaluation_suites SET name='rewritten' WHERE id=$1", [original.id])).rejects.toThrow(/immutable/);
    const caseId = (await pool.query("SELECT id FROM evaluation_cases WHERE suite_id=$1", [original.id])).rows[0].id;
    await expect(pool.query("UPDATE evaluation_cases SET name='rewritten' WHERE id=$1", [caseId])).rejects.toThrow(/append-only/);
  });

  it("enforces API roles, tenant boundaries, and test-only deterministic execution mode", async () => {
    const provider = new DeterministicTestProvider([], {}, "evaluation-api-provider");
    const agent = await new AgentRepository(pool).create(manager, agentInput(provider.id));
    const app = createApp(pool, new DevelopmentHeaderAuthenticator());
    const headers = { "content-type": "application/json", "x-tenant-id": manager.tenantId, "x-user-id": manager.userId, "x-roles": "evaluation_manager,evaluation_viewer" };
    const denied = await app.request("/evaluation-suites", { method: "POST", headers: { ...headers, "x-roles": "evaluation_viewer" }, body: JSON.stringify(suiteInput(["answer"])) });
    expect(denied.status).toBe(403);
    const createdResponse = await app.request("/evaluation-suites", { method: "POST", headers, body: JSON.stringify(suiteInput(["answer"])) });
    expect(createdResponse.status).toBe(201);
    const suite = await createdResponse.json() as { id: string };
    const deterministic = await app.request(`/evaluation-suites/${suite.id}/executions`, { method: "POST", headers, body: JSON.stringify({ agentId: agent.id, mode: "deterministic_ci", candidateLabel: "forbidden-production-mode" }) });
    expect(deterministic.status).toBe(409);
    const modelDependent = await app.request(`/evaluation-suites/${suite.id}/executions`, { method: "POST", headers, body: JSON.stringify({ agentId: agent.id, mode: "model_dependent", candidateLabel: "explicit-provider-run" }) });
    expect(modelDependent.status).toBe(201);
    const otherHeaders = { ...headers, "x-tenant-id": otherTenant.tenantId, "x-user-id": otherTenant.userId };
    expect((await app.request(`/evaluation-suites/${suite.id}`, { headers: otherHeaders })).status).toBe(404);
  });
});
