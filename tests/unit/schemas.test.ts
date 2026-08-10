import { describe, expect, it } from "vitest";
import { CreateAgentSchema, CreateRunSchema } from "../../src/domain/schemas.js";

describe("domain schemas", () => {
  it("rejects empty goals", () => expect(CreateRunSchema.safeParse({ goal: "  " }).success).toBe(false));
  it("rejects non-positive limits", () => {
    const result = CreateAgentSchema.safeParse({
      name: "agent", systemInstructions: "act safely", model: { provider: "openai", model: "configured-later" },
      allowedTools: [], maximumSteps: 0, tokenBudget: 1, costBudgetMicrousd: 1, approvalPolicy: {}, outputSchema: null,
    });
    expect(result.success).toBe(false);
  });
  it("rejects inconsistent memory composition and write policy", () => {
    const base = {
      name: "memory-agent", systemInstructions: "retain deliberately", model: { provider: "local", model: "configured" },
      allowedTools: ["memory_store"], maximumSteps: 10, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null,
    };
    expect(CreateAgentSchema.safeParse({ ...base, memoryPolicy: { writeEnabled: true, allowedScopes: ["user"], maxWritesPerRun: 1, retrievalEnabled: false, maxContextItems: 5, maxContextTokens: 1000 } }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { memory: "native" }, memoryPolicy: { writeEnabled: true, allowedScopes: [], maxWritesPerRun: 0, retrievalEnabled: false, maxContextItems: 5, maxContextTokens: 1000 } }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { memory: "native" }, memoryPolicy: { writeEnabled: true, allowedScopes: ["user"], maxWritesPerRun: 1, retrievalEnabled: false, maxContextItems: 5, maxContextTokens: 1000 } }).success).toBe(true);
  });
  it("rejects inconsistent retrieval policy and composition", () => {
    const base = { name: "retrieval-agent", systemInstructions: "cite sources", model: { provider: "local", model: "configured" }, allowedTools: [], maximumSteps: 10, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null };
    expect(CreateAgentSchema.safeParse({ ...base, retrievalPolicy: { enabled: true, maxContextChunks: 3, maxContextTokens: 500, minimumScore: 0.2 } }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { retriever: "native" }, retrievalPolicy: { enabled: false, maxContextChunks: 3, maxContextTokens: 500, minimumScore: 0.2 } }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { retriever: "native" }, retrievalPolicy: { enabled: true, maxContextChunks: 3, maxContextTokens: 500, minimumScore: 0.2 } }).success).toBe(true);
  });
  it("rejects inconsistent connector and skill composition", () => {
    const base = { name: "external-agent", systemInstructions: "use governed capabilities", model: { provider: "local", model: "configured" }, allowedTools: ["mcp_call"], maximumSteps: 10, tokenBudget: 1000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null };
    const connectorId = "93000000-0000-4000-8000-000000000001"; const skillId = "93000000-0000-4000-8000-000000000002";
    expect(CreateAgentSchema.safeParse({ ...base, connectorPolicy: { enabled: true, connectorIds: [connectorId], maxContextTools: 10 } }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { connectors: "native" }, connectorPolicy: { enabled: true, connectorIds: [connectorId], maxContextTools: 10 } }).success).toBe(true);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { skills: "native" }, skillPolicy: { enabled: true, skillIds: [skillId], maxContextTokens: 500 } }).success).toBe(true);
    expect(CreateAgentSchema.safeParse({ ...base, skillPolicy: { enabled: true, skillIds: [skillId], maxContextTokens: 500 } }).success).toBe(false);
  });
  it("requires bounded native orchestration authority", () => {
    const childId = "93000000-0000-4000-8000-000000000003";
    const base = { name: "supervisor", systemInstructions: "delegate bounded work", model: { provider: "local", model: "configured" }, allowedTools: ["delegate_run"], maximumSteps: 20, tokenBudget: 5000, costBudgetMicrousd: 1000, approvalPolicy: {}, outputSchema: null };
    const policy = { enabled: true, allowedAgentIds: [childId], allowedRoles: ["researcher"], maximumChildren: 1, maximumParallel: 1, maximumDepth: 2, maximumChildTokenBudget: 1000, maximumChildCostBudgetMicrousd: 100, allowSharedContext: false };
    expect(CreateAgentSchema.safeParse({ ...base, orchestrationPolicy: policy }).success).toBe(false);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { orchestrator: "native" }, orchestrationPolicy: policy }).success).toBe(true);
    expect(CreateAgentSchema.safeParse({ ...base, composition: { orchestrator: "native" }, orchestrationPolicy: { ...policy, maximumParallel: 2 } }).success).toBe(false);
  });
});
