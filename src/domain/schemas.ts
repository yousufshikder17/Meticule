import { z } from "zod";
import { RunStateSchema } from "./run-state.js";
import { AgentCompositionSchema } from "../execution/composition.js";
import { ProviderIdSchema } from "../models/model-types.js";

export const UuidSchema = z.uuid();

const ProviderTargetSchema = z.object({
  provider: ProviderIdSchema, model: z.string().min(1), maxOutputTokens: z.int().positive().default(1024),
  timeoutMs: z.int().min(100).max(600_000).default(60_000), inputCostMicrousdPerMillion: z.int().nonnegative().default(0),
  outputCostMicrousdPerMillion: z.int().nonnegative().default(0), cachedCostMicrousdPerMillion: z.int().nonnegative().default(0),
});

export const ModelConfigurationSchema = ProviderTargetSchema.extend({
  retryPolicy: z.object({ maxAttempts: z.int().min(1).max(10).default(1), baseDelayMs: z.int().min(0).max(60_000).default(250), maxDelayMs: z.int().min(0).max(300_000).default(5_000) }).default({ maxAttempts: 1, baseDelayMs: 250, maxDelayMs: 5_000 }),
  fallbacks: z.array(ProviderTargetSchema).max(5).default([]),
});

export const OrchestrationPolicySchema = z.object({
  enabled: z.boolean().default(false),
  allowedAgentIds: z.array(z.uuid()).max(100).refine((values) => new Set(values).size === values.length, "Allowed child agents must be unique").default([]),
  allowedRoles: z.array(z.string().trim().min(1).max(120)).max(50).refine((values) => new Set(values).size === values.length, "Delegation roles must be unique").default([]),
  maximumChildren: z.int().min(0).max(1_000).default(0),
  maximumParallel: z.int().min(1).max(100).default(1),
  maximumDepth: z.int().min(1).max(16).default(1),
  maximumChildTokenBudget: z.int().min(0).max(100_000_000).default(0),
  maximumChildCostBudgetMicrousd: z.int().min(0).default(0),
  allowSharedContext: z.boolean().default(false),
}).default({ enabled: false, allowedAgentIds: [], allowedRoles: [], maximumChildren: 0, maximumParallel: 1, maximumDepth: 1, maximumChildTokenBudget: 0, maximumChildCostBudgetMicrousd: 0, allowSharedContext: false });

const AgentConfigurationObjectSchema = z.object({
  name: z.string().trim().min(1).max(120),
  systemInstructions: z.string().min(1).max(100_000),
  model: ModelConfigurationSchema,
  composition: AgentCompositionSchema.default({ executionEngine: "native", contextBuilder: "native", outputParser: "native", planner: "disabled", retriever: "disabled", memory: "disabled", connectors: "disabled", skills: "disabled", orchestrator: "disabled" }),
  allowedTools: z.array(z.string().min(1)).max(100).default([]),
  maximumSteps: z.int().positive().max(10_000),
  tokenBudget: z.int().nonnegative(),
  costBudgetMicrousd: z.int().nonnegative(),
  approvalPolicy: z.record(z.string(), z.union([
    z.boolean(),
    z.object({
      requiredApproverRole: z.string().min(1).default("approval_reviewer"),
      separationOfDuties: z.boolean().default(true),
      riskExplanation: z.string().min(1).max(2_000).optional(),
    }),
  ])).default({}),
  outputSchema: z.record(z.string(), z.unknown()).nullable().default(null),
  contextPolicy: z.object({
    maxInputTokens: z.int().min(256).max(1_000_000).default(8192),
    recentSteps: z.int().min(1).max(500).default(20),
    summaryTargetTokens: z.int().min(64).max(100_000).default(2000),
  }).default({ maxInputTokens: 8192, recentSteps: 20, summaryTargetTokens: 2000 }),
  memoryPolicy: z.object({
    writeEnabled: z.boolean().default(false), allowedScopes: z.array(z.enum(["user", "agent"])).max(2).default([]),
    maxWritesPerRun: z.int().min(0).max(100).default(0), retrievalEnabled: z.boolean().default(false),
    maxContextItems: z.int().min(1).max(50).default(5), maxContextTokens: z.int().min(64).max(10_000).default(1000),
  }).default({ writeEnabled: false, allowedScopes: [], maxWritesPerRun: 0, retrievalEnabled: false, maxContextItems: 5, maxContextTokens: 1000 }),
  retrievalPolicy: z.object({
    enabled: z.boolean().default(false), maxContextChunks: z.int().min(1).max(20).default(5),
    maxContextTokens: z.int().min(32).max(20_000).default(2000), minimumScore: z.number().min(-1).max(1).default(0.2),
  }).default({ enabled: false, maxContextChunks: 5, maxContextTokens: 2000, minimumScore: 0.2 }),
  connectorPolicy: z.object({
    enabled: z.boolean().default(false), connectorIds: z.array(z.uuid()).max(20).refine((values) => new Set(values).size === values.length, "Connector IDs must be unique").default([]),
    maxContextTools: z.int().min(1).max(100).default(20),
  }).default({ enabled: false, connectorIds: [], maxContextTools: 20 }),
  skillPolicy: z.object({
    enabled: z.boolean().default(false), skillIds: z.array(z.uuid()).max(20).refine((values) => new Set(values).size === values.length, "Skill IDs must be unique").default([]),
    maxContextTokens: z.int().min(64).max(20_000).default(2_000),
  }).default({ enabled: false, skillIds: [], maxContextTokens: 2_000 }),
  orchestrationPolicy: OrchestrationPolicySchema,
});

export const AgentConfigurationSchema = AgentConfigurationObjectSchema.superRefine((value, context) => {
  const enabled = value.memoryPolicy.writeEnabled || value.memoryPolicy.retrievalEnabled;
  if (enabled && value.composition.memory !== "native") context.addIssue({ code: "custom", path: ["composition", "memory"], message: "Native memory composition is required when memory is enabled" });
  if (value.memoryPolicy.writeEnabled && (!value.memoryPolicy.allowedScopes.length || value.memoryPolicy.maxWritesPerRun < 1)) context.addIssue({ code: "custom", path: ["memoryPolicy"], message: "Enabled memory writes require an allowed scope and positive per-run quota" });
  if (!value.memoryPolicy.writeEnabled && (value.memoryPolicy.allowedScopes.length || value.memoryPolicy.maxWritesPerRun !== 0)) context.addIssue({ code: "custom", path: ["memoryPolicy"], message: "Disabled memory writes must have no allowed scopes and a zero quota" });
  if (value.retrievalPolicy.enabled && value.composition.retriever !== "native") context.addIssue({ code: "custom", path: ["composition", "retriever"], message: "Native retriever composition is required when retrieval is enabled" });
  if (!value.retrievalPolicy.enabled && value.composition.retriever !== "disabled") context.addIssue({ code: "custom", path: ["retrievalPolicy"], message: "Retriever composition must be disabled when retrieval policy is disabled" });
  if (value.connectorPolicy.enabled && (value.composition.connectors !== "native" || !value.connectorPolicy.connectorIds.length || !value.allowedTools.includes("mcp_call"))) context.addIssue({ code: "custom", path: ["connectorPolicy"], message: "Enabled connectors require native composition, at least one connector, and mcp_call in the agent tool allowlist" });
  if (!value.connectorPolicy.enabled && (value.composition.connectors !== "disabled" || value.connectorPolicy.connectorIds.length)) context.addIssue({ code: "custom", path: ["connectorPolicy"], message: "Disabled connectors require disabled composition and no connector IDs" });
  if (value.skillPolicy.enabled && (value.composition.skills !== "native" || !value.skillPolicy.skillIds.length)) context.addIssue({ code: "custom", path: ["skillPolicy"], message: "Enabled skills require native composition and at least one skill" });
  if (!value.skillPolicy.enabled && (value.composition.skills !== "disabled" || value.skillPolicy.skillIds.length)) context.addIssue({ code: "custom", path: ["skillPolicy"], message: "Disabled skills require disabled composition and no skill IDs" });
  const orchestration = value.orchestrationPolicy;
  if (orchestration.enabled && (value.composition.orchestrator !== "native" || !value.allowedTools.includes("delegate_run") || !orchestration.allowedAgentIds.length || !orchestration.allowedRoles.length || orchestration.maximumChildren < 1 || orchestration.maximumParallel > orchestration.maximumChildren || orchestration.maximumChildTokenBudget < 1)) context.addIssue({ code: "custom", path: ["orchestrationPolicy"], message: "Enabled orchestration requires native composition, delegate_run, allowed agents and roles, and coherent child/concurrency/token limits" });
  if (!orchestration.enabled && (value.composition.orchestrator !== "disabled" || orchestration.allowedAgentIds.length || orchestration.allowedRoles.length || orchestration.maximumChildren !== 0 || orchestration.maximumChildTokenBudget !== 0 || orchestration.maximumChildCostBudgetMicrousd !== 0 || orchestration.allowSharedContext)) context.addIssue({ code: "custom", path: ["orchestrationPolicy"], message: "Disabled orchestration requires disabled composition and empty delegation authority" });
});

export const CreateAgentSchema = AgentConfigurationSchema;
export const PatchAgentSchema = AgentConfigurationObjectSchema.partial().extend({ expectedVersion: z.int().positive() });
export const CreateRunSchema = z.object({ goal: z.string().trim().min(1).max(100_000) });
export const ResumeRunSchema = z.object({ expectedVersion: z.int().positive() });
export const ApprovalDecisionSchema = z.object({ comment: z.string().trim().max(2_000).optional() });

export const RunSchema = z.object({
  id: UuidSchema,
  tenantId: UuidSchema,
  agentId: UuidSchema,
  createdBy: UuidSchema,
  goal: z.string(),
  status: RunStateSchema,
  currentStep: z.int().nonnegative(),
  version: z.int().positive(),
  leaseOwner: z.string().nullable(),
  leaseExpiresAt: z.date().nullable(),
  cancellationRequestedAt: z.date().nullable(),
  agentVersion: z.int().positive(), agentConfigurationSnapshot: AgentConfigurationSchema,
  parentRunId: UuidSchema.nullable(), rootRunId: UuidSchema, delegationDepth: z.int().nonnegative(), delegationRole: z.string().nullable(), contextScope: z.enum(["private","shared"]),
  tokenBudgetLimit: z.int().nonnegative(), costBudgetLimitMicrousd: z.int().nonnegative(), reservedChildTokens: z.int().nonnegative(), reservedChildCostMicrousd: z.int().nonnegative(),
  inputTokens: z.int().nonnegative(),
  outputTokens: z.int().nonnegative(),
  costMicrousd: z.int().nonnegative(),
  finalOutput: z.unknown().nullable(),
  errorDetails: z.unknown().nullable(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export type AgentConfiguration = z.infer<typeof AgentConfigurationSchema>;
export type AgentConfigurationInput = z.input<typeof AgentConfigurationSchema>;
export type Run = z.infer<typeof RunSchema>;
