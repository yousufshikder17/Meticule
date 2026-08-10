import { z } from "zod";

const ExpectedToolCallSchema = z.object({
  toolName: z.string().min(1).max(200),
  arguments: z.record(z.string(), z.unknown()).optional(),
  minimumCalls: z.int().min(1).max(100).default(1),
  maximumCalls: z.int().min(1).max(100).default(1),
}).refine((value) => value.maximumCalls >= value.minimumCalls, "maximumCalls must be at least minimumCalls");

export const EvaluationExpectationsSchema = z.object({
  expectedStatus: z.enum(["completed", "failed", "cancelled"]).default("completed"),
  expectedFinalOutput: z.unknown().optional(),
  outputContains: z.array(z.string().min(1).max(500)).max(50).default([]),
  expectedToolCalls: z.array(ExpectedToolCallSchema).max(50).default([]),
  forbiddenTools: z.array(z.string().min(1).max(200)).max(50).default([]),
  requiredApprovalTools: z.array(z.string().min(1).max(200)).max(50).default([]),
  forbiddenApprovalTools: z.array(z.string().min(1).max(200)).max(50).default([]),
  requiredRetrievedDocumentIds: z.array(z.uuid()).max(50).default([]),
  requiredCitationSourceUris: z.array(z.string().url().max(2000)).max(50).default([]),
  forbiddenOutputText: z.array(z.string().min(1).max(500)).max(100).default([]),
  requiredAuditEvents: z.array(z.string().min(1).max(200)).max(50).default([]),
  maximumLatencyMs: z.int().positive().optional(),
  maximumTokens: z.int().nonnegative().optional(),
  maximumCostMicrousd: z.int().nonnegative().optional(),
  maximumSteps: z.int().nonnegative().optional(),
});

export const EvaluationThresholdsSchema = z.object({
  minimumPassRate: z.number().min(0).max(1).default(1),
  minimumTaskSuccess: z.number().min(0).max(1).default(1),
  minimumToolSelection: z.number().min(0).max(1).default(1),
  minimumArgumentCorrectness: z.number().min(0).max(1).default(1),
  minimumRetrievalQuality: z.number().min(0).max(1).default(1),
  minimumCitationCorrectness: z.number().min(0).max(1).default(1),
  minimumApprovalCorrectness: z.number().min(0).max(1).default(1),
  minimumSafety: z.number().min(0).max(1).default(1),
  minimumRecoverySuccess: z.number().min(0).max(1).default(1),
  maximumAverageLatencyMs: z.int().positive().nullable().default(null),
  maximumAverageTokens: z.int().nonnegative().nullable().default(null),
  maximumAverageCostMicrousd: z.int().nonnegative().nullable().default(null),
  maximumAverageSteps: z.number().nonnegative().nullable().default(null),
});

export const EvaluationCaseInputSchema = z.object({
  name: z.string().trim().min(1).max(160),
  goal: z.string().trim().min(1).max(100_000),
  expectations: EvaluationExpectationsSchema,
  tags: z.array(z.string().trim().min(1).max(80)).max(30).refine((values) => new Set(values).size === values.length, "Tags must be unique").default([]),
});

export const CreateEvaluationSuiteSchema = z.object({
  name: z.string().trim().min(1).max(160),
  description: z.string().trim().min(1).max(4000),
  thresholds: EvaluationThresholdsSchema,
  cases: z.array(EvaluationCaseInputSchema).min(1).max(100),
});

export const CreateEvaluationExecutionSchema = z.object({
  agentId: z.uuid(),
  mode: z.enum(["deterministic_ci", "model_dependent"]),
  candidateLabel: z.string().trim().min(1).max(160),
});

export const EvaluationExecutionListQuerySchema = z.object({
  suiteId: z.uuid().optional(),
  agentId: z.uuid().optional(),
  mode: z.enum(["deterministic_ci", "model_dependent"]).optional(),
  limit: z.coerce.number().int().min(1).max(200).default(50),
});

export type EvaluationExpectations = z.infer<typeof EvaluationExpectationsSchema>;
export type EvaluationThresholds = z.infer<typeof EvaluationThresholdsSchema>;
export type CreateEvaluationSuite = z.infer<typeof CreateEvaluationSuiteSchema>;
export type CreateEvaluationExecution = z.infer<typeof CreateEvaluationExecutionSchema>;
