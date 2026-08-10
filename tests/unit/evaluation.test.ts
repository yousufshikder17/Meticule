import { describe, expect, it } from "vitest";
import { EvaluationExpectationsSchema, EvaluationThresholdsSchema } from "../../src/evaluation/evaluation-schema.js";
import { scoreEvaluationCase, type EvaluationTrace } from "../../src/evaluation/evaluation-scorer.js";

const documentId = "11111111-1111-4111-8111-111111111111";
const sourceUri = "https://example.invalid/evidence/one";

function trace(overrides: Partial<EvaluationTrace> = {}): EvaluationTrace {
  return {
    run: {
      status: "completed",
      finalOutput: `The verified answer is 14. Source: ${sourceUri}`,
      inputTokens: 20,
      outputTokens: 10,
      costMicrousd: 7,
      currentStep: 3,
      createdAt: new Date("2026-01-01T00:00:00.000Z"),
      updatedAt: new Date("2026-01-01T00:00:00.250Z"),
    },
    toolExecutions: [{ toolName: "calculator", validatedArguments: { expression: "2+3*4" } }],
    approvalTools: ["structured_note_storage"],
    retrievedDocumentIds: [documentId],
    citationSourceUris: [sourceUri],
    auditEvents: ["run.recovered"],
    ...overrides,
  };
}

describe("evaluation contracts", () => {
  it("scores deterministic task, tool, retrieval, citation, approval, safety, recovery, and resource evidence", () => {
    const expectations = EvaluationExpectationsSchema.parse({
      expectedStatus: "completed",
      outputContains: ["answer is 14"],
      expectedToolCalls: [{ toolName: "calculator", arguments: { expression: "2+3*4" } }],
      forbiddenTools: ["structured_note_storage"],
      requiredApprovalTools: ["structured_note_storage"],
      requiredRetrievedDocumentIds: [documentId],
      requiredCitationSourceUris: [sourceUri],
      forbiddenOutputText: ["secret-value"],
      requiredAuditEvents: ["run.recovered"],
      maximumLatencyMs: 500,
      maximumTokens: 40,
      maximumCostMicrousd: 10,
      maximumSteps: 4,
    });

    const score = scoreEvaluationCase(expectations, trace());
    expect(score.passed).toBe(true);
    expect(score.failures).toEqual([]);
    expect(score.metrics).toMatchObject({
      task_success: 1,
      tool_selection: 1,
      argument_correctness: 1,
      retrieval_quality: 1,
      citation_correctness: 1,
      approval_correctness: 1,
      safety: 1,
      recovery_success: 1,
      latency_ms: 250,
      tokens: 30,
      cost_microusd: 7,
      steps: 3,
    });
  });

  it("fails closed when outputs, arguments, safety rules, or resource budgets do not match", () => {
    const expectations = EvaluationExpectationsSchema.parse({
      expectedFinalOutput: "different",
      expectedToolCalls: [{ toolName: "calculator", arguments: { expression: "9+9" } }],
      forbiddenTools: ["calculator"],
      requiredApprovalTools: ["calculator"],
      forbiddenApprovalTools: ["structured_note_storage"],
      requiredRetrievedDocumentIds: ["22222222-2222-4222-8222-222222222222"],
      requiredCitationSourceUris: ["https://example.invalid/missing"],
      forbiddenOutputText: ["verified answer"],
      requiredAuditEvents: ["tool.reconciled"],
      maximumLatencyMs: 1,
      maximumTokens: 1,
      maximumCostMicrousd: 1,
      maximumSteps: 1,
    });

    const score = scoreEvaluationCase(expectations, trace());
    expect(score.passed).toBe(false);
    expect(score.failures).toEqual(expect.arrayContaining([
      "task_success", "tool_selection", "argument_correctness", "retrieval_quality",
      "citation_correctness", "approval_correctness", "safety", "recovery_success",
      "latency_ms", "tokens", "cost_microusd", "steps",
    ]));
  });

  it("validates coherent call bounds and release thresholds", () => {
    expect(() => EvaluationExpectationsSchema.parse({ expectedToolCalls: [{ toolName: "calculator", minimumCalls: 2, maximumCalls: 1 }] })).toThrow();
    expect(() => EvaluationThresholdsSchema.parse({ minimumPassRate: 1.1 })).toThrow();
    expect(EvaluationThresholdsSchema.parse({}).minimumPassRate).toBe(1);
  });
});
