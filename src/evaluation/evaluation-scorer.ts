import { canonicalJsonHash } from "../domain/canonical-json.js";
import type { EvaluationExpectations } from "./evaluation-schema.js";

export interface EvaluationTrace {
  run: { status: string; finalOutput: unknown; inputTokens: number; outputTokens: number; costMicrousd: number; currentStep: number; createdAt: Date; updatedAt: Date };
  toolExecutions: Array<{ toolName: string; validatedArguments: unknown }>;
  approvalTools: string[];
  retrievedDocumentIds: string[];
  citationSourceUris: string[];
  auditEvents: string[];
}

export interface Measurement { name: string; value: number; unit: string; passed: boolean; details: Record<string, unknown> }
export interface CaseScore { passed: boolean; failures: string[]; metrics: Record<string, number>; measurements: Measurement[] }

function ratio(matched: number, total: number): number { return total === 0 ? 1 : matched / total; }
function outputText(value: unknown): string { return typeof value === "string" ? value : JSON.stringify(value) ?? ""; }

export function scoreEvaluationCase(expectations: EvaluationExpectations, trace: EvaluationTrace): CaseScore {
  const failures: string[] = []; const measurements: Measurement[] = [];
  const add = (name: string, value: number, unit: string, passed: boolean, details: Record<string, unknown> = {}) => {
    measurements.push({ name, value, unit, passed, details }); if (!passed) failures.push(name);
  };
  const text = outputText(trace.run.finalOutput); const lower = text.toLowerCase();
  const statusOk = trace.run.status === expectations.expectedStatus;
  const exactOutputOk = expectations.expectedFinalOutput === undefined || canonicalJsonHash(trace.run.finalOutput) === canonicalJsonHash(expectations.expectedFinalOutput);
  const containsMatched = expectations.outputContains.filter((item) => lower.includes(item.toLowerCase())).length;
  const taskScore = (statusOk ? 1 : 0) * (exactOutputOk ? 1 : 0) * ratio(containsMatched, expectations.outputContains.length);
  add("task_success", taskScore, "ratio", taskScore === 1, { expectedStatus: expectations.expectedStatus, actualStatus: trace.run.status, exactOutputRequired: expectations.expectedFinalOutput !== undefined });

  let selected = 0; let argumentsCorrect = 0;
  for (const expected of expectations.expectedToolCalls) {
    const calls = trace.toolExecutions.filter((call) => call.toolName === expected.toolName);
    if (calls.length >= expected.minimumCalls && calls.length <= expected.maximumCalls) selected += 1;
    if (expected.arguments === undefined || calls.some((call) => canonicalJsonHash(call.validatedArguments) === canonicalJsonHash(expected.arguments))) argumentsCorrect += 1;
  }
  const forbiddenSelected = expectations.forbiddenTools.filter((name) => trace.toolExecutions.some((call) => call.toolName === name));
  const toolScore = ratio(selected, expectations.expectedToolCalls.length) * (forbiddenSelected.length ? 0 : 1);
  add("tool_selection", toolScore, "ratio", toolScore === 1, { forbiddenSelected });
  const argumentScore = ratio(argumentsCorrect, expectations.expectedToolCalls.length);
  add("argument_correctness", argumentScore, "ratio", argumentScore === 1);

  const retrieved = expectations.requiredRetrievedDocumentIds.filter((id) => trace.retrievedDocumentIds.includes(id)).length;
  const retrievalScore = ratio(retrieved, expectations.requiredRetrievedDocumentIds.length);
  add("retrieval_quality", retrievalScore, "ratio", retrievalScore === 1);
  const cited = expectations.requiredCitationSourceUris.filter((uri) => trace.citationSourceUris.includes(uri) && lower.includes(uri.toLowerCase())).length;
  const citationScore = ratio(cited, expectations.requiredCitationSourceUris.length);
  add("citation_correctness", citationScore, "ratio", citationScore === 1);

  const requiredApprovals = expectations.requiredApprovalTools.filter((name) => trace.approvalTools.includes(name)).length;
  const forbiddenApprovals = expectations.forbiddenApprovalTools.filter((name) => trace.approvalTools.includes(name));
  const approvalScore = ratio(requiredApprovals, expectations.requiredApprovalTools.length) * (forbiddenApprovals.length ? 0 : 1);
  add("approval_correctness", approvalScore, "ratio", approvalScore === 1, { forbiddenApprovals });
  const forbiddenOutput = expectations.forbiddenOutputText.filter((item) => lower.includes(item.toLowerCase()));
  add("safety", forbiddenOutput.length ? 0 : 1, "ratio", forbiddenOutput.length === 0, { matchedRuleCount: forbiddenOutput.length });
  const recoveryMatched = expectations.requiredAuditEvents.filter((event) => trace.auditEvents.includes(event)).length;
  const recoveryScore = ratio(recoveryMatched, expectations.requiredAuditEvents.length);
  add("recovery_success", recoveryScore, "ratio", recoveryScore === 1);

  const latency = Math.max(0, trace.run.updatedAt.getTime() - trace.run.createdAt.getTime());
  const tokens = trace.run.inputTokens + trace.run.outputTokens;
  add("latency_ms", latency, "milliseconds", expectations.maximumLatencyMs === undefined || latency <= expectations.maximumLatencyMs, { maximum: expectations.maximumLatencyMs ?? null });
  add("tokens", tokens, "tokens", expectations.maximumTokens === undefined || tokens <= expectations.maximumTokens, { maximum: expectations.maximumTokens ?? null });
  add("cost_microusd", trace.run.costMicrousd, "micro_usd", expectations.maximumCostMicrousd === undefined || trace.run.costMicrousd <= expectations.maximumCostMicrousd, { maximum: expectations.maximumCostMicrousd ?? null });
  add("steps", trace.run.currentStep, "steps", expectations.maximumSteps === undefined || trace.run.currentStep <= expectations.maximumSteps, { maximum: expectations.maximumSteps ?? null });
  return { passed: failures.length === 0, failures, metrics: Object.fromEntries(measurements.map((item) => [item.name, item.value])), measurements };
}
