import type { ToolDefinition } from "./types.js";
import { DelegateRunInputSchema, DelegateRunOutputSchema } from "../orchestration/orchestration-schema.js";
import { OrchestrationService } from "../orchestration/orchestration-service.js";

export function createDelegateRunTool(orchestration: OrchestrationService): ToolDefinition {
  return {
    name: "delegate_run",
    description: "Create one tenant-scoped child run with a bounded goal, role, context scope, and budget. The child uses the same durable lifecycle and must be awaited with pause reason wait_for_children.",
    inputSchema: DelegateRunInputSchema,
    outputSchema: DelegateRunOutputSchema,
    riskLevel: "medium",
    authorization: { requiredRoles: [] },
    approvalRequirement: "policy",
    timeoutMs: 5_000,
    retryPolicy: { maxAttempts: 1, retryableErrors: [] },
    idempotency: "keyed_side_effect",
    retrySafety: "reconcilable",
    async authorize(input, context) {
      await orchestration.authorizeDelegation(context.principal, context.runId, DelegateRunInputSchema.parse(input), context.idempotencyKey);
    },
    async execute(input, context) {
      return orchestration.delegate(context.principal, context.runId, context.idempotencyKey, DelegateRunInputSchema.parse(input));
    },
    async reconcile(_input, context) {
      const result = await orchestration.reconcileDelegation(context.principal, context.runId, context.idempotencyKey);
      return result ? { status: "succeeded", output: result } : { status: "failed", error: { code: "delegated_run_not_found" } };
    },
  };
}
