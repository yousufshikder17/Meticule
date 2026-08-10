import { z } from "zod";
import { MemoryService } from "../memory/memory-service.js";
import { MemoryToolInputSchema } from "../memory/memory-schema.js";
import type { ToolDefinition } from "./types.js";

export const memoryStoreTool: ToolDefinition = {
  name: "memory_store",
  description: "Deliberately retain one user- or agent-scoped memory with provenance.",
  inputSchema: MemoryToolInputSchema,
  outputSchema: z.object({ memoryId: z.uuid(), version: z.int().positive() }),
  riskLevel: "high",
  authorization: { requiredRoles: ["memory_write"] },
  approvalRequirement: "always",
  timeoutMs: 5_000,
  retryPolicy: { maxAttempts: 1, retryableErrors: [] },
  idempotency: "keyed_side_effect",
  retrySafety: "reconcilable",
  async authorize(input, context) {
    await new MemoryService(context.pool).authorizeToolWrite(context.principal, context.runId, input, context.idempotencyKey);
  },
  async execute(input, context) {
    return new MemoryService(context.pool).createFromApprovedTool(context.principal, context.runId, context.idempotencyKey, input);
  },
  async reconcile(_input, context) {
    const result = await new MemoryService(context.pool).reconcileToolWrite(context.principal, context.idempotencyKey);
    return result ? { status: "succeeded", output: result } : { status: "failed", error: { code: "memory_not_found" } };
  },
};
