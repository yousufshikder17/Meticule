import { z } from "zod";
import { PlanSchema } from "../planning/plan-schema.js";

export const AgentActionSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("call_tool"), toolName: z.string().min(1), arguments: z.record(z.string(), z.unknown()), idempotencyKey: z.string().min(1).optional() }),
  z.object({ type: z.literal("final_answer"), output: z.unknown() }),
  z.object({ type: z.literal("request_clarification"), question: z.string().min(1) }),
  z.object({ type: z.literal("update_plan"), plan: PlanSchema }),
  z.object({ type: z.literal("pause"), reason: z.string().min(1) }),
]);
export type AgentAction = z.infer<typeof AgentActionSchema>;

export const AGENT_ACTION_JSON_SCHEMA = {
  type: "object", oneOf: [
    { properties: { type: { const: "call_tool" }, toolName: { type: "string" }, arguments: { type: "object" }, idempotencyKey: { type: "string" } }, required: ["type","toolName","arguments"], additionalProperties: false },
    { properties: { type: { const: "final_answer" }, output: {} }, required: ["type","output"], additionalProperties: false },
    { properties: { type: { const: "request_clarification" }, question: { type: "string" } }, required: ["type","question"], additionalProperties: false },
    { properties: { type: { const: "update_plan" }, plan: { type: "object", properties: {
      objective: { type: "string" }, tasks: { type: "array", minItems: 1, maxItems: 100, items: { type: "object", properties: {
        taskId: { type: "string" }, objective: { type: "string" }, dependencies: { type: "array", items: { type: "string" } },
        status: { enum: ["pending","ready","running","completed","failed","blocked"] }, assignedExecutor: { const: "native" },
        attemptCount: { type: "integer", minimum: 0 }, result: {}, error: {}, evidenceStepId: { type: ["string","null"], format: "uuid" },
      }, required: ["taskId","objective"], additionalProperties: false } },
    }, required: ["objective","tasks"], additionalProperties: false } }, required: ["type","plan"], additionalProperties: false },
    { properties: { type: { const: "pause" }, reason: { type: "string" } }, required: ["type","reason"], additionalProperties: false }
  ]
} as const;
