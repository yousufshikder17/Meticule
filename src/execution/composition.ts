import { z } from "zod";
export const AgentCompositionSchema=z.object({executionEngine:z.literal("native").default("native"),contextBuilder:z.literal("native").default("native"),outputParser:z.literal("native").default("native"),planner:z.enum(["disabled","native"]).default("disabled"),retriever:z.enum(["disabled","native"]).default("disabled"),memory:z.enum(["disabled","native"]).default("disabled"),connectors:z.enum(["disabled","native"]).default("disabled"),skills:z.enum(["disabled","native"]).default("disabled"),orchestrator:z.enum(["disabled","native"]).default("disabled")});
export type AgentComposition=z.infer<typeof AgentCompositionSchema>;
export function validateComposition(value:unknown):AgentComposition{return AgentCompositionSchema.parse(value);}
