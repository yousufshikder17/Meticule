import { z } from "zod";

export const DelegateRunInputSchema = z.object({
  targetAgentId: z.uuid(),
  goal: z.string().trim().min(1).max(100_000),
  roleName: z.string().trim().min(1).max(120),
  required: z.boolean().default(true),
  contextScope: z.enum(["private", "shared"]).default("private"),
  tokenBudget: z.int().positive().max(100_000_000),
  costBudgetMicrousd: z.int().nonnegative(),
});

export const DelegateRunOutputSchema = z.object({
  childRunId: z.uuid(),
  status: z.literal("queued"),
  depth: z.int().positive(),
  roleName: z.string().min(1),
});

export type DelegateRunInput = z.infer<typeof DelegateRunInputSchema>;
export type DelegateRunOutput = z.infer<typeof DelegateRunOutputSchema>;
