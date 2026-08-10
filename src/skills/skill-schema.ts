import { z } from "zod";

const ToolNameSchema = z.string().trim().min(1).max(200).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
export const SkillProvenanceSchema = z.object({
  source: z.enum(["user_authored", "public_reference", "generated_and_reviewed"]),
  reason: z.string().trim().min(1).max(2_000),
  sourceUri: z.url().max(2_000).nullable().default(null),
}).strict();
export const CreateSkillSchema = z.object({
  name: z.string().trim().min(1).max(200), instructions: z.string().trim().min(1).max(20_000),
  allowedTools: z.array(ToolNameSchema).max(100).refine((values) => new Set(values).size === values.length, "Allowed tools must be unique"),
  provenance: SkillProvenanceSchema,
});
export const RevokeSkillSchema = z.object({ reason: z.string().trim().min(1).max(2_000) });
export type CreateSkillInput = z.infer<typeof CreateSkillSchema>;
