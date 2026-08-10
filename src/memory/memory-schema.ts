import { z } from "zod";

export const MemoryScopeSchema = z.enum(["user", "agent", "tenant"]);
export const MemoryTypeSchema = z.enum(["fact", "note", "procedure", "outcome"]);
const BoundedMetadataSchema = z.record(z.string().trim().min(1).max(100), z.json())
  .refine((value) => Object.keys(value).length <= 50, "Metadata may contain at most 50 keys")
  .refine((value) => JSON.stringify(value).length <= 20_000, "Metadata exceeds 20,000 serialized characters");
export const MemoryContentSchema = z.object({ text: z.string().trim().min(1).max(20_000), attributes: BoundedMetadataSchema.default({}) });
export const MemoryRelevanceSchema = z.object({ tags: z.array(z.string().trim().min(1).max(100)).max(50).default([]), importance: z.number().min(0).max(1).default(0.5) }).default({ tags: [], importance: 0.5 });
export const CreateMemorySchema = z.object({
  scope: MemoryScopeSchema, agentId: z.uuid().nullable().default(null), memoryType: MemoryTypeSchema,
  content: MemoryContentSchema, provenance: BoundedMetadataSchema.default({}),
  creationReason: z.string().trim().min(1).max(2_000), sourceRunId: z.uuid().nullable().default(null),
  relevance: MemoryRelevanceSchema, retentionUntil: z.iso.datetime().nullable().default(null),
}).superRefine((value, context) => {
  if ((value.scope === "agent") !== Boolean(value.agentId)) context.addIssue({ code: "custom", path: ["agentId"], message: "agentId is required only for agent scope" });
});
export const CorrectMemorySchema = z.object({
  content: MemoryContentSchema, creationReason: z.string().trim().min(1).max(2_000),
  provenance: BoundedMetadataSchema.default({}), relevance: MemoryRelevanceSchema,
  retentionUntil: z.iso.datetime().nullable().default(null),
});
export const MemoryListQuerySchema = z.object({ agentId: z.uuid().optional(), memoryType: MemoryTypeSchema.optional(), q: z.string().trim().max(500).optional(), limit: z.coerce.number().int().min(1).max(100).default(50) });
export const MemoryToolInputSchema = z.object({
  scope: z.enum(["user", "agent"]), memoryType: MemoryTypeSchema, content: MemoryContentSchema,
  creationReason: z.string().trim().min(1).max(2_000), relevance: MemoryRelevanceSchema,
  retentionDays: z.int().min(1).max(3650).nullable().default(null),
});
export type CreateMemoryInput = z.infer<typeof CreateMemorySchema>;
export type MemoryToolInput = z.infer<typeof MemoryToolInputSchema>;
