import { z } from "zod";

const GeminiPartSchema = z.object({
  text: z.string().optional(),
  functionCall: z.object({
    id: z.string().optional(),
    name: z.string(),
    args: z.record(z.string(), z.unknown()),
  }).optional(),
}).refine((part) => part.text !== undefined || part.functionCall !== undefined);

export const GeminiResponseSchema = z.object({
  candidates: z.array(z.object({
    finishReason: z.string().optional(),
    content: z.object({ parts: z.array(GeminiPartSchema) }),
  })).min(1),
  usageMetadata: z.object({
    promptTokenCount: z.number().int().nonnegative().optional(),
    candidatesTokenCount: z.number().int().nonnegative().optional(),
    cachedContentTokenCount: z.number().int().nonnegative().optional(),
  }).optional(),
});
