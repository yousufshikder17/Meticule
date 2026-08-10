import { z } from "zod";

export const ProviderIdSchema = z.string().trim().min(1).max(100).regex(/^[a-z0-9][a-z0-9._-]*$/);
export type ProviderId = string;

export const CanonicalContentSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({ type: z.literal("image_url"), url: z.url() }),
]);
export const CanonicalToolCallSchema = z.object({ id: z.string().min(1), name: z.string().min(1), arguments: z.record(z.string(), z.unknown()) });
export const CanonicalMessageSchema = z.object({
  role: z.enum(["system", "user", "assistant", "tool"]),
  content: z.array(CanonicalContentSchema),
  toolCallId: z.string().optional(),
  toolCalls: z.array(CanonicalToolCallSchema).optional(),
});
export const CanonicalToolDefinitionSchema = z.object({ name: z.string(), description: z.string(), inputSchema: z.record(z.string(), z.unknown()) });

export interface ProviderCapabilities {
  toolCalling: boolean; structuredOutput: boolean; streaming: boolean; vision: boolean; tokenUsage: boolean;
  contextWindow: number | null; nativeIdempotency: boolean; execution: "local" | "remote";
}
export interface ModelRequest {
  requestId: string; providerId: string; modelId: string; messages: z.infer<typeof CanonicalMessageSchema>[];
  tools: z.infer<typeof CanonicalToolDefinitionSchema>[]; outputSchema?: Record<string, unknown>;
  maxOutputTokens: number; timeoutMs: number; signal: AbortSignal; metadata: Record<string, string>;
}
export interface ModelResponse {
  text: string; toolCalls: z.infer<typeof CanonicalToolCallSchema>[]; stopReason: string;
  usage: { inputTokens: number; outputTokens: number; cachedTokens: number };
  providerRequestId?: string; metadata: Record<string, unknown>;
}
