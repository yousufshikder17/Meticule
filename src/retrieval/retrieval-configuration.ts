import { z } from "zod";
import type pg from "pg";
import { EmbeddingConfigurationError } from "./embedding-provider.js";
import { OllamaEmbeddingProvider } from "./ollama-embedding-provider.js";
import { RetrievalService } from "./retrieval-service.js";

const SettingsSchema = z.object({
  OLLAMA_EMBEDDING_MODEL: z.string().trim().min(1),
  OLLAMA_EMBEDDING_DIMENSIONS: z.coerce.number().int().min(1).max(4096),
  OLLAMA_EMBEDDING_BASE_URL: z.url().default("http://localhost:11434"),
  OLLAMA_EMBEDDING_TIMEOUT_MS: z.coerce.number().int().min(100).max(600_000).default(60_000),
});

export function createRetrievalService(pool: pg.Pool, env: NodeJS.ProcessEnv = process.env): RetrievalService | null {
  if (!env.OLLAMA_EMBEDDING_MODEL) return null;
  const parsed = SettingsSchema.safeParse(env);
  if (!parsed.success) throw new EmbeddingConfigurationError(`Invalid Ollama embedding configuration: ${z.prettifyError(parsed.error)}`);
  return new RetrievalService(pool, new OllamaEmbeddingProvider({ baseUrl: parsed.data.OLLAMA_EMBEDDING_BASE_URL, model: parsed.data.OLLAMA_EMBEDDING_MODEL, dimensions: parsed.data.OLLAMA_EMBEDDING_DIMENSIONS, timeoutMs: parsed.data.OLLAMA_EMBEDDING_TIMEOUT_MS }));
}
