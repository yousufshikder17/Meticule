import { z } from "zod";
import { EmbeddingProviderError, type EmbeddingBatch, type EmbeddingProvider } from "./embedding-provider.js";

const EmbedResponseSchema = z.object({
  model: z.string().optional(), embeddings: z.array(z.array(z.number().finite())), prompt_eval_count: z.number().int().nonnegative().optional(),
});
const TagsResponseSchema = z.object({ models: z.array(z.object({ name: z.string(), model: z.string().optional() })) });

export class OllamaEmbeddingProvider implements EmbeddingProvider {
  readonly id: string; readonly execution = "local" as const;
  constructor(private readonly settings: { baseUrl: string; model: string; dimensions: number; timeoutMs: number; providerId?: string }) {
    this.id = settings.providerId ?? "ollama-embeddings";
  }
  get model(): string { return this.settings.model; }
  get dimensions(): number { return this.settings.dimensions; }

  async embed(texts: string[], signal: AbortSignal): Promise<EmbeddingBatch> {
    if (!texts.length || texts.length > 128) throw new EmbeddingProviderError("malformed_response", "Embedding batches must contain 1–128 inputs", false);
    if (signal.aborted) throw new EmbeddingProviderError("cancelled", "Embedding request was cancelled", false, { cause: signal.reason });
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason); signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(new Error("embedding timeout")), this.settings.timeoutMs);
    try {
      const response = await fetch(`${this.settings.baseUrl.replace(/\/$/, "")}/api/embed`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: this.model, input: texts, truncate: false }), signal: controller.signal });
      if (!response.ok) {
        throw new EmbeddingProviderError(response.status === 404 ? "model_not_found" : "unavailable", `Ollama embedding request failed with HTTP ${response.status}`, response.status >= 500);
      }
      let body: unknown; try { body = await response.json(); } catch (error) { throw new EmbeddingProviderError("malformed_response", "Ollama embedding response was not valid JSON", false, { cause: error }); }
      const parsed = EmbedResponseSchema.safeParse(body);
      if (!parsed.success || parsed.data.embeddings.length !== texts.length || parsed.data.embeddings.some((vector) => vector.length !== this.dimensions)) throw new EmbeddingProviderError("malformed_response", "Ollama returned an invalid embedding batch or unexpected dimension", false);
      return { vectors: parsed.data.embeddings, inputTokens: parsed.data.prompt_eval_count ?? null, providerRequestId: null };
    } catch (error) {
      if (error instanceof EmbeddingProviderError) throw error;
      if (controller.signal.aborted) throw new EmbeddingProviderError(signal.aborted ? "cancelled" : "timeout", signal.aborted ? "Embedding request was cancelled" : "Embedding request timed out", !signal.aborted, { cause: error });
      throw new EmbeddingProviderError("connection_failure", "Could not connect to Ollama embedding endpoint", true, { cause: error });
    } finally { clearTimeout(timer); signal.removeEventListener("abort", abort); }
  }

  async healthCheck(signal = new AbortController().signal): Promise<{ healthy: boolean; details?: string }> {
    try {
      const response = await fetch(`${this.settings.baseUrl.replace(/\/$/, "")}/api/tags`, { signal });
      if (!response.ok) return { healthy: false, details: `HTTP ${response.status}` };
      const parsed = TagsResponseSchema.safeParse(await response.json());
      if (!parsed.success) return { healthy: false, details: "malformed tags response" };
      const matches = (candidate: string | undefined): boolean => candidate === this.model || candidate === `${this.model}:latest` || this.model === `${candidate}:latest`;
      const available = parsed.data.models.some((entry) => matches(entry.name) || matches(entry.model));
      if (!available) return { healthy: false, details: `model not installed: ${this.model}` };
      await this.embed(["embedding health check"], signal);
      return { healthy: true };
    } catch (error) { return { healthy: false, details: error instanceof Error ? error.message : String(error) }; }
  }
}
