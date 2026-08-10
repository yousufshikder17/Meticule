export interface EmbeddingBatch {
  vectors: number[][];
  inputTokens: number | null;
  providerRequestId: string | null;
}

export interface EmbeddingProvider {
  readonly id: string;
  readonly model: string;
  readonly dimensions: number;
  readonly execution: "local" | "remote";
  embed(texts: string[], signal: AbortSignal): Promise<EmbeddingBatch>;
  healthCheck(signal?: AbortSignal): Promise<{ healthy: boolean; details?: string }>;
}

export class EmbeddingConfigurationError extends Error {}
export class EmbeddingProviderError extends Error {
  constructor(public readonly code: "unavailable" | "timeout" | "malformed_response" | "cancelled" | "model_not_found" | "connection_failure", message: string, public readonly retryable: boolean, options?: ErrorOptions) {
    super(message, options); this.name = "EmbeddingProviderError";
  }
}

export class EmbeddingProviderRegistry {
  private readonly providers = new Map<string, EmbeddingProvider>();
  register(provider: EmbeddingProvider): void { if (this.providers.has(provider.id)) throw new EmbeddingConfigurationError(`Duplicate embedding provider: ${provider.id}`); this.providers.set(provider.id, provider); }
  get(id: string): EmbeddingProvider { const provider = this.providers.get(id); if (!provider) throw new EmbeddingConfigurationError(`Embedding provider is not configured: ${id}`); return provider; }
  list(): { id: string; model: string; dimensions: number; execution: "local" | "remote" }[] { return [...this.providers.values()].map(({ id, model, dimensions, execution }) => ({ id, model, dimensions, execution })); }
}
