import { createHash } from "node:crypto";
import type { EmbeddingBatch, EmbeddingProvider } from "../../src/retrieval/embedding-provider.js";

/** Test-only semantic approximation. It is never registered by production configuration. */
export class DeterministicTestEmbeddingProvider implements EmbeddingProvider {
  readonly id = "test-deterministic-embeddings"; readonly model = "test-hash-v1"; readonly dimensions = 32; readonly execution = "local" as const; calls = 0;
  async embed(texts: string[], signal: AbortSignal): Promise<EmbeddingBatch> {
    if (signal.aborted) throw signal.reason ?? new Error("cancelled"); this.calls += 1;
    return { vectors: texts.map((text) => this.vector(text)), inputTokens: texts.reduce((total, text) => total + Math.ceil(text.length / 4), 0), providerRequestId: `test-embedding-${this.calls}` };
  }
  async healthCheck(): Promise<{ healthy: boolean }> { return { healthy: true }; }
  private vector(text: string): number[] {
    const values = Array.from({ length: this.dimensions }, () => 0);
    for (const token of text.toLowerCase().split(/[^a-z0-9]+/).filter((value) => value.length > 2)) {
      const digest = createHash("sha256").update(token).digest(); const index = digest[0]! % this.dimensions; values[index]! += digest[1]! % 2 ? 1 : -1;
    }
    const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
    return values.map((value) => value / norm);
  }
}
