import type { ModelProvider } from "./model-provider.js";
import { ProviderIdSchema } from "./model-types.js";

export class ProviderRegistry {
  private readonly providers = new Map<string, ModelProvider>();
  register(provider: ModelProvider): void { ProviderIdSchema.parse(provider.id); if (this.providers.has(provider.id)) throw new Error(`Duplicate provider: ${provider.id}`); this.providers.set(provider.id, provider); }
  get(id: string): ModelProvider { ProviderIdSchema.parse(id); const provider=this.providers.get(id); if(!provider) throw new Error(`Unsupported provider: ${id}`); return provider; }
  list(): string[] { return [...this.providers.keys()]; }
}
