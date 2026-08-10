import type { ProviderConfiguration, ProviderFactory } from "./provider-factory.js";
import type { ModelProvider } from "./model-provider.js";

export class ProviderFactoryRegistry {
  private readonly factories = new Map<string, ProviderFactory>();
  register(factory: ProviderFactory): void { if(this.factories.has(factory.type)) throw new Error(`Duplicate provider factory: ${factory.type}`); this.factories.set(factory.type,factory); }
  create(configuration: ProviderConfiguration): ModelProvider { const factory=this.factories.get(configuration.type); if(!factory) throw new Error(`Unsupported provider type: ${configuration.type}`); return factory.create(configuration); }
}
