import type { ModelProvider } from "./model-provider.js";

export interface ProviderConfiguration {
  id: string; type: string; enabled: boolean; baseUrl?: string; secret?: string; allowedModels: string[];
  modelCapabilities: Record<string, Record<string, unknown>>; defaultTimeoutMs: number;
}
export interface ProviderFactory { readonly type: string; create(configuration: ProviderConfiguration): ModelProvider }
