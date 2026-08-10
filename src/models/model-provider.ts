import type { ModelRequest, ModelResponse, ProviderCapabilities } from "./model-types.js";

export interface ModelProvider {
  readonly id: string;
  capabilities(modelId: string): ProviderCapabilities;
  invoke(request: ModelRequest): Promise<ModelResponse>;
  healthCheck(signal: AbortSignal): Promise<{ healthy: boolean; details: string | undefined }>;
}
