import type { ProviderCapabilities } from "../model-types.js";
import { ModelProviderError, normalizeTransportError } from "../model-errors.js";

export interface AdapterSettings { id: string; baseUrl: string; allowedModels: string[]; modelCapabilities: Record<string, Partial<ProviderCapabilities>>; defaultTimeoutMs: number; secret?: string }
export const baseCapabilities = (execution: "local"|"remote"): ProviderCapabilities => ({ toolCalling:false,structuredOutput:false,streaming:false,vision:false,tokenUsage:true,contextWindow:null,nativeIdempotency:false,execution });
export function resolveCapabilities(settings: AdapterSettings, model: string, execution: "local"|"remote"): ProviderCapabilities {
  if (!settings.allowedModels.includes(model)) throw new ModelProviderError("model_not_found", `Model is not allowlisted: ${model}`, false);
  return { ...baseCapabilities(execution), ...settings.modelCapabilities[model] };
}
export async function providerFetch(url: string, init: RequestInit, timeoutMs: number, signal: AbortSignal): Promise<Response> {
  const controller = new AbortController(); const timer=setTimeout(()=>controller.abort(new Error("Provider timeout")), timeoutMs);
  const abort=()=>controller.abort(signal.reason); signal.addEventListener("abort",abort,{once:true});
  try { return await fetch(url,{...init,signal:controller.signal}); }
  catch(error){ if(controller.signal.aborted && !signal.aborted) throw new ModelProviderError("timeout","Provider call timed out",true); throw normalizeTransportError(error); }
  finally { clearTimeout(timer); signal.removeEventListener("abort",abort); }
}
