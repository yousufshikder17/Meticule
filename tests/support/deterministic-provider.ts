import type { ModelProvider } from "../../src/models/model-provider.js";
import type { ModelRequest,ModelResponse,ProviderCapabilities } from "../../src/models/model-types.js";
import { ModelProviderError } from "../../src/models/model-errors.js";

export interface DeterministicReply { body?: unknown | (() => unknown | Promise<unknown>); toolCalls?: ModelResponse["toolCalls"]; usage?: Partial<ModelResponse["usage"]>; error?: ModelProviderError; beforeReturn?: () => Promise<void> }
export class DeterministicTestProvider implements ModelProvider {
  readonly id:string; calls=0; readonly requests:ModelRequest[]=[];
  constructor(private readonly replies:DeterministicReply[],private readonly caps:Partial<ProviderCapabilities>={},id="test-deterministic"){this.id=id;}
  capabilities():ProviderCapabilities{return{toolCalling:true,structuredOutput:true,streaming:false,vision:false,tokenUsage:true,contextWindow:10_000,nativeIdempotency:false,execution:"local",...this.caps};}
  async healthCheck(){return{healthy:true,details:undefined};}
  async invoke(request:ModelRequest):Promise<ModelResponse>{this.requests.push(request);const reply=this.replies[this.calls++];if(!reply)throw new ModelProviderError("malformed_response","No deterministic reply configured",false);if(reply.beforeReturn)await reply.beforeReturn();if(reply.error)throw reply.error;const body=typeof reply.body==="function"?await reply.body():reply.body??{};return{text:JSON.stringify(body),toolCalls:reply.toolCalls??[],stopReason:"stop",usage:{inputTokens:reply.usage?.inputTokens??10,outputTokens:reply.usage?.outputTokens??5,cachedTokens:reply.usage?.cachedTokens??0},providerRequestId:`test-${this.calls}`,metadata:{testOnly:true}};}
}
