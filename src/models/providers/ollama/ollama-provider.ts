import type { ModelProvider } from "../../model-provider.js";
import type { ModelRequest, ModelResponse } from "../../model-types.js";
import { ModelProviderError, mapHttpError } from "../../model-errors.js";
import { OllamaResponseSchema } from "./ollama-schemas.js";
import { providerFetch, resolveCapabilities, type AdapterSettings } from "../shared.js";

export class OllamaProvider implements ModelProvider {
  readonly id:string; constructor(private readonly settings:AdapterSettings){this.id=settings.id;}
  capabilities(modelId:string){return resolveCapabilities(this.settings,modelId,"local");}
  async healthCheck(signal:AbortSignal):Promise<{healthy:boolean;details:string|undefined}>{try{const response=await providerFetch(`${this.settings.baseUrl}/api/tags`,{},this.settings.defaultTimeoutMs,signal);return {healthy:response.ok,details:response.ok?undefined:`HTTP ${response.status}`};}catch(error){return {healthy:false,details:error instanceof Error?error.message:String(error)};}}
  async invoke(request:ModelRequest):Promise<ModelResponse>{
    const capabilities=this.capabilities(request.modelId); if(request.tools.length&&!capabilities.toolCalling) throw new ModelProviderError("invalid_request",`Model ${request.modelId} is not configured for tool calling`,false);
    if(request.outputSchema&&!capabilities.structuredOutput) throw new ModelProviderError("invalid_request",`Model ${request.modelId} is not configured for structured output`,false);
    const messages=request.messages.map((message)=>({role:message.role,content:message.content.filter((part)=>part.type==="text").map((part)=>part.type==="text"?part.text:"").join("\n"),...(message.toolCalls?{tool_calls:message.toolCalls.map((call)=>({function:{name:call.name,arguments:call.arguments}}))}:{} )}));
    const tools=request.tools.map((tool)=>({type:"function",function:{name:tool.name,description:tool.description,parameters:tool.inputSchema}}));
    const response=await providerFetch(`${this.settings.baseUrl}/api/chat`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({model:request.modelId,messages,stream:false,...(tools.length?{tools}:{}),...(request.outputSchema?{format:request.outputSchema}:{}),options:{num_predict:request.maxOutputTokens}})},request.timeoutMs,request.signal);
    if(!response.ok) throw mapHttpError(response.status,`Ollama request failed with HTTP ${response.status}`);
    const raw=OllamaResponseSchema.safeParse(await response.json()); if(!raw.success) throw new ModelProviderError("malformed_response","Ollama response failed validation",false,{issues:raw.error.issues.length});
    return {text:raw.data.message.content,toolCalls:(raw.data.message.tool_calls??[]).map((call,index)=>({id:`ollama-${request.requestId}-${index}`,name:call.function.name,arguments:call.function.arguments})),stopReason:raw.data.done_reason??(raw.data.done?"stop":"unknown"),usage:{inputTokens:raw.data.prompt_eval_count??0,outputTokens:raw.data.eval_count??0,cachedTokens:0},metadata:{provider:"ollama"}};
  }
}
