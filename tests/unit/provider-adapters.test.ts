import { afterEach,describe,expect,it,vi } from "vitest";
import type { ModelRequest } from "../../src/models/model-types.js";
import { OllamaProvider } from "../../src/models/providers/ollama/ollama-provider.js";
import { OpenAICompatibleProvider } from "../../src/models/providers/openai-compatible/openai-compatible-provider.js";
import { AnthropicProvider } from "../../src/models/providers/anthropic/anthropic-provider.js";
import { GeminiProvider } from "../../src/models/providers/gemini/gemini-provider.js";

const request:ModelRequest={requestId:"11111111-1111-4111-8111-111111111111",providerId:"p",modelId:"model",messages:[{role:"user",content:[{type:"text",text:"hello"}]}],tools:[],maxOutputTokens:10,timeoutMs:1000,signal:new AbortController().signal,metadata:{}};
const settings={id:"p",baseUrl:"http://provider.test",allowedModels:["model"],modelCapabilities:{model:{tokenUsage:true}},defaultTimeoutMs:1000,secret:"test-secret"};
afterEach(()=>vi.unstubAllGlobals());
const respond=(body:unknown)=>vi.stubGlobal("fetch",vi.fn(async()=>new Response(JSON.stringify(body),{status:200,headers:{"content-type":"application/json"}})));

describe("provider adapter contract normalization",()=>{
  it("normalizes Ollama",async()=>{respond({model:"model",message:{role:"assistant",content:"{\"type\":\"final_answer\",\"output\":\"ok\"}"},done:true,done_reason:"stop",prompt_eval_count:4,eval_count:2});const result=await new OllamaProvider(settings).invoke(request);expect(result.usage).toEqual({inputTokens:4,outputTokens:2,cachedTokens:0});expect(result.stopReason).toBe("stop");});
  it("normalizes OpenAI-compatible tool calls",async()=>{respond({id:"req",choices:[{finish_reason:"tool_calls",message:{content:null,tool_calls:[{id:"call",function:{name:"calculator",arguments:"{\"expression\":\"2+2\"}"}}]}}],usage:{prompt_tokens:4,completion_tokens:2,prompt_tokens_details:{cached_tokens:1}}});const result=await new OpenAICompatibleProvider(settings).invoke(request);expect(result.toolCalls[0]).toEqual({id:"call",name:"calculator",arguments:{expression:"2+2"}});expect(result.providerRequestId).toBe("req");});
  it("normalizes Anthropic content blocks",async()=>{respond({id:"msg",content:[{type:"text",text:"hi"},{type:"tool_use",id:"call",name:"calculator",input:{expression:"2+2"}}],stop_reason:"tool_use",usage:{input_tokens:4,output_tokens:2,cache_read_input_tokens:1}});const result=await new AnthropicProvider(settings).invoke(request);expect(result.text).toBe("hi");expect(result.toolCalls[0]?.name).toBe("calculator");expect(result.usage.cachedTokens).toBe(1);});
  it("normalizes Gemini parts",async()=>{respond({candidates:[{finishReason:"STOP",content:{parts:[{text:"hi"},{functionCall:{id:"call",name:"calculator",args:{expression:"2+2"}}}]}}],usageMetadata:{promptTokenCount:4,candidatesTokenCount:2,cachedContentTokenCount:1}});const result=await new GeminiProvider(settings).invoke(request);expect(result.text).toBe("hi");expect(result.toolCalls[0]?.id).toBe("call");expect(result.usage.outputTokens).toBe(2);});
  it("enforces model allowlists in every adapter",()=>{for(const provider of [new OllamaProvider(settings),new OpenAICompatibleProvider(settings),new AnthropicProvider(settings),new GeminiProvider(settings)])expect(()=>provider.capabilities("not-allowed")).toThrow(/allowlisted/);});
});
