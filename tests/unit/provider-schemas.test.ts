import { describe,expect,it } from "vitest";
import { OllamaResponseSchema } from "../../src/models/providers/ollama/ollama-schemas.js";
import { AnthropicResponseSchema } from "../../src/models/providers/anthropic/anthropic-schemas.js";
import { GeminiResponseSchema } from "../../src/models/providers/gemini/gemini-schemas.js";
import { OpenAICompatibleResponseSchema } from "../../src/models/providers/openai-compatible/openai-compatible-schemas.js";
describe("provider raw-response contracts",()=>{
  it("validates Ollama usage and tool calls",()=>expect(OllamaResponseSchema.safeParse({model:"m",message:{role:"assistant",content:"",tool_calls:[{function:{name:"x",arguments:{a:1}}}]},done:true,prompt_eval_count:2,eval_count:3}).success).toBe(true));
  it("validates Anthropic content blocks",()=>expect(AnthropicResponseSchema.safeParse({id:"m",content:[{type:"tool_use",id:"t",name:"x",input:{a:1}}],stop_reason:"tool_use",usage:{input_tokens:2,output_tokens:3}}).success).toBe(true));
  it("validates Gemini parts",()=>expect(GeminiResponseSchema.safeParse({candidates:[{content:{parts:[{functionCall:{name:"x",args:{a:1}}}]}}],usageMetadata:{promptTokenCount:2}}).success).toBe(true));
  it("validates compatible tool arguments as raw strings",()=>expect(OpenAICompatibleResponseSchema.safeParse({id:"r",choices:[{finish_reason:"tool_calls",message:{content:null,tool_calls:[{id:"t",function:{name:"x",arguments:"{}"}}]}}],usage:{prompt_tokens:2,completion_tokens:3}}).success).toBe(true));
  it("rejects malformed provider payloads",()=>expect(OllamaResponseSchema.safeParse({done:true}).success).toBe(false));
});
