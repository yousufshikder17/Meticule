import { describe,expect,it } from "vitest";
import { ProviderIdSchema } from "../../src/models/model-types.js";
import { ProviderRegistry } from "../../src/models/provider-registry.js";
import { ProviderFactoryRegistry } from "../../src/models/provider-factory-registry.js";
import { AgentActionSchema } from "../../src/models/agent-action.js";
import { mapHttpError } from "../../src/models/model-errors.js";
import { validateComposition } from "../../src/execution/composition.js";
import { calculateCostMicrousd } from "../../src/models/cost.js";
import { DeterministicTestProvider } from "../support/deterministic-provider.js";
import { loadProviderConfigurations } from "../../src/models/provider-configuration.js";

describe("provider-neutral model foundation",()=>{
  it("validates open-ended provider IDs",()=>{expect(ProviderIdSchema.parse("future.deepseek-v2")).toBe("future.deepseek-v2");expect(()=>ProviderIdSchema.parse("Bad Provider")).toThrow();});
  it("registers providers without a central switch and rejects duplicates/unknown IDs",()=>{const registry=new ProviderRegistry();registry.register(new DeterministicTestProvider([]));expect(registry.get("test-deterministic").id).toBe("test-deterministic");expect(()=>registry.register(new DeterministicTestProvider([]))).toThrow(/Duplicate/);expect(()=>registry.get("missing")).toThrow(/Unsupported/);});
  it("constructs providers through registered factories",()=>{const factories=new ProviderFactoryRegistry();factories.register({type:"custom",create:()=>new DeterministicTestProvider([])});expect(factories.create({id:"x",type:"custom",enabled:true,allowedModels:[],modelCapabilities:{},defaultTimeoutMs:1}).id).toBe("test-deterministic");expect(()=>factories.create({id:"x",type:"missing",enabled:true,allowedModels:[],modelCapabilities:{},defaultTimeoutMs:1})).toThrow(/Unsupported/);});
  it("validates every canonical action discriminator",()=>{for(const action of [{type:"call_tool",toolName:"calculator",arguments:{}},{type:"final_answer",output:"ok"},{type:"request_clarification",question:"which?"},{type:"update_plan",plan:{objective:"finish",tasks:[{taskId:"first",objective:"do the work"}]}},{type:"pause",reason:"wait"}])expect(AgentActionSchema.safeParse(action).success).toBe(true);expect(AgentActionSchema.safeParse({type:"complete"}).success).toBe(false);});
  it("normalizes retryability and calculates integer micro-USD cost",()=>{expect(mapHttpError(429).retryable).toBe(true);expect(mapHttpError(401).retryable).toBe(false);expect(calculateCostMicrousd({inputTokens:1_000_000,outputTokens:500_000,cachedTokens:200_000},{inputCostMicrousdPerMillion:1000,outputCostMicrousdPerMillion:2000,cachedCostMicrousdPerMillion:100})).toBe(1820);});
  it("accepts only supported native composition choices",()=>{expect(validateComposition({}).executionEngine).toBe("native");expect(validateComposition({planner:"native"}).planner).toBe("native");expect(()=>validateComposition({executionEngine:"langchain"})).toThrow();expect(()=>validateComposition({retriever:"hybrid"})).toThrow();});
  it("treats blank optional provider settings as unset when providers are disabled",()=>{expect(loadProviderConfigurations({OLLAMA_ENABLED:"false",OPENAI_COMPAT_ENABLED:"false",OPENAI_COMPAT_BASE_URL:"",OPENAI_COMPAT_API_KEY:"",ANTHROPIC_ENABLED:"false",ANTHROPIC_API_KEY:"",GEMINI_ENABLED:"false",GEMINI_API_KEY:""})).toEqual([]);});
});
