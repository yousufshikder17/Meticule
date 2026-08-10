import { z } from "zod";
import type { ProviderConfiguration } from "./provider-factory.js";
import { ProviderFactoryRegistry } from "./provider-factory-registry.js";
import { ProviderRegistry } from "./provider-registry.js";
import { OllamaProviderFactory } from "./providers/ollama/ollama-factory.js";
import { OpenAICompatibleProviderFactory } from "./providers/openai-compatible/openai-compatible-factory.js";
import { AnthropicProviderFactory } from "./providers/anthropic/anthropic-factory.js";
import { GeminiProviderFactory } from "./providers/gemini/gemini-factory.js";

const bool=(fallback:boolean)=>z.string().optional().transform(v=>v===undefined?fallback:v.toLowerCase()==="true");
const optionalUrl=z.preprocess((value)=>typeof value==="string"&&!value.trim()?undefined:value,z.string().url().optional());
const EnvSchema=z.object({
  OLLAMA_ENABLED:bool(true),OLLAMA_BASE_URL:z.string().url().default("http://localhost:11434"),OLLAMA_MODELS:z.string().default(""),OLLAMA_MODEL_CAPABILITIES_JSON:z.string().default("{}"),
  OPENAI_COMPAT_ENABLED:bool(false),OPENAI_COMPAT_BASE_URL:optionalUrl,OPENAI_COMPAT_MODELS:z.string().default(""),OPENAI_COMPAT_API_KEY:z.string().optional(),OPENAI_COMPAT_MODEL_CAPABILITIES_JSON:z.string().default("{}"),
  ANTHROPIC_ENABLED:bool(false),ANTHROPIC_MODELS:z.string().default(""),ANTHROPIC_API_KEY:z.string().optional(),ANTHROPIC_MODEL_CAPABILITIES_JSON:z.string().default("{}"),
  GEMINI_ENABLED:bool(false),GEMINI_MODELS:z.string().default(""),GEMINI_API_KEY:z.string().optional(),GEMINI_MODEL_CAPABILITIES_JSON:z.string().default("{}"),
});
const models=(value:string)=>value.split(",").map(v=>v.trim()).filter(Boolean);const capabilities=(value:string)=>z.record(z.string(),z.record(z.string(),z.unknown())).parse(JSON.parse(value));
export function loadProviderConfigurations(env:NodeJS.ProcessEnv=process.env):ProviderConfiguration[]{const v=EnvSchema.parse(env);const configs:ProviderConfiguration[]=[];
  if(v.OLLAMA_ENABLED&&models(v.OLLAMA_MODELS).length)configs.push({id:"ollama",type:"ollama",enabled:true,baseUrl:v.OLLAMA_BASE_URL,allowedModels:models(v.OLLAMA_MODELS),modelCapabilities:capabilities(v.OLLAMA_MODEL_CAPABILITIES_JSON),defaultTimeoutMs:60_000});
  if(v.OPENAI_COMPAT_ENABLED){if(!v.OPENAI_COMPAT_BASE_URL)throw new Error("OPENAI_COMPAT_BASE_URL is required");configs.push({id:"openai-compatible",type:"openai-compatible",enabled:true,baseUrl:v.OPENAI_COMPAT_BASE_URL,allowedModels:models(v.OPENAI_COMPAT_MODELS),modelCapabilities:capabilities(v.OPENAI_COMPAT_MODEL_CAPABILITIES_JSON),defaultTimeoutMs:60_000,...(v.OPENAI_COMPAT_API_KEY?{secret:v.OPENAI_COMPAT_API_KEY}:{})});}
  if(v.ANTHROPIC_ENABLED){if(!v.ANTHROPIC_API_KEY)throw new Error("ANTHROPIC_API_KEY is required when Anthropic is enabled");configs.push({id:"anthropic",type:"anthropic",enabled:true,allowedModels:models(v.ANTHROPIC_MODELS),modelCapabilities:capabilities(v.ANTHROPIC_MODEL_CAPABILITIES_JSON),defaultTimeoutMs:60_000,secret:v.ANTHROPIC_API_KEY});}
  if(v.GEMINI_ENABLED){if(!v.GEMINI_API_KEY)throw new Error("GEMINI_API_KEY is required when Gemini is enabled");configs.push({id:"gemini",type:"gemini",enabled:true,allowedModels:models(v.GEMINI_MODELS),modelCapabilities:capabilities(v.GEMINI_MODEL_CAPABILITIES_JSON),defaultTimeoutMs:60_000,secret:v.GEMINI_API_KEY});}
  return configs;}
export function createProviderRegistry(env:NodeJS.ProcessEnv=process.env):ProviderRegistry{const factories=new ProviderFactoryRegistry();factories.register(new OllamaProviderFactory());factories.register(new OpenAICompatibleProviderFactory());factories.register(new AnthropicProviderFactory());factories.register(new GeminiProviderFactory());const registry=new ProviderRegistry();for(const config of loadProviderConfigurations(env))registry.register(factories.create(config));return registry;}
