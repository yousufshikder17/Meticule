import type { AgentAction } from "../models/agent-action.js";import type{ModelProvider}from"../models/model-provider.js";import type{CanonicalMessageSchema,CanonicalToolDefinitionSchema,ModelResponse}from"../models/model-types.js";import type{z}from"zod";
export interface EngineTurnRequest{requestId:string;provider:ModelProvider;modelId:string;messages:z.infer<typeof CanonicalMessageSchema>[];tools:z.infer<typeof CanonicalToolDefinitionSchema>[];maxOutputTokens:number;timeoutMs:number;signal:AbortSignal;}
export interface EngineTurnResult{action:AgentAction;response:ModelResponse;}
export interface ExecutionEngine{readonly id:string;readonly capabilities:{boundedTurn:boolean;external:boolean};executeTurn(request:EngineTurnRequest):Promise<EngineTurnResult>;}
