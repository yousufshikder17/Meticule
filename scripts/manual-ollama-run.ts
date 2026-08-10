import { getPool,closePool } from "../src/db/pool.js";
import { AgentRepository,RunRepository } from "../src/db/repositories.js";
import { createProviderRegistry } from "../src/models/provider-configuration.js";
import { ExecutionEngineRegistry } from "../src/execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../src/execution/native-engine.js";
import { createToolRegistry } from "../src/tools/registry.js";
import { AgentLoop } from "../src/execution/agent-loop.js";
import { LifecycleWorker } from "../src/worker/worker.js";

const pool=getPool();
try{
  const principal={tenantId:"99999999-9999-4999-8999-999999999999",userId:"aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",roles:[]};
  const model=process.env.OLLAMA_MANUAL_MODEL??"llama3.1:8b";
  const providers=createProviderRegistry();const provider=providers.get("ollama");const health=await provider.healthCheck(AbortSignal.timeout(5000));if(!health.healthy)throw new Error(`Ollama unhealthy: ${health.details}`);
  const agent=await new AgentRepository(pool).create(principal,{name:`ollama-manual-${Date.now()}`,systemInstructions:"Follow the canonical JSON action protocol. For this goal, return one final_answer action and do not request tools.",model:{provider:"ollama",model,maxOutputTokens:256,timeoutMs:120_000,inputCostMicrousdPerMillion:0,outputCostMicrousdPerMillion:0,cachedCostMicrousdPerMillion:0},allowedTools:[],maximumSteps:3,tokenBudget:4096,costBudgetMicrousd:1,approvalPolicy:{},outputSchema:null});
  const run=await new RunRepository(pool).create(principal,agent.id,"Return a concise greeting in a final_answer action.");
  const engines=new ExecutionEngineRegistry();engines.register(new NativeExecutionEngine());const registry=createToolRegistry(pool);const loop=new AgentLoop(pool,providers,engines,registry);const worker=new LifecycleWorker(pool,{workerId:`manual-${process.pid}`,leaseSeconds:180},{execute:(id,workerId)=>loop.execute(id,workerId)});
  await worker.tick();const completed=await new RunRepository(pool).get(principal.tenantId,run.id);const attempts=await pool.query("SELECT provider_id,model_id,status,input_tokens,output_tokens,cost_microusd FROM model_attempts WHERE run_id=$1 ORDER BY created_at",[run.id]);
  console.log(JSON.stringify({runId:run.id,status:completed.status,finalOutput:completed.finalOutput,attempts:attempts.rows},null,2));if(completed.status!=="completed")process.exitCode=1;
}finally{await closePool();}
