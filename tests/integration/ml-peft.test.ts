import pg from "pg";
import {randomUUID} from "node:crypto";
import {mkdtemp,readdir,rename,rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
import {afterAll,beforeAll,beforeEach,describe,expect,it} from "vitest";
import type {Principal} from "../../src/db/types.js";
import {RunRepository} from "../../src/db/repositories.js";
import {LifecycleWorker} from "../../src/worker/worker.js";
import {PostgresArtifactStore} from "../../src/ml/artifact-store.js";
import {PeftBackend} from "../../src/ml/peft-backend.js";
import {MlTrainingService} from "../../src/ml/persistence.js";
import {MlModelService} from "../../src/ml/model-service.js";
import {TrainingProcessor} from "../../src/ml/training-processor.js";
import type {MlBackend} from "../../src/ml/backend.js";

const cpu=process.env.ML_TEST_PEFT_PYTHON,gpu=process.env.ML_TEST_QLORA_PYTHON;
const url=process.env.TEST_DATABASE_URL;
const pool=new pg.Pool({connectionString:url,max:6});
const manager:Principal={tenantId:"11111111-1111-4111-8111-111111111111",userId:"22222222-2222-4222-8222-222222222222",roles:["tenant_admin"]};
const outsider:Principal={...manager,tenantId:"33333333-3333-4333-8333-333333333333"};
const rows=Array.from({length:60},(_,i)=>({seq:Array.from({length:8+i%5},(_,j)=>String((i*3+j)%8)).join(" ")}));
let store:PostgresArtifactStore;
beforeAll(async()=>{if(!url)return;await pool.query("SELECT 1 FROM ml_training_checkpoints LIMIT 1");store=new PostgresArtifactStore(pool);});
beforeEach(async()=>{if(url)await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE");});
afterAll(async()=>{await pool.end();});

async function fixture(python=cpu!,device="cpu",quantization="none",root=""){
  const backend=new PeftBackend(python,180_000,undefined,root);
  const map=new Map<string,MlBackend>([["peft",backend]]);
  const training=new MlTrainingService(pool,store),models=new MlModelService(pool,store,map);
  const dataset=await training.ingest(manager,{name:"local tokens",schema:{id:randomUUID(),columns:[{name:"seq",type:"tokens"}]},rows});
  const pipeline=await training.pipeline(manager,{name:"tokens",definition:{features:["seq"],steps:[]}});
  const experiment=await training.experiment(manager,"local adapters");
  const job=(extra:Record<string,unknown>={})=>training.queue(manager,{experimentId:experiment.id,spec:{datasetVersionId:dataset.id,featurePipelineId:pipeline.id,backend:"peft",algorithm:"lora_causal_lm",seed:7,
    hyperparameters:{epochs:6,batchSize:8,dropout:.1,learningRate:.02,quantization,...extra},evaluation:{metrics:["token_cross_entropy","perplexity","next_token_accuracy"]},
    training:{checkpointEveryEpochs:2,earlyStopping:null,device,precision:"fp32"},split:{id:"44444444-4444-4444-8444-444444444444",strategy:"random",train:.6,validation:.2,test:.2,stratify:false}}});
  const drain=async(workerId="public-peft")=>{const worker=new LifecycleWorker(pool,{workerId,leaseSeconds:300,kind:"ml"},new TrainingProcessor(pool,store,map));while(await worker.tick());};
  const checkpoints=async(id:string)=>(await pool.query("SELECT c.*,t.attempt FROM ml_training_checkpoints c JOIN ml_training_runs t ON t.id=c.training_run_id WHERE t.job_id=$1 ORDER BY t.attempt,c.epoch",[id])).rows;
  const hook=(action:()=>Promise<unknown>)=>{let calls=0;const original=backend.trainChunk.bind(backend);backend.trainChunk=async(...args)=>{const result=await original(...args);if(++calls===2)await action();return result;};return ()=>{backend.trainChunk=original;};};
  const interrupt=async(runId:string)=>{const restore=hook(()=>pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1",[runId]));
    // One tick preserves the abandoned attempt; draining would immediately recover and complete it.
    try{await new LifecycleWorker(pool,{workerId:"expired-worker",leaseSeconds:300,kind:"ml"},new TrainingProcessor(pool,store,map)).tick();}
    catch(error){expect((error as Error).message).toBe("Worker lease was lost during execution");}finally{restore();}};
  const serve=async(attemptId:string)=>{const entry=await models.registry(manager,{name:"local adapter"});const version=await models.register(manager,entry.id,{trainingRunId:attemptId});
    await expect(models.endpoint(manager,{name:"premature",modelVersionId:version.id})).rejects.toThrow(/READY/);
    await models.transition(manager,version.id,"READY");const endpoint=await models.endpoint(manager,{name:"adapter",modelVersionId:version.id});return {entry,version,endpoint};};
  return {training,models,map,job,drain,checkpoints,hook,interrupt,serve};
}

async function corrupt(key:string){
  const client=await pool.connect();try{await client.query("BEGIN");await client.query("ALTER TABLE ml_blobs DISABLE TRIGGER ml_blob_append_only");
    await client.query("UPDATE ml_blobs SET content=$3 WHERE tenant_id=$1 AND content_hash=$2",[manager.tenantId,key,Buffer.from("injected corruption")]);
    await client.query("ALTER TABLE ml_blobs ENABLE TRIGGER ml_blob_append_only");await client.query("COMMIT");
  }catch(error){await client.query("ROLLBACK");throw error;}finally{client.release();}
}

for(const profile of [{name:"CPU LoRA",python:cpu,device:"cpu",quantization:"none"},{name:"CUDA NF4",python:gpu,device:"cuda",quantization:"nf4"},{name:"CUDA FP4",python:gpu,device:"cuda",quantization:"fp4"}]){
  it.runIf(Boolean(url&&profile.python))(`${profile.name}: real learning, fenced recovery, immutable adapter registration and inference`,async()=>{
    const f=await fixture(profile.python!,profile.device,profile.quantization);
    const reference=await f.job();await f.drain();const ref=await f.training.getJob(manager,reference.id);expect(ref.job.status).toBe("COMPLETED");
    const job=await f.job();await f.interrupt(job.run_id);const interrupted=await f.training.getJob(manager,job.id);
    expect(interrupted.job.status).toBe("TRAINING");expect(interrupted.artifacts).toHaveLength(0);expect((await f.checkpoints(job.id)).map(c=>c.epoch)).toEqual([2]);
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);await f.drain("replacement");const result=await f.training.getJob(manager,job.id);
    expect(result.job.status).toBe("COMPLETED");expect(result.attempts[1].resumed_from_epoch).toBe(2);
    expect(result.artifacts[0].content.sha256).toBe(ref.artifacts[0].content.sha256);
    const measured=(metrics:Record<string,unknown>[])=>metrics.map(m=>({name:m.name,partition:m.partition,value:m.value})).sort((a,b)=>String(a.partition)+String(a.name)<String(b.partition)+String(b.name)?-1:1);
    expect(measured(result.metrics)).toEqual(measured(ref.metrics));
    expect(result.attempts[1].performance.finalTrainLoss).toBeLessThan(result.attempts[1].performance.initialTrainLoss*.5);
    expect(result.attempts[1].environment).toMatchObject({device_actual:profile.device,quantization:profile.quantization,artifact_kind:"lora_adapter"});
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
    const {entry,version,endpoint}=await f.serve(result.attempts[1].id);
    const prediction=await f.models.predict(manager,endpoint.id,[{seq:"0 1 2 3 4 5"}],new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED");expect(prediction.output[0]).toHaveLength(6);
    await expect(f.models.predict(outsider,endpoint.id,[{seq:"0 1"}],new AbortController().signal)).rejects.toThrow();
    await expect(f.models.register(outsider,entry.id,{trainingRunId:result.attempts[1].id})).rejects.toThrow();
    await expect(f.models.transition({...manager,roles:[]},version.id,"RETIRED")).rejects.toThrow();
    await expect(store.get(outsider.tenantId,result.artifacts[0].content)).rejects.toThrow();
    await expect(f.training.checkpoint(job.run_id,"expired-worker",interrupted.attempts[0].id,[],{epoch:4,content:result.artifacts[0].content,format:"peft-lora-checkpoint-v1",schemaVersion:1})).rejects.toThrow();
    await corrupt(result.artifacts[0].content.key);
    const failed=await f.models.predict(manager,endpoint.id,[{seq:"0 1"}],new AbortController().signal);expect(failed.status).toBe("FAILED");expect(failed.output).toBeNull();
    expect(await new RunRepository(pool).recoverExpired()).toBe(0);
  },600_000);
}

describe.runIf(Boolean(cpu&&url))("public adapter failure and dependency boundaries",()=>{
  it("rejects a corrupted committed checkpoint without restarting or publishing",async()=>{
    const f=await fixture();const job=await f.job();await f.interrupt(job.run_id);await corrupt((await f.checkpoints(job.id))[0].content.key);
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);await f.drain();const result=await f.training.getJob(manager,job.id);
    expect(result.job.status).toBe("FAILED");expect(result.artifacts).toHaveLength(0);expect(result.metrics).toHaveLength(0);
  },600_000);
  it("cancels a real chunk at the canonical boundary and rejects its uncommitted progress",async()=>{
    const f=await fixture();const job=await f.job();f.hook(()=>new RunRepository(pool).requestCancellation(manager,job.run_id));await f.drain();
    const result=await f.training.getJob(manager,job.id);expect(result.job.status).toBe("CANCELLED");expect(result.artifacts).toHaveLength(0);
    expect((await f.checkpoints(job.id)).map(c=>c.epoch)).toEqual([2]);
  },600_000);
  it("serves an approved provisioned base, fails when missing, and serves again after restoration",async()=>{
    const root=await mkdtemp(join(tmpdir(),"meticule-public-base-"));
    try{
      await promisify(execFile)(cpu!,["-B","-c","import sys;sys.path.insert(0,'tests/python');from lora_provisioning_fixtures import build;build(sys.argv[1])",root],{timeout:120_000});
      const f=await fixture(cpu!,"cpu","none",root);const job=await f.job({baseModel:"provisioned-llama",pretrainedModel:"approved-llama"});await f.drain();
      const result=await f.training.getJob(manager,job.id);expect(result.job.status).toBe("COMPLETED");const {endpoint}=await f.serve(result.attempts[0].id);
      const predict=()=>f.models.predict(manager,endpoint.id,[{seq:"0 1 2"}],new AbortController().signal);
      expect((await predict()).status).toBe("COMPLETED");const base=join(root,"approved-llama");await rename(base,base+"-hidden");
      try{const failed=await predict();expect(failed.status).toBe("FAILED");expect(failed.output).toBeNull();}finally{await rename(base+"-hidden",base);}
      expect((await predict()).status).toBe("COMPLETED");
    }finally{await rm(root,{recursive:true,force:true});}
  },600_000);
});

for(const profile of [{name:"CPU",python:cpu,device:"cpu",quantization:"none",provisioned:false},{name:"CUDA NF4",python:gpu,device:"cuda",quantization:"nf4",provisioned:true}]){
  it.runIf(Boolean(url&&profile.python))(`${profile.name}: real isolated scratch training and registered inference clean all per-call files`,async()=>{
    const scratch=async()=>(await readdir(tmpdir())).filter(name=>name.startsWith("meticule-runner-")).sort();const before=await scratch();
    const root=profile.provisioned?await mkdtemp(join(tmpdir(),"meticule-public-quantized-base-")):"";
    try{
    if(root)await promisify(execFile)(profile.python!,["-B","-c","import sys;sys.path.insert(0,'tests/python');from lora_provisioning_fixtures import build;build(sys.argv[1])",root],{timeout:120_000});
    const f=await fixture(profile.python!,profile.device,profile.quantization,root);const job=await f.job({epochs:2,...(root?{baseModel:"provisioned-llama",pretrainedModel:"approved-llama"}:{})});await f.drain();
    const result=await f.training.getJob(manager,job.id);expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].performance.finalTrainLoss).toBeLessThan(result.attempts[0].performance.initialTrainLoss);
    expect(result.attempts[0].environment).toMatchObject({device_actual:profile.device,quantization:profile.quantization,base_model:root?"provisioned-llama":"meticule-tiny-llama-v1"});
    if(root){expect(result.attempts[0].environment.base_model_name).toBe("approved-llama");expect(result.attempts[0].environment.tokenizer_digest).toMatch(/^[a-f0-9]{64}$/);}
    const {endpoint}=await f.serve(result.attempts[0].id);
    expect((await f.models.predict(manager,endpoint.id,[{seq:"0 1 2"}],new AbortController().signal)).status).toBe("COMPLETED");
    expect(await scratch()).toEqual(before);
    }finally{if(root)await rm(root,{recursive:true,force:true});}
  },300_000);
}
