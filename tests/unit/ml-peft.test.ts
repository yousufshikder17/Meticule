import {describe,expect,it} from "vitest";
import {PeftBackend} from "../../src/ml/peft-backend.js";
import {mlConfiguration} from "../../src/ml/configuration.js";
import {getModel,validateHyperparameters} from "../../src/ml/model-catalog.js";
import {TrainingSpec,type Snapshot} from "../../src/ml/domain.js";
import {snapshotFor} from "./ml-fixtures.js";

const snapshot=(hyperparameters:Record<string,unknown>={}):Snapshot=>snapshotFor({backend:"peft",algorithm:"lora_causal_lm",columns:[{name:"seq",type:"tokens"}],features:["seq"],target:null,hyperparameters});
describe("public local adapter configuration",()=>{
  it("is opt-in and keeps every existing backend disabled by default",()=>{
    expect(mlConfiguration().backends.size).toBe(0);
    expect(mlConfiguration({ML_ENABLED:"true"}).backends.has("peft")).toBe(false);
    expect(mlConfiguration({ML_ENABLED:"true",ML_PEFT_ENABLED:"true",ML_PEFT_PYTHON:"unused"}).backends.get("peft")).toBeInstanceOf(PeftBackend);
    expect(()=>mlConfiguration({ML_ENABLED:"true",ML_PEFT_ENABLED:"true",ML_TRANSFORMERS_MODEL_ROOT:"relative"})).toThrow(/absolute/);
  });
  it("validates the exact adapter artifact and checkpoint contracts",()=>{
    const backend=new PeftBackend("unused");backend.validate(snapshot());
    expect(backend.capabilities).toMatchObject({artifactFormats:["meticule-lora-adapter-v1"],checkpointFormats:["peft-lora-checkpoint-v1"],tasks:["language_modeling"],preprocessing:[],inputs:["tokens"]});
    for(const hp of [{rank:17},{quantization:"int8"},{computeDtype:"fp16"},{doubleQuantization:true},{targetModules:["q_proj","q_proj"]},{baseModel:"provisioned-llama"},{pretrainedModel:"../escape"},{learningRate:NaN},{epochs:Infinity}])expect(()=>backend.validate(snapshot(hp))).toThrow();
    expect(()=>backend.validate(snapshot({quantization:"nf4"}))).toThrow(/CUDA/);
    backend.validate({...snapshot({quantization:"fp4",computeDtype:"bf16",doubleQuantization:true}),training:{checkpointEveryEpochs:1,earlyStopping:null,device:"cuda",precision:"fp32"}});
    expect(()=>backend.validate({...snapshot(),dataset:{...snapshot().dataset,rowCount:1025}})).toThrow();
  });
  it("rejects private execution and distillation requests rather than silently dropping them",()=>{
    for(const extension of [{distillation:{}},{execution:{backend:"ray"}},{teacher:{}}])expect(()=>TrainingSpec.parse({...snapshot(),...extension})).toThrow();
    const model=getModel("peft","lora_causal_lm");
    expect(()=>validateHyperparameters(model,{baseModel:"provisioned-llama",pretrainedModel:"approved-model"})).not.toThrow();
  });
});
