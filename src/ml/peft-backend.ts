import {resolve} from "node:path";
import {ConflictError} from "../domain/errors.js";
import type {Snapshot} from "./domain.js";
import {NeuralRunnerBackend} from "./neural-backend.js";
import {LORA_LIMITS,PEFT_DEVICES} from "./peft-models.js";

export const PEFT_ARTIFACT_FORMAT="meticule-lora-adapter-v1";
export const PEFT_CHECKPOINT_FORMAT="peft-lora-checkpoint-v1";

/** Local adapters reuse the public epoch lifecycle, blob storage, registry and inference. */
export class PeftBackend extends NeuralRunnerBackend {
  constructor(python:string,timeoutMs=120_000,script=resolve(process.cwd(),"ml/lora_backend.py"),private readonly modelRoot="") {
    super({id:"peft",artifactFormat:PEFT_ARTIFACT_FORMAT,checkpointFormat:PEFT_CHECKPOINT_FORMAT,
      devices:PEFT_DEVICES,mixedPrecision:false,python,script,timeoutMs,tasks:["language_modeling"],inputs:["tokens"],preprocessing:[]});
  }
  protected override requestExtras():Record<string,unknown>{return {modelRoot:this.modelRoot};}
  override validate(snapshot:Snapshot):void {
    super.validate(snapshot);
    if(snapshot.dataset.rowCount>LORA_LIMITS.maxRows)throw new ConflictError(`LoRA accepts at most ${LORA_LIMITS.maxRows} sequences`);
    if((snapshot.hyperparameters.quantization??"none")!=="none"&&(snapshot.training?.device??"cpu")!=="cuda")
      throw new ConflictError("Four-bit adapter training requires CUDA");
  }
}
