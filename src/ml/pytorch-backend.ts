import { resolve } from "node:path";
import { NeuralRunnerBackend } from "./neural-backend.js";
import { PYTORCH_DEVICES, PYTORCH_MIXED_PRECISION } from "./pytorch-models.js"; // also registers the neural definitions under `pytorch`

export const PYTORCH_ARTIFACT_FORMAT = "pytorch-state-dict-v1";
export const PYTORCH_CHECKPOINT_FORMAT = "pytorch-checkpoint-v1";

/**
 * PyTorch backend, on CPU or explicitly requested CUDA. Everything beyond the id and formats lives in NeuralRunnerBackend
 * and ml/pytorch_runner.py.
 */
export class PytorchBackend extends NeuralRunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/pytorch_runner.py")) {
    super({ id: "pytorch", artifactFormat: PYTORCH_ARTIFACT_FORMAT, checkpointFormat: PYTORCH_CHECKPOINT_FORMAT, devices: PYTORCH_DEVICES, mixedPrecision: PYTORCH_MIXED_PRECISION, python, script, timeoutMs });
  }
}
