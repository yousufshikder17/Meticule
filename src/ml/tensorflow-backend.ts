import { resolve } from "node:path";
import { NeuralRunnerBackend } from "./neural-backend.js";
import { TENSORFLOW_DEVICES, TENSORFLOW_MIXED_PRECISION } from "./tensorflow-models.js"; // also registers the neural definitions under `tensorflow`

export const TENSORFLOW_ARTIFACT_FORMAT = "tensorflow-variables-v1";
export const TENSORFLOW_CHECKPOINT_FORMAT = "tensorflow-checkpoint-v1";

/** TensorFlow (Keras) backend on CPU. Everything beyond the id and formats lives in NeuralRunnerBackend and ml/tensorflow_runner.py. */
export class TensorflowBackend extends NeuralRunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/tensorflow_runner.py")) {
    super({ id: "tensorflow", artifactFormat: TENSORFLOW_ARTIFACT_FORMAT, checkpointFormat: TENSORFLOW_CHECKPOINT_FORMAT, devices: TENSORFLOW_DEVICES, mixedPrecision: TENSORFLOW_MIXED_PRECISION, python, script, timeoutMs });
  }
}
