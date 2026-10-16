import { resolve } from "node:path";
import { JAX_DEVICES, JAX_MIXED_PRECISION } from "./jax-models.js"; // also registers the neural definitions under `jax`
import { NeuralRunnerBackend } from "./neural-backend.js";

export const JAX_ARTIFACT_FORMAT = "jax-params-v1";
export const JAX_CHECKPOINT_FORMAT = "jax-checkpoint-v1";

/** Native JAX backend on CPU. Everything beyond the id and formats lives in NeuralRunnerBackend and ml/jax_runner.py. */
export class JaxBackend extends NeuralRunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/jax_runner.py")) {
    super({ id: "jax", artifactFormat: JAX_ARTIFACT_FORMAT, checkpointFormat: JAX_CHECKPOINT_FORMAT, devices: JAX_DEVICES, mixedPrecision: JAX_MIXED_PRECISION, python, script, timeoutMs });
  }
}
