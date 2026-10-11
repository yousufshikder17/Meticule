import { resolve } from "node:path";
import type { BackendCapabilities } from "./backend.js";
import { RunnerBackend } from "./runner-backend.js";

// Every preprocessing operation the scikit-learn runner interprets (ml/sklearn_runner.py build_pipeline).
const SKLEARN_PREPROCESSING = ["impute_median", "impute_mean", "standard_scale", "min_max_scale", "impute_most_frequent", "one_hot_encode", "ordinal_encode", "variance_threshold"];

/**
 * One fit per attempt, on CPU, in fp32, with no checkpoints. Nothing here is iterative, so a training policy is refused
 * by the shared validation.
 */
export const SKLEARN_CAPABILITIES = {
  artifactFormats: ["sklearn-pickle-v1"],
  checkpointFormats: [],
  devices: ["cpu"],
  precisions: ["fp32"],
  preprocessing: SKLEARN_PREPROCESSING,
  iterative: false,
} as const satisfies BackendCapabilities;

export class SklearnBackend extends RunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/sklearn_runner.py")) {
    super({ id: "sklearn", capabilities: SKLEARN_CAPABILITIES, python, script, timeoutMs });
  }
}
