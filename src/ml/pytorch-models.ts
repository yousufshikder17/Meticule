import { registerModels, type ModelDefinition } from "./model-catalog.js";
import { mlpDefinitions } from "./neural-models.js";

export { MLP_LIMITS } from "./neural-models.js";

/** Devices and precision the PyTorch runner executes. The backend declares them, and its models may only use these. */
export const PYTORCH_DEVICES = ["cpu", "cuda"] as const;
export const PYTORCH_MIXED_PRECISION = true;

/** Feed-forward tabular networks implemented with PyTorch. Registered under the `pytorch` backend, never special-cased. */
export const PYTORCH_MODELS: readonly ModelDefinition[] = mlpDefinitions("pytorch", PYTORCH_DEVICES, PYTORCH_MIXED_PRECISION);

registerModels(PYTORCH_MODELS);
