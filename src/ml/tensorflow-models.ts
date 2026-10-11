import { registerModels, type ModelDefinition } from "./model-catalog.js";
import { mlpDefinitions } from "./neural-models.js";

/**
 * Feed-forward tabular networks implemented with TensorFlow (Keras). CPU only: native Windows TensorFlow has no CUDA build,
 * and the runner never redirects a CUDA request to CPU. Mixed precision is not offered.
 */
export const TENSORFLOW_DEVICES = ["cpu"] as const;
export const TENSORFLOW_MIXED_PRECISION = false;
export const TENSORFLOW_MODELS: readonly ModelDefinition[] = mlpDefinitions("tensorflow", TENSORFLOW_DEVICES, TENSORFLOW_MIXED_PRECISION);

registerModels(TENSORFLOW_MODELS);
