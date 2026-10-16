import { registerModels, type ModelDefinition } from "./model-catalog.js";
import { mlpDefinitions } from "./neural-models.js";

/**
 * Feed-forward tabular networks written directly in JAX (autodiff, jit, hand-written optimizers). CPU only: there is no
 * CUDA jaxlib for native Windows, and the runner refuses a CUDA request instead of running it on CPU. No mixed precision.
 */
export const JAX_DEVICES = ["cpu"] as const;
export const JAX_MIXED_PRECISION = false;
export const JAX_MODELS: readonly ModelDefinition[] = mlpDefinitions("jax", JAX_DEVICES, JAX_MIXED_PRECISION);

registerModels(JAX_MODELS);
