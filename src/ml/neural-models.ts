import { defineModel, type ModelDefinition, type ParameterSpec } from "./model-catalog.js";

// Phase 3A bounds. They are part of the allowlist, so an absurd architecture is rejected before any worker starts.
export const MLP_LIMITS = { maxLayers: 4, maxWidth: 256, maxTotalUnits: 512, maxEpochs: 200, maxBatchSize: 256 } as const;

const ACTIVATIONS = ["relu", "gelu", "tanh"] as const;
const OPTIMIZERS = ["sgd", "adam", "adamw"] as const;

function mlpParameters(losses: readonly string[], defaultLoss: string): Record<string, ParameterSpec> {
  return {
    hiddenLayers: { kind: "intList", minItems: 1, maxItems: MLP_LIMITS.maxLayers, min: 1, max: MLP_LIMITS.maxWidth, default: [64, 32] },
    activation: { kind: "choice", values: ACTIVATIONS, default: "relu" },
    dropout: { kind: "float", min: 0, max: 0.8, default: 0 },
    epochs: { kind: "int", min: 1, max: MLP_LIMITS.maxEpochs, default: 20 },
    batchSize: { kind: "int", min: 1, max: MLP_LIMITS.maxBatchSize, default: 32 },
    learningRate: { kind: "float", min: 0, max: 1, openMin: true, default: 0.001 },
    optimizer: { kind: "choice", values: OPTIMIZERS, default: "adam" },
    loss: { kind: "choice", values: losses, default: defaultLoss },
  };
}

// Cross-parameter rule: total hidden width bounds the parameter count and the training cost of one epoch.
const architectureLimit = (params: Record<string, unknown>): string | null => {
  const layers = params.hiddenLayers as number[] | undefined;
  if (layers && layers.reduce((sum, width) => sum + width, 0) > MLP_LIMITS.maxTotalUnits) return `hidden layers exceed ${MLP_LIMITS.maxTotalUnits} total units`;
  return null;
};

/**
 * Feed-forward tabular networks for one neural backend. Each backend registers its own copy under its own id, with the
 * devices its runner really supports, so a capability can never be claimed that the runner does not provide.
 */
export function mlpDefinitions(backend: string, devices: readonly string[], mixedPrecision: boolean): ModelDefinition[] {
  const capabilities = (scores: "probability" | "none") => ({ scores, scaleSensitive: true, seeded: true, iterative: true, devices, mixedPrecision } as const);
  return [
    defineModel({ backend, id: "mlp_classifier", taskType: "classification", displayName: "Multilayer perceptron classifier", parameters: mlpParameters(["cross_entropy"], "cross_entropy"), capabilities: capabilities("probability"), check: architectureLimit }),
    defineModel({ backend, id: "mlp_regressor", taskType: "regression", displayName: "Multilayer perceptron regressor", parameters: mlpParameters(["mse", "l1"], "mse"), capabilities: capabilities("none"), check: architectureLimit }),
  ];
}
