import { describe, expect, it } from "vitest";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { MLP_LIMITS } from "../../src/ml/pytorch-models.js";
import { generateCandidates } from "../../src/ml/search.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { snapshotFor } from "./ml-fixtures.js";

const mlp = (id = "mlp_classifier") => getModel("pytorch", id);
const reject = (fn: () => unknown) => expect(fn).toThrow();
const columns = [{ name: "a", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "label", type: "string" as const }];
const pytorchSnapshot = (opts: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({ backend: "pytorch", algorithm: "mlp_classifier", columns, target: "label", features: ["a", "kind"], steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }], ...opts });

describe("PyTorch model catalog (Phase 3A)", () => {
  it("registers two CPU-only neural definitions under the pytorch backend, with capability metadata", () => {
    const models = listModels("pytorch");
    expect(models.map(m => m.id).sort()).toEqual(["mlp_classifier", "mlp_regressor"]);
    for (const m of models) {
      expect(m.capabilities).toMatchObject({ iterative: true, devices: ["cpu", "cuda"], mixedPrecision: true, seeded: true });
      expect(() => validateHyperparameters(m, m.defaults), m.id).not.toThrow();
      expect(m.defaults.hiddenLayers).toEqual([64, 32]);
    }
    expect(mlp().capabilities.scores).toBe("probability"); expect(mlp("mlp_regressor").capabilities.scores).toBe("none");
    expect(listModels("sklearn").filter(m => m.taskType === "classification" || m.taskType === "regression")).toHaveLength(15);
  });
  it("accepts a bounded architecture and an explicit training configuration", () => {
    expect(validateHyperparameters(mlp(), { hiddenLayers: [128, 64, 32], activation: "gelu", dropout: 0.5, epochs: 50, batchSize: 64, learningRate: 0.0005, optimizer: "adamw", loss: "cross_entropy" })).toMatchObject({ hiddenLayers: [128, 64, 32] });
    expect(validateHyperparameters(mlp("mlp_regressor"), { loss: "l1", optimizer: "sgd" })).toMatchObject({ loss: "l1" });
  });
  it("rejects malformed and absurd architectures and training options before any worker starts", () => {
    const bad: Record<string, unknown>[] = [
      { hiddenLayers: [] }, { hiddenLayers: [8, 8, 8, 8, 8] }, { hiddenLayers: [MLP_LIMITS.maxWidth + 1] }, { hiddenLayers: [0] }, { hiddenLayers: [1.5] },
      { hiddenLayers: [256, 256, 256] }, { hiddenLayers: "64,32" }, { dropout: 0.9 }, { dropout: -0.1 }, { activation: "swish" },
      { epochs: 0 }, { epochs: MLP_LIMITS.maxEpochs + 1 }, { batchSize: 0 }, { batchSize: MLP_LIMITS.maxBatchSize + 1 },
      { learningRate: 0 }, { learningRate: 2 }, { optimizer: "lbfgs" }, { loss: "mse" }, { n_jobs: 4 }, { module: "torch.nn" }, { seed: 1 },
    ];
    for (const hp of bad) reject(() => validateHyperparameters(mlp(), hp));
    reject(() => validateHyperparameters(mlp("mlp_regressor"), { loss: "cross_entropy" }));
  });
  it("cannot be searched over the architecture: list parameters are not searchable", () => {
    reject(() => generateCandidates(mlp(), { hiddenLayers: { type: "int", min: 1, max: 8 } }, {}, "random", 4, 1));
    const points = generateCandidates(mlp(), { learningRate: { type: "float", min: 0.001, max: 0.1, log: true }, batchSize: { type: "choice", values: [16, 32] }, dropout: { type: "float", min: 0, max: 0.5 }, optimizer: { type: "choice", values: ["adam", "sgd"] } }, {}, "random", 5, 3);
    expect(points).toHaveLength(5);
    for (const p of points) expect(() => validateHyperparameters(mlp(), p)).not.toThrow();
  });
});

describe("PyTorch backend validation", () => {
  const backend = new PytorchBackend("unused");
  it("accepts a valid classification snapshot and refuses backend mismatches", () => {
    expect(() => backend.validate(pytorchSnapshot())).not.toThrow();
    reject(() => backend.validate(pytorchSnapshot({ backend: "sklearn" })));
  });
  it("refuses preprocessing the NumPy path does not implement, instead of silently changing semantics", () => {
    reject(() => backend.validate(pytorchSnapshot({ steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }, { operation: "variance_threshold" }] })));
  });
  it("refuses stratified regression and metrics the model cannot produce", () => {
    reject(() => backend.validate(pytorchSnapshot({ algorithm: "mlp_regressor", columns: [{ name: "a", type: "number" }, { name: "y", type: "number" }], target: "y", features: ["a"], steps: [], stratify: true })));
    reject(() => backend.validate(pytorchSnapshot({ algorithm: "mlp_regressor", columns: [{ name: "a", type: "number" }, { name: "y", type: "number" }], target: "y", features: ["a"], steps: [], evaluation: { metrics: ["log_loss"] } })));
  });
  it("refuses unknown neural algorithms and unsafe hyperparameters at validation time", () => {
    reject(() => backend.validate(pytorchSnapshot({ algorithm: "arbitrary_module" })));
    reject(() => backend.validate(pytorchSnapshot({ hyperparameters: { hiddenLayers: [512, 512] } })));
  });
});

describe("ML configuration for optional backends", () => {
  it("keeps PyTorch disabled by default and enables it only with the ML master switch and its own flag", () => {
    expect(mlConfiguration({}).backends.has("pytorch")).toBe(false);
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("pytorch")).toBe(false);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_PYTORCH_ENABLED: "true" }).backends.keys()].sort()).toEqual(["pytorch", "sklearn"]);
    expect(mlConfiguration({ ML_PYTORCH_ENABLED: "true" }).backends.has("pytorch")).toBe(false);
  });
});
