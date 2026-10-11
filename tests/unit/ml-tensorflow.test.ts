import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import type { Snapshot } from "../../src/ml/domain.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";
import { snapshotFor } from "./ml-fixtures.js";

const reject = (fn: () => unknown) => expect(fn).toThrow();
const columns = [{ name: "a", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "label", type: "string" as const }];
const tensorflowSnapshot = (opts: Parameters<typeof snapshotFor>[0] = {}): Snapshot => snapshotFor({ backend: "tensorflow", algorithm: "mlp_classifier", columns, target: "label", features: ["a", "kind"], steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }], ...opts });
const withTraining = (snapshot: Snapshot, training: Record<string, unknown>): Snapshot => ({ ...snapshot, training } as Snapshot);

describe("TensorFlow model catalog (Phase 3D)", () => {
  it("registers two CPU-only neural definitions under the tensorflow backend", () => {
    const models = listModels("tensorflow");
    expect(models.map(m => m.id).sort()).toEqual(["mlp_classifier", "mlp_regressor"]);
    for (const m of models) {
      expect(m.capabilities).toMatchObject({ iterative: true, devices: ["cpu"], mixedPrecision: false, seeded: true });
      expect(() => validateHyperparameters(m, m.defaults), m.id).not.toThrow();
    }
    expect(getModel("tensorflow", "mlp_classifier").capabilities.scores).toBe("probability");
    expect(getModel("tensorflow", "mlp_regressor").capabilities.scores).toBe("none");
  });
  it("keeps the PyTorch catalog unchanged: CUDA stays a PyTorch capability only", () => {
    expect(getModel("pytorch", "mlp_classifier").capabilities.devices).toEqual(["cpu", "cuda"]);
  });
  it("rejects architectures and training options outside the shared allowlist", () => {
    const model = getModel("tensorflow", "mlp_classifier");
    for (const hp of [{ hiddenLayers: [512, 512] }, { activation: "swish" }, { optimizer: "lbfgs" }, { module: "os" }, { n_jobs: 4 }]) reject(() => validateHyperparameters(model, hp));
  });
});

describe("TensorFlow backend validation", () => {
  const backend = new TensorflowBackend("unused");
  it("accepts a valid classification snapshot and refuses backend mismatches", () => {
    expect(() => backend.validate(tensorflowSnapshot())).not.toThrow();
    reject(() => backend.validate(tensorflowSnapshot({ backend: "pytorch" })));
  });
  it("refuses CUDA explicitly, because native Windows TensorFlow has no CUDA path", () => {
    expect(() => backend.validate(withTraining(tensorflowSnapshot(), { checkpointEveryEpochs: 2, earlyStopping: null, device: "cuda", precision: "fp32" }))).toThrow(ConflictError);
    expect(() => backend.validate(withTraining(tensorflowSnapshot(), { checkpointEveryEpochs: 2, earlyStopping: null, device: "cuda", precision: "fp32" }))).toThrow(/does not support device cuda/);
  });
  it("refuses mixed precision, which the TensorFlow runner does not offer", () => {
    reject(() => backend.validate(withTraining(tensorflowSnapshot(), { checkpointEveryEpochs: 2, earlyStopping: null, device: "cuda", precision: "mixed" })));
  });
  it("refuses preprocessing the NumPy path does not implement, instead of silently changing semantics", () => {
    reject(() => backend.validate(tensorflowSnapshot({ steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }, { operation: "variance_threshold" }] })));
  });
  it("refuses stratified regression and metrics the model cannot produce", () => {
    reject(() => backend.validate(tensorflowSnapshot({ algorithm: "mlp_regressor", columns: [{ name: "a", type: "number" }, { name: "y", type: "number" }], target: "y", features: ["a"], steps: [], stratify: true })));
    reject(() => backend.validate(tensorflowSnapshot({ algorithm: "mlp_regressor", columns: [{ name: "a", type: "number" }, { name: "y", type: "number" }], target: "y", features: ["a"], steps: [], evaluation: { metrics: ["log_loss"] } })));
  });
  it("reports the artifact format it writes", () => {
    expect(backend.capabilities.artifactFormats).toEqual(["tensorflow-variables-v1"]);
    expect(backend.id).toBe("tensorflow");
  });
});

describe("ML configuration for TensorFlow", () => {
  it("keeps TensorFlow disabled by default and enables it only with the ML master switch and its own flag", () => {
    expect(mlConfiguration({}).backends.has("tensorflow")).toBe(false);
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("tensorflow")).toBe(false);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_TENSORFLOW_ENABLED: "true" }).backends.keys()].sort()).toEqual(["sklearn", "tensorflow"]);
    expect(mlConfiguration({ ML_TENSORFLOW_ENABLED: "true" }).backends.has("tensorflow")).toBe(false);
  });
  it("can run alongside PyTorch, each with its own interpreter", () => {
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_PYTORCH_ENABLED: "true", ML_TENSORFLOW_ENABLED: "true" }).backends.keys()].sort()).toEqual(["pytorch", "sklearn", "tensorflow"]);
  });
});
