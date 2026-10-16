import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import type { Snapshot } from "../../src/ml/domain.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { JaxBackend } from "../../src/ml/jax-backend.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { snapshotFor } from "./ml-fixtures.js";

const columns = [{ name: "a", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "label", type: "string" as const }];
const jaxSnapshot = (opts: Parameters<typeof snapshotFor>[0] = {}): Snapshot => snapshotFor({ backend: "jax", algorithm: "mlp_classifier", columns, target: "label", features: ["a", "kind"], steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }], ...opts });
const withTraining = (snapshot: Snapshot, training: Record<string, unknown>): Snapshot => ({ ...snapshot, training } as Snapshot);

describe("JAX model catalog (Phase 4 M2)", () => {
  it("registers two CPU-only neural definitions under the jax backend, separate from the PyTorch ones with the same ids", () => {
    const models = listModels("jax");
    expect(models.map(m => m.id).sort()).toEqual(["mlp_classifier", "mlp_regressor"]);
    for (const m of models) {
      expect(m.capabilities).toMatchObject({ iterative: true, devices: ["cpu"], mixedPrecision: false, seeded: true });
      expect(() => validateHyperparameters(m, m.defaults), m.id).not.toThrow();
    }
    expect(getModel("jax", "mlp_classifier").capabilities.scores).toBe("probability");
    expect(getModel("pytorch", "mlp_classifier").capabilities.devices).toEqual(["cpu", "cuda"]);
  });
  it("rejects architectures and options outside the shared allowlist", () => {
    const model = getModel("jax", "mlp_classifier");
    for (const hp of [{ hiddenLayers: [512, 512] }, { activation: "swish" }, { optimizer: "lbfgs" }, { module: "os" }]) expect(() => validateHyperparameters(model, hp)).toThrow();
  });
});

describe("JAX backend validation", () => {
  const backend = new JaxBackend("unused");
  it("declares exactly what the runner does: iterative, cpu, fp32, tabular classification and regression", () => {
    expect(backend.capabilities).toMatchObject({ artifactFormats: ["jax-params-v1"], checkpointFormats: ["jax-checkpoint-v1"], devices: ["cpu"], precisions: ["fp32"], iterative: true, tasks: ["classification", "regression"] });
    expect(() => backend.validate(jaxSnapshot())).not.toThrow();
  });
  it("refuses CUDA and mixed precision before any runner starts", () => {
    expect(() => backend.validate(withTraining(jaxSnapshot(), { checkpointEveryEpochs: 2, earlyStopping: null, device: "cuda", precision: "fp32" }))).toThrow(ConflictError);
    expect(() => backend.validate(withTraining(jaxSnapshot(), { checkpointEveryEpochs: 2, earlyStopping: null, device: "cpu", precision: "mixed" }))).toThrow(ConflictError);
  });
  it("refuses preprocessing the NumPy path does not implement, and another backend's snapshot", () => {
    expect(() => backend.validate(jaxSnapshot({ steps: [{ operation: "impute_median" }, { operation: "one_hot_encode" }, { operation: "variance_threshold" }] }))).toThrow();
    expect(() => backend.validate(jaxSnapshot({ backend: "pytorch" }))).toThrow();
  });
});

describe("ML configuration for JAX", () => {
  it("keeps JAX disabled by default and enables it only with the ML master switch and its own flag", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("jax")).toBe(false);
    expect(mlConfiguration({ ML_JAX_ENABLED: "true" }).backends.has("jax")).toBe(false);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_JAX_ENABLED: "true" }).backends.keys()].sort()).toEqual(["jax", "sklearn"]);
  });
});
