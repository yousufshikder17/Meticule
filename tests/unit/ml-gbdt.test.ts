import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import type { Snapshot } from "../../src/ml/domain.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { LightgbmBackend, XgboostBackend } from "../../src/ml/gbdt-backend.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { snapshotFor } from "./ml-fixtures.js";

const columns = [{ name: "a", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "label", type: "string" as const }];
const gbdtSnapshot = (backend: string, opts: Parameters<typeof snapshotFor>[0] = {}): Snapshot =>
  snapshotFor({ backend, algorithm: `${backend}_classifier`, columns, target: "label", features: ["a", "kind"], steps: [{ operation: "one_hot_encode" }], ...opts });

describe("gradient-boosting catalog (Phase 4 M3)", () => {
  it("registers native classifier and regressor definitions under each backend: seeded, scale-insensitive, one-shot, CPU", () => {
    for (const backend of ["xgboost", "lightgbm"]) {
      // LightGBM also registers its forecaster (Phase 4 M7), which reads a timestamp and is tested in ml-forecasting.test.ts.
      const models = listModels(backend).filter(m => m.taskType !== "forecasting");
      expect(listModels(backend).map(m => m.id).sort()).toEqual(backend === "lightgbm" ? ["lightgbm_classifier", "lightgbm_forecaster", "lightgbm_regressor"] : [`${backend}_classifier`, `${backend}_regressor`]);
      for (const m of models) {
        expect(m.capabilities).toMatchObject({ seeded: true, scaleSensitive: false, iterative: false, devices: ["cpu"] });
        expect(() => validateHyperparameters(m, m.defaults), m.id).not.toThrow();
      }
      expect(getModel(backend, `${backend}_classifier`).capabilities.scores).toBe("probability");
      expect(getModel(backend, `${backend}_regressor`).capabilities.scores).toBe("none");
    }
  });
  it("bounds every hyperparameter and refuses names the runner would not understand", () => {
    const xgb = getModel("xgboost", "xgboost_classifier"), lgb = getModel("lightgbm", "lightgbm_classifier");
    for (const hp of [{ n_estimators: 0 }, { n_estimators: 5000 }, { learning_rate: 0 }, { max_depth: 17 }, { early_stopping_rounds: 201 }, { booster: "dart" }, { num_leaves: 31 }]) expect(() => validateHyperparameters(xgb, hp)).toThrow();
    for (const hp of [{ num_leaves: 1 }, { max_depth: 33 }, { min_child_samples: 0 }, { min_child_weight: 1 }]) expect(() => validateHyperparameters(lgb, hp)).toThrow();
    expect(() => validateHyperparameters(lgb, { max_depth: null, num_leaves: 63, early_stopping_rounds: 10 })).not.toThrow();
  });
});

describe("gradient-boosting backend validation", () => {
  for (const backend of [new XgboostBackend("unused"), new LightgbmBackend("unused")]) {
    it(`${backend.id}: declares exactly what its runner does`, () => {
      expect(backend.capabilities).toMatchObject({ devices: ["cpu"], precisions: ["fp32"], iterative: false, checkpointFormats: [], tasks: backend.id === "lightgbm" ? ["classification", "regression", "forecasting"] : ["classification", "regression"] });
      expect(backend.capabilities.artifactFormats).toEqual([backend.id === "xgboost" ? "xgboost-json-v1" : "lightgbm-text-v1"]);
      expect(() => backend.validate(gbdtSnapshot(backend.id))).not.toThrow();
    });
    it(`${backend.id}: refuses a Meticule training policy, CUDA, undeclared preprocessing, and early stopping inside cross-validation`, () => {
      const snapshot = gbdtSnapshot(backend.id);
      expect(() => backend.validate({ ...snapshot, training: { checkpointEveryEpochs: 2, earlyStopping: null, device: "cpu", precision: "fp32" } } as Snapshot)).toThrow(/iterative/);
      expect(() => backend.validate({ ...snapshot, training: { checkpointEveryEpochs: 2, earlyStopping: null, device: "cuda", precision: "fp32" } } as Snapshot)).toThrow(ConflictError);
      expect(() => backend.validate(gbdtSnapshot(backend.id, { steps: [{ operation: "one_hot_encode" }, { operation: "variance_threshold" }] }))).toThrow(/does not support preprocessing/);
      const cv = { cv: { strategy: "kfold" as const, folds: 3 } };
      expect(() => backend.validate(gbdtSnapshot(backend.id, { evaluation: cv }))).not.toThrow();
      expect(() => backend.validate(gbdtSnapshot(backend.id, { evaluation: cv, hyperparameters: { early_stopping_rounds: 5 } }))).toThrow(/Early stopping is not supported with cross-validation/);
    });
  }
});

describe("ML configuration for the gradient-boosting backends", () => {
  it("keeps both disabled by default and enables each only with the ML master switch and its own flag", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("xgboost")).toBe(false);
    expect(mlConfiguration({ ML_XGBOOST_ENABLED: "true", ML_LIGHTGBM_ENABLED: "true" }).backends.size).toBe(0);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_XGBOOST_ENABLED: "true" }).backends.keys()].sort()).toEqual(["sklearn", "xgboost"]);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_LIGHTGBM_ENABLED: "true" }).backends.keys()].sort()).toEqual(["lightgbm", "sklearn"]);
  });
});
