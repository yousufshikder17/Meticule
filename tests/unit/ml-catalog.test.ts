import { describe, expect, it } from "vitest";
import { validatePipeline } from "../../src/ml/preprocessing.js";
import { describeModel, getModel, listModels, registerModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { assertMetricsSupported, METRICS, DEFAULT_METRICS } from "../../src/ml/metrics.js";
import { snapshotFor } from "./ml-fixtures.js";

describe("model catalog", () => {
  it("lists the classical catalog by task and rejects duplicate registration", () => {
    const models = listModels("sklearn");
    expect(models.filter(m => m.taskType === "classification")).toHaveLength(6);
    expect(models.filter(m => m.taskType === "regression")).toHaveLength(9);
    expect(() => registerModels([models[0]!])).toThrow(/Duplicate/);
    expect(() => getModel("sklearn", "xgboost_classifier")).toThrow(/Unsupported/);
    expect(() => getModel("xgboost", "ridge")).toThrow(/Unsupported/);
  });
  it("validates every default and rejects unknown, out-of-range and impossible parameters", () => {
    for (const m of listModels()) {
      expect(() => validateHyperparameters(m, m.defaults), m.id).not.toThrow();
      expect(() => validateHyperparameters(m, { n_jobs: 4 }), m.id).toThrow();
      expect(describeModel(m)).not.toHaveProperty("parameterSchema");
    }
    const rf = getModel("sklearn", "random_forest_classifier");
    for (const bad of [{ n_estimators: 0 }, { n_estimators: 1.5 }, { max_depth: "deep" }, { max_features: "all" }, { class_weight: "custom" }]) expect(() => validateHyperparameters(rf, bad)).toThrow();
    const svc = getModel("sklearn", "svc");
    expect(() => validateHyperparameters(svc, { kernel: "linear", degree: 4 })).toThrow(/degree/);
    expect(() => validateHyperparameters(svc, { kernel: "poly", degree: 4 })).not.toThrow();
    expect(() => validateHyperparameters(getModel("sklearn", "elastic_net"), { l1_ratio: 1.5 })).toThrow();
  });
  it("declares metric direction in one place and rejects metrics a model cannot produce", () => {
    expect(METRICS.log_loss.direction).toBe("lower"); expect(METRICS.rmse.direction).toBe("lower"); expect(METRICS.r2.direction).toBe("higher");
    const svc = getModel("sklearn", "svc"), forest = getModel("sklearn", "random_forest_classifier"), ridge = getModel("sklearn", "ridge");
    expect(() => assertMetricsSupported(svc, ["accuracy", "f1", "roc_auc"])).not.toThrow();
    expect(() => assertMetricsSupported(svc, ["log_loss"])).toThrow(/cannot produce/);
    expect(() => assertMetricsSupported(forest, ["log_loss", "roc_auc"])).not.toThrow();
    expect(() => assertMetricsSupported(ridge, ["accuracy"])).toThrow(/does not apply/);
    expect(() => assertMetricsSupported(ridge, ["nonsense"])).toThrow(/Unknown/);
    expect(DEFAULT_METRICS.regression).toEqual(["mae", "mse"]);
  });
  it("validates preprocessing as explicit, ordered, typed operations", () => {
    const mixed = [{ name: "n", type: "number" as const, nullable: true }, { name: "c", type: "string" as const, nullable: true }, { name: "b", type: "boolean" as const }, { name: "y", type: "number" as const }];
    const build = (steps: { operation: string; parameters?: Record<string, unknown> }[], features = ["n", "c", "b"]) => snapshotFor({ columns: mixed, features, steps });
    const ok = [{ operation: "impute_median" }, { operation: "min_max_scale" }, { operation: "impute_most_frequent" }, { operation: "one_hot_encode" }, { operation: "variance_threshold", parameters: { threshold: 0.01 } }];
    expect(() => validatePipeline(build(ok))).not.toThrow();
    expect(() => validatePipeline(build([{ operation: "ordinal_encode", parameters: { categories: { c: ["a", "b"], b: ["false", "true"] } } }]))).not.toThrow();
    for (const bad of [[], ok.slice(0, 3), [{ operation: "one_hot_encode" }, { operation: "impute_most_frequent" }], [...ok, { operation: "one_hot_encode" }], [{ operation: "one_hot_encode" }, { operation: "ordinal_encode", parameters: { categories: {} } }],
      [{ operation: "one_hot_encode" }, { operation: "impute_mean" }, { operation: "impute_median" }], [{ operation: "one_hot_encode" }, { operation: "standard_scale" }, { operation: "min_max_scale" }], [{ operation: "one_hot_encode" }, { operation: "arbitrary_python" }],
      [{ operation: "one_hot_encode" }, { operation: "standard_scale", parameters: { with_mean: false } }], [{ operation: "ordinal_encode", parameters: { categories: { c: ["a"] } } }], [{ operation: "one_hot_encode" }, { operation: "variance_threshold", parameters: { threshold: -1 } }]]) expect(() => validatePipeline(build(bad))).toThrow();
    expect(() => validatePipeline(build([{ operation: "one_hot_encode" }], ["n"]))).toThrow(/require categorical/);
    expect(() => validatePipeline(snapshotFor({ steps: [{ operation: "impute_median" }, { operation: "standard_scale" }] }))).not.toThrow();
  });
});

describe("backend extensibility", () => {
  it("accepts another backend's definitions without touching sklearn behaviour", () => {
    // XGBoost/LightGBM are deferred (see docs/ml.md); this proves the seam they would use.
    const [template] = listModels("sklearn");
    registerModels([{ ...template!, id: "xgboost_classifier", backend: "xgboost" }]);
    expect(listModels("xgboost").map(m => m.id)).toEqual(["xgboost_classifier"]);
    expect(getModel("xgboost", "xgboost_classifier").backend).toBe("xgboost");
    expect(() => getModel("sklearn", "xgboost_classifier")).toThrow(/Unsupported/);
    expect(listModels("sklearn")).toHaveLength(15);
  });
});
