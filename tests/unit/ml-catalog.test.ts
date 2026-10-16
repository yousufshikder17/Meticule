import { describe, expect, it } from "vitest";
import { validatePipeline } from "../../src/ml/preprocessing.js";
import { defineModel, describeModel, getModel, listModels, registerModels, validateHyperparameters, type ModelCapabilities } from "../../src/ml/model-catalog.js";
import { assertMetricsSupported, assertRankingMetrics, defaultMetrics, METRICS, DEFAULT_METRICS, type MetricName } from "../../src/ml/metrics.js";
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { SKLEARN_CAPABILITIES } from "../../src/ml/sklearn-backend.js";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";
import { snapshotFor } from "./ml-fixtures.js";

const scored: ModelCapabilities = { scores: "decision", scaleSensitive: false, seeded: true };

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
  it("treats only string and boolean features as categorical, never text", () => {
    const textual = [{ name: "review", type: "text" as const }, { name: "n", type: "number" as const }, { name: "y", type: "number" as const }];
    expect(() => validatePipeline(snapshotFor({ columns: textual, features: ["review", "n"], steps: [{ operation: "impute_median" }] }))).not.toThrow();
    expect(() => validatePipeline(snapshotFor({ columns: textual, features: ["review", "n"], steps: [{ operation: "one_hot_encode" }] }))).toThrow(/require categorical/);
  });
  it("declares every model's inputs and learning, defaulting to 1-199 tabular columns and supervised", () => {
    for (const m of listModels()) {
      expect(m.inputs, m.id).toEqual({ types: ["number", "string", "boolean"], min: 1, max: 199 });
      expect(m.learning, m.id).toBe("supervised");
      expect(describeModel(m)).toMatchObject({ inputs: m.inputs, learning: "supervised" });
    }
    const embedder = defineModel({ backend: "test", id: "text_embedder", taskType: "embedding", displayName: "Text embedder", parameters: { model: { kind: "name", default: null } }, capabilities: scored, inputs: { types: ["text"], min: 1, max: 1 }, learning: "pretrained" });
    expect(describeModel(embedder)).toMatchObject({ taskType: "embedding", inputs: { types: ["text"], min: 1, max: 1 }, learning: "pretrained", defaults: { model: null } });
    for (const ok of [{}, { model: null }, { model: "bert-base-uncased" }, { model: "a" }, { model: "m1.v2_x-3" }, { model: "x".repeat(64) }]) expect(() => validateHyperparameters(embedder, ok), JSON.stringify(ok)).not.toThrow();
    for (const bad of [{ model: "" }, { model: "Bert" }, { model: "../x" }, { model: "a/b" }, { model: "a\b" }, { model: ".hidden" }, { model: "-x" }, { model: "a b" }, { model: "x".repeat(65) }, { model: 1 }])
      expect(() => validateHyperparameters(embedder, bad), JSON.stringify(bad)).toThrow();
  });
  it("declares truthful backend tasks and inputs: every registered model's task and input types fit its backend", () => {
    const backends = { sklearn: SKLEARN_CAPABILITIES, pytorch: new PytorchBackend("python").capabilities, tensorflow: new TensorflowBackend("python").capabilities };
    for (const [backend, capabilities] of Object.entries(backends)) {
      expect(capabilities.tasks, backend).toEqual(["classification", "regression"]);
      expect(capabilities.inputs, backend).toEqual(["number", "string", "boolean"]);
      const models = listModels(backend);
      expect(models.length, backend).toBeGreaterThan(0);
      for (const m of models) {
        expect(capabilities.tasks, m.id).toContain(m.taskType);
        for (const type of m.inputs.types) expect(capabilities.inputs, m.id).toContain(type);
      }
    }
  });
  it("registers every M1 metric with its tasks, label need and diagnostics, and defaults per task and labelling", () => {
    expect(Object.keys(METRICS).sort()).toEqual(["accuracy", "anomaly_f1", "anomaly_precision", "anomaly_recall", "average_precision", "f1", "log_loss", "mae", "mase", "mean_pairwise_cosine", "mse", "precision", "r2",
      "recall", "retrieval_mrr", "retrieval_precision_at_1", "rmse", "roc_auc", "smape", "threshold_calibration_error"]);
    expect(METRICS.roc_auc).toEqual({ tasks: ["classification", "anomaly_detection"], direction: "higher", needs: "scores", labels: true });
    expect(METRICS.mae.tasks).toEqual(["regression", "forecasting"]);
    expect((Object.keys(METRICS) as MetricName[]).filter(n => !METRICS[n].labels)).toEqual(["threshold_calibration_error", "mean_pairwise_cosine"]);
    expect((Object.keys(METRICS) as MetricName[]).filter(n => METRICS[n].diagnostic)).toEqual(["threshold_calibration_error", "mean_pairwise_cosine"]);
    expect(DEFAULT_METRICS).toEqual({ classification: ["accuracy"], regression: ["mae", "mse"] });
    expect(defaultMetrics("forecasting", true)).toEqual(["mae", "rmse"]);
    expect([defaultMetrics("anomaly_detection", true), defaultMetrics("anomaly_detection", false)]).toEqual([["average_precision"], ["threshold_calibration_error"]]);
    expect([defaultMetrics("embedding", true), defaultMetrics("embedding", false)]).toEqual([["retrieval_precision_at_1"], ["mean_pairwise_cosine"]]);
    for (const task of ["classification", "regression", "forecasting"] as const) expect(() => defaultMetrics(task, false)).toThrow(/requires a target/);
  });
  it("refuses metrics that need labels an unlabelled dataset lacks, and diagnostics as ranking metrics", () => {
    const detector = defineModel({ backend: "test", id: "detector", taskType: "anomaly_detection", displayName: "Detector", parameters: {}, capabilities: scored, learning: "unsupervised" });
    expect(() => assertMetricsSupported(detector, ["roc_auc", "average_precision", "anomaly_precision", "anomaly_recall", "anomaly_f1", "threshold_calibration_error"])).not.toThrow();
    expect(() => assertMetricsSupported(detector, ["threshold_calibration_error"], false)).not.toThrow();
    expect(() => assertMetricsSupported(detector, ["average_precision"], false)).toThrow(/needs a target/);
    expect(() => assertMetricsSupported(detector, ["accuracy"])).toThrow(/does not apply/);
    expect(() => assertMetricsSupported({ ...detector, capabilities: { ...scored, scores: "none" } }, ["average_precision"])).toThrow(/cannot produce/);
    expect(() => assertMetricsSupported(getModel("sklearn", "ridge"), ["smape"])).toThrow(/does not apply/);
    expect(() => assertMetricsSupported(getModel("sklearn", "ridge"), ["mae"], false)).toThrow(/needs a target/);
    expect(() => assertRankingMetrics(["accuracy", "average_precision", "nonsense"])).not.toThrow();
    for (const diagnostic of ["threshold_calibration_error", "mean_pairwise_cosine"]) expect(() => assertRankingMetrics(["f1", diagnostic])).toThrow(/diagnostic/);
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
