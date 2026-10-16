import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { snapshotFor } from "./ml-fixtures.js";

const python = process.env.ML_TEST_PYTHON;
const anomaly = (algorithm: string, opts: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({
  backend: "sklearn", algorithm, columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "outlier", type: "boolean" }], target: "outlier", features: ["a", "b"], ...opts,
});

describe("anomaly detection in the scikit-learn backend (Phase 4 M7)", () => {
  it("registers three unsupervised models scored by anomaly score, with a numeric contamination default", () => {
    const models = listModels("sklearn").filter(m => m.taskType === "anomaly_detection");
    expect(models.map(m => m.id).sort()).toEqual(["isolation_forest", "local_outlier_factor", "robust_covariance"]);
    for (const m of models) { expect(m.learning).toBe("unsupervised"); expect(m.defaults.contamination).toBe(0.1); expect(m.capabilities.scores).toBe("decision"); }
    for (const hp of [{ contamination: 0 }, { contamination: 0.6 }, { contamination: "auto" }]) expect(() => validateHyperparameters(getModel("sklearn", "isolation_forest"), hp)).toThrow();
    expect(new SklearnBackend("unused").capabilities.tasks).toContain("anomaly_detection");
  });
  it("accepts unlabelled data, refuses non-boolean labels, CV and metrics that need labels on unlabelled data", () => {
    const backend = new SklearnBackend("unused");
    expect(() => backend.validate(anomaly("isolation_forest"))).not.toThrow();
    expect(() => backend.validate(anomaly("isolation_forest", { columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }], target: null }))).not.toThrow();
    expect(() => backend.validate(anomaly("isolation_forest", { columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "outlier", type: "string" }] }))).toThrow(ConflictError);
    expect(() => backend.validate(anomaly("isolation_forest", { evaluation: { cv: { strategy: "kfold", folds: 3 } } }))).toThrow(ConflictError);
    expect(() => backend.validate(anomaly("isolation_forest", { columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }], target: null, evaluation: { metrics: ["average_precision"] } }))).toThrow(/needs a target/);
    expect(() => backend.validate(anomaly("local_outlier_factor", { evaluation: { metrics: ["average_precision", "anomaly_f1", "roc_auc"] } }))).not.toThrow();
  });
});

// Every named behaviour is asserted by name, with the TypeScript defaults compared against scikit-learn's.
it.runIf(Boolean(python))("real scikit-learn anomaly detection: planted outliers, score direction, threshold, labels unused, schemas, defaults", () => {
  const defaults = Object.fromEntries(listModels("sklearn").filter(m => m.taskType === "anomaly_detection").map(m => [m.id, m.defaults]));
  const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_anomaly.py"), JSON.stringify(defaults)], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(24);
}, 300_000);
