import { resolve } from "node:path";
import type { MlBackend } from "../../src/ml/backend.js";
import { ForecastingBackend } from "../../src/ml/forecasting-backend.js";
import { LightgbmBackend } from "../../src/ml/gbdt-backend.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { ANOMALY_ROWS, SERIES_ROWS } from "./modality-rows.js";
import type { TaskTarget } from "./task-suite.js";

// Forecasting runs on the scikit-learn interpreter (the forecasting backend) or the gradient-boosting one (LightGBM).
const sklearnPython = process.env.ML_TEST_PYTHON ?? "", gbdtPython = process.env.ML_TEST_LIGHTGBM_PYTHON ?? "";
const CHRONOLOGICAL = { strategy: "chronological", timeColumn: "at", train: 0.6, validation: 0.2, test: 0.2 };

export function forecastTarget(backend: "forecasting" | "lightgbm", algorithm: string, hyperparameters: Record<string, unknown>): TaskTarget {
  const python = backend === "forecasting" ? sklearnPython : gbdtPython;
  const create = (timeoutMs?: number): MlBackend => (backend === "forecasting" ? new ForecastingBackend(python, timeoutMs) : new LightgbmBackend(python, timeoutMs));
  return {
    id: backend, create, python, script: resolve(`ml/${backend === "forecasting" ? "forecasting" : "lightgbm"}_runner.py`), algorithm, hyperparameters,
    columns: [{ name: "at", type: "timestamp" }, { name: "y", type: "number" }], feature: "at", target: "y", rows: SERIES_ROWS, metrics: ["mae", "rmse", "smape", "mase"],
    split: CHRONOLOGICAL, predictRows: SERIES_ROWS.slice(-30), requiredInCi: backend === "forecasting" ? process.env.CI === "true" : process.env.ML_GBDT_JOB === "true",
  };
}

/** One anomaly target over two numeric features; `labelled` adds the boolean label for the label-based metrics. */
export function anomalyTarget(algorithm: string, labelled: boolean): TaskTarget {
  return {
    id: "sklearn", create: timeoutMs => new SklearnBackend(sklearnPython, timeoutMs), python: sklearnPython, script: resolve("ml/sklearn_runner.py"), algorithm, hyperparameters: { contamination: 0.1 },
    columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }, ...(labelled ? [{ name: "outlier", type: "boolean" as const }] : [])],
    feature: "a", target: labelled ? "outlier" : null, rows: labelled ? ANOMALY_ROWS : ANOMALY_ROWS.map(r => ({ a: r.a ?? null, b: r.b ?? null })),
    metrics: labelled ? ["average_precision", "anomaly_f1"] : ["threshold_calibration_error"], requiredInCi: process.env.CI === "true",
    // Label-based metrics are undefined on a partition without an anomaly, so labelled data is split stratified.
    split: { strategy: "random", train: 0.6, validation: 0.2, test: 0.2, stratify: labelled },
  };
}
