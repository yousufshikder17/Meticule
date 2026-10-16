import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { ForecastingBackend } from "../../src/ml/forecasting-backend.js";
import { LightgbmBackend, XgboostBackend } from "../../src/ml/gbdt-backend.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { predictionCount } from "../../src/ml/tasks.js";
import { snapshotFor } from "./ml-fixtures.js";

const python = process.env.ML_TEST_PYTHON;
const chronological = { strategy: "chronological", timeColumn: "at" };
const series = (algorithm = "ridge_forecaster", opts: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({
  backend: "forecasting", algorithm, columns: [{ name: "at", type: "timestamp" }, { name: "y", type: "number" }], target: "y", features: ["at"], split: chronological, rowCount: 160, ...opts,
});

describe("forecasting catalog and validation (Phase 4 M7)", () => {
  it("registers two forecasters, plus LightGBM's, each reading only a timestamp, with a bounded horizon", () => {
    expect(listModels("forecasting").map(m => m.id).sort()).toEqual(["ridge_forecaster", "seasonal_naive_forecaster"]);
    for (const m of [...listModels("forecasting"), getModel("lightgbm", "lightgbm_forecaster")]) {
      expect(m).toMatchObject({ taskType: "forecasting", inputs: { types: ["timestamp"], min: 1, max: 1 } });
      expect(() => validateHyperparameters(m, { horizon: 0 })).toThrow(); expect(() => validateHyperparameters(m, { horizon: 101 })).toThrow();
    }
    expect(() => validateHyperparameters(getModel("lightgbm", "lightgbm_forecaster"), { early_stopping_rounds: 5 })).toThrow();
    expect(() => validateHyperparameters(getModel("forecasting", "ridge_forecaster"), { windows: [1] })).toThrow();
    expect(new LightgbmBackend("unused").capabilities).toMatchObject({ tasks: ["classification", "regression", "forecasting"], inputs: ["number", "string", "boolean", "timestamp"] });
    expect(new XgboostBackend("unused").capabilities.tasks).toEqual(["classification", "regression"]);
  });
  it("needs a chronological split on a non-nullable time column, only that column as feature, and no CV", () => {
    const backend = new ForecastingBackend("unused");
    expect(() => backend.validate(series())).not.toThrow();
    expect(() => backend.validate(series("ridge_forecaster", { split: { strategy: "random", stratify: false } }))).toThrow(ConflictError);
    expect(() => backend.validate(series("ridge_forecaster", { evaluation: { cv: { strategy: "kfold", folds: 3 } } }))).toThrow(ConflictError);
    expect(() => backend.validate(series("ridge_forecaster", { columns: [{ name: "at", type: "timestamp", nullable: true }, { name: "y", type: "number" }] }))).toThrow(ConflictError);
    expect(() => backend.validate(series("ridge_forecaster", { columns: [{ name: "at", type: "timestamp" }, { name: "y", type: "string" }] }))).toThrow(ConflictError);
    expect(() => backend.validate(series("ridge_forecaster", { columns: [{ name: "at", type: "timestamp" }, { name: "z", type: "number" }, { name: "y", type: "number" }], features: ["at", "z"] }))).toThrow(ConflictError);
    expect(predictionCount(getModel("forecasting", "ridge_forecaster"), series("ridge_forecaster", { hyperparameters: { horizon: 12 } }), 30)).toBe(12);
  });
  it("is disabled by default and enabled only with ML_ENABLED and its own flag", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("forecasting")).toBe(false);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_FORECASTING_ENABLED: "true" }).backends.keys()].sort()).toEqual(["forecasting", "sklearn"]);
  });
});

// Every named behaviour is asserted by name: known answers, leakage proofs, refusals, outputs and artifact safety.
it.runIf(Boolean(python))("real forecasting runner: known-answer series, leakage proofs, refusals, outputs and artifacts", () => {
  const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_forecasting.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(25);
}, 300_000);
