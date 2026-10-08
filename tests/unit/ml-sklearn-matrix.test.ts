import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { listModels } from "../../src/ml/model-catalog.js";
import { snapshotFor } from "./ml-fixtures.js";

const python = process.env.ML_TEST_PYTHON;
describe("sklearn adapter catalog and cross-validation contract", () => {
  it("validates cross-validation and metric choices through the backend contract", () => {
    const backend = new SklearnBackend("unused");
    const cv = { cv: { strategy: "kfold", folds: 5 }, metrics: ["rmse", "r2"] };
    expect(() => backend.validate(snapshotFor({ evaluation: cv }))).not.toThrow();
    expect(() => backend.validate(snapshotFor({ evaluation: { cv: { strategy: "stratified_kfold", folds: 5 } } }))).toThrow(/stratify/);
    expect(() => backend.validate(snapshotFor({ evaluation: { metrics: ["accuracy"] } }))).toThrow(/does not apply/);
    expect(() => backend.validate(snapshotFor({ evaluation: { cv: { strategy: "kfold", folds: 1 } } }))).toThrow();
    expect(() => backend.validate(snapshotFor({ algorithm: "logistic_regression", evaluation: { cv: { strategy: "stratified_kfold", folds: 3 }, metrics: ["f1", "log_loss"] }, stratify: true }))).not.toThrow();
  });
  it.runIf(Boolean(python))("trains, scores and reproduces every catalog model under the Python adapter", () => {
    const metadata = Object.fromEntries(listModels("sklearn").map(m => [m.id, { taskType: m.taskType, scores: m.capabilities.scores }]));
    const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_models.py"), JSON.stringify(metadata)], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" } }).trim().split("\n").pop()!);
    expect(out.failures).toEqual([]);
    expect(out.catalog).toEqual(listModels("sklearn").map(m => m.id).sort());
    expect(Object.keys(out.summary)).toHaveLength(15);
    for (const m of listModels("sklearn")) for (const [name, value] of Object.entries(m.defaults)) expect(out.defaults[m.id][name], `${m.id}.${name}`).toEqual(value);
  }, 240_000);
});
