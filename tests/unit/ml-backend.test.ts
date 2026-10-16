import { randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import { expect, it, vi } from "vitest";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TrainingSnapshot } from "../../src/ml/domain.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { contractFor, ML_BACKEND_CONTRACT_V2 } from "../../src/ml/contract.js";
import { snapshotFor } from "./ml-fixtures.js";
import { CHRONO_ROWS, chronologicalSnapshot, expectTimeOrdered } from "../conformance/contract-suite.js";

export function numericSnapshot(algorithm = "ridge") {
  const id = randomUUID(), tenantId = randomUUID();
  return TrainingSnapshot.parse({ datasetVersionId: id, featurePipelineId: id, backend: "sklearn", algorithm, seed: 42,
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2 },
    dataset: { id, tenantId, createdAt: new Date().toISOString(), datasetId: id, version: 1, schema: { id, columns: [{ name: "x", type: "number" }, { name: "y", type: "number" }], target: "y" }, content: { key: "a".repeat(64), sha256: "a".repeat(64), size: 1, mediaType: "application/json" }, rowCount: 60, validation: { valid: true } },
    pipeline: { id, tenantId, logicalId: id, version: 1, name: "numeric", createdAt: new Date().toISOString(), definition: { features: ["x"], steps: [{ operation: "standard_scale" }] } } });
}
it("keeps ML disabled by default and validates the bounded sklearn contract", () => {
  expect(mlConfiguration({}).enabled).toBe(false);
  const backend = new SklearnBackend("unused"); const s = numericSnapshot();
  expect(() => backend.validate(s)).not.toThrow();
  for (const change of [{ algorithm: "arbitrary_import" }, { hyperparameters: { n_jobs: 100 } }, { split: { ...s.split, stratify: true } }, { pipeline: { ...s.pipeline, definition: { features: ["y"], steps: [] } } }]) expect(() => backend.validate({ ...s, ...change })).toThrow();
});
it.runIf(Boolean(process.env.ML_TEST_PYTHON))("trains, reloads, evaluates and predicts with real sklearn; freezes split/environment", async () => {
  const backend = new SklearnBackend(process.env.ML_TEST_PYTHON!); const s = numericSnapshot(); s.hyperparameters = { alpha: .01 };
  const rows = Array.from({ length: 60 }, (_, i) => ({ x: i, y: 2 * i + 1 })); const signal = new AbortController().signal;
  const prepared = await backend.prepare(s, rows, signal);
  expect(await backend.prepare(s, rows, signal)).toEqual(prepared);
  expect(new Set(Object.values(prepared.indices).flat()).size).toBe(rows.length);
  const model = await backend.train(s, rows, prepared, signal);
  const fittedMean = Number(execFileSync(process.env.ML_TEST_PYTHON!, ["-c", "import pickle,sys; print(pickle.loads(sys.stdin.buffer.read())['pipeline'][0].named_transformers_['numeric'][0].mean_[0])"], { input: model.bytes, windowsHide: true }).toString().trim());
  expect(fittedMean).toBeCloseTo(prepared.indices.train.reduce((sum, i) => sum + rows[i]!.x, 0) / prepared.indices.train.length, 12);
  const metrics = await backend.evaluate(s, rows, prepared, model, signal);
  expect(metrics).toHaveLength(6); expect(metrics.find(m => m.name === "mae" && m.partition === "test")!.value).toBeLessThan(2);
  const output = await backend.predict(s, model.bytes, [{ x: 10 }], model.environment, signal);
  expect(output[0]).toBeCloseTo(21, 0);
  await expect(backend.predict(s, model.bytes, [{ x: 10 }], { ...model.environment, sklearn: "wrong" }, signal)).rejects.toThrow();
  const aborted = new AbortController(); aborted.abort(); await expect(backend.prepare(s, rows, aborted.signal)).rejects.toThrow();
}, 60_000);
it.runIf(Boolean(process.env.ML_TEST_PYTHON)).each(["logistic_regression", "random_forest_classifier", "random_forest_regressor"])("executes the public %s algorithm with saved preprocessing", async algorithm => {
  const backend = new SklearnBackend(process.env.ML_TEST_PYTHON!); const s = numericSnapshot(algorithm);
  const classification = algorithm !== "random_forest_regressor";
  if (classification) { s.dataset.schema.columns[1]!.type = "string"; if (s.split.strategy === "random") s.split.stratify = true; }
  if (algorithm.startsWith("random_forest")) s.hyperparameters = { n_estimators: 8, max_depth: 6 };
  const rows = Array.from({ length: 60 }, (_, i) => ({ x: i, y: classification ? (i < 30 ? "lower" : "upper") : 2 * i + 1 }));
  const signal = new AbortController().signal; const prepared = await backend.prepare(s, rows, signal);
  const model = await backend.train(s, rows, prepared, signal);
  const metrics = await backend.evaluate(s, rows, prepared, model, signal);
  expect(metrics).toHaveLength(classification ? 3 : 6);
  const prediction = (await backend.predict(s, model.bytes, [{ x: 10 }], model.environment, signal))[0];
  if (classification) expect(prediction).toBe("lower"); else expect(Number.isFinite(prediction)).toBe(true);
}, 60_000);

it("applies generic target and modality validation only to v2 snapshots", () => {
  const backend = new SklearnBackend("unused");
  const columns = [{ name: "x", type: "number" as const }, { name: "y", type: "boolean" as const }];
  // v1 (A4): a boolean-target regression keeps the Phase 3 rules, which allow it.
  expect(() => backend.validate(snapshotFor({ columns }))).not.toThrow();
  expect(() => backend.validate(snapshotFor({ columns: [...columns, { name: "at", type: "timestamp" }], split: { strategy: "chronological", timeColumn: "at" } }))).toThrow(/Target type/);
  expect(() => backend.validate(snapshotFor({ target: null }))).toThrow(/requires a target/);
  for (const type of ["text", "image", "audio", "timestamp"] as const) expect(() => backend.validate(snapshotFor({ columns: [{ name: "x", type }, { name: "y", type: "number" }] }))).toThrow(/feature column types/);
});

it("enforces the chronological split rules on registered tabular models", () => {
  const backend = new SklearnBackend("unused"), at = { name: "at", type: "timestamp" as const };
  const chronological = (extra: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({ columns: [{ name: "x", type: "number" }, at, { name: "y", type: "number" }], split: { strategy: "chronological", timeColumn: "at" }, ...extra });
  expect(() => backend.validate(chronological())).not.toThrow();
  for (const [snapshot, message] of [
    [chronological({ split: { strategy: "chronological", timeColumn: "x" } }), /non-nullable timestamp column/],
    [chronological({ columns: [{ name: "x", type: "number" }, { ...at, nullable: true }, { name: "y", type: "number" }] }), /non-nullable timestamp column/],
    [chronological({ rowCount: 4 }), /at least one row/], // floor(4 * .2) = 0 validation rows
    [chronological({ evaluation: { cv: { strategy: "kfold", folds: 2 } } }), /chronological split/],
  ] as const) expect(() => backend.validate(snapshot)).toThrow(message);
});

it.runIf(Boolean(process.env.ML_TEST_PYTHON))("reloads and predicts from a stored v1 ridge snapshot with a boolean target", async () => {
  const backend = new SklearnBackend(process.env.ML_TEST_PYTHON!);
  const snapshot = snapshotFor({ columns: [{ name: "x", type: "number" }, { name: "y", type: "boolean" }], rowCount: 60 });
  const rows = Array.from({ length: 60 }, (_, i) => ({ x: i, y: i >= 30 })), signal = new AbortController().signal;
  const prepared = await backend.prepare(snapshot, rows, signal), model = await backend.train(snapshot, rows, prepared, signal);
  const stored = TrainingSnapshot.parse(JSON.parse(JSON.stringify(snapshot)));
  const predictions = await backend.predict(stored, model.bytes, [{ x: 10 }, { x: 50 }], model.environment, signal);
  expect(predictions).toHaveLength(2); expect(predictions.every(p => typeof p === "number" && Number.isFinite(p))).toBe(true);
  expect(predictions[1]).toBeGreaterThan(predictions[0] as number);
}, 60_000);

it.runIf(Boolean(process.env.ML_TEST_PYTHON))("trains and predicts ridge over a contract v2 chronological split with real sklearn", async () => {
  const backend = new SklearnBackend(process.env.ML_TEST_PYTHON!), signal = new AbortController().signal;
  const snapshot = chronologicalSnapshot("sklearn", { algorithm: "ridge", hyperparameters: { alpha: 1 } }, "regression");
  expect(contractFor(snapshot, "regression")).toBe(ML_BACKEND_CONTRACT_V2);
  const calls = vi.spyOn(backend as unknown as { call(...args: unknown[]): Promise<unknown> }, "call");
  const prepared = await backend.prepare(snapshot, CHRONO_ROWS, signal);
  expectTimeOrdered(prepared);
  const model = await backend.train(snapshot, CHRONO_ROWS, prepared, signal);
  const metrics = await backend.evaluate(snapshot, CHRONO_ROWS, prepared, model, signal);
  expect(metrics.find(m => m.name === "mae" && m.partition === "test")!.value).toBeLessThan(1);
  const predictions = await backend.predict(snapshot, model.bytes, CHRONO_ROWS.slice(0, 5), model.environment, signal);
  expect(predictions).toHaveLength(5); expect(predictions.every(p => typeof p === "number" && Number.isFinite(p))).toBe(true);
  expect(calls.mock.calls.map(c => c[0])).toEqual(["prepare", "train", "evaluate", "predict"]);
  for (const r of calls.mock.results) expect(((await r.value) as { contract: string }).contract).toBe(ML_BACKEND_CONTRACT_V2);
}, 60_000);
