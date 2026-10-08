import { randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import { expect, it } from "vitest";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TrainingSnapshot } from "../../src/ml/domain.js";
import { mlConfiguration } from "../../src/ml/configuration.js";

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
  if (classification) { s.dataset.schema.columns[1]!.type = "string"; s.split.stratify = true; }
  if (algorithm.startsWith("random_forest")) s.hyperparameters = { n_estimators: 8, max_depth: 6 };
  const rows = Array.from({ length: 60 }, (_, i) => ({ x: i, y: classification ? (i < 30 ? "lower" : "upper") : 2 * i + 1 }));
  const signal = new AbortController().signal; const prepared = await backend.prepare(s, rows, signal);
  const model = await backend.train(s, rows, prepared, signal);
  const metrics = await backend.evaluate(s, rows, prepared, model, signal);
  expect(metrics).toHaveLength(classification ? 3 : 6);
  const prediction = (await backend.predict(s, model.bytes, [{ x: 10 }], model.environment, signal))[0];
  if (classification) expect(prediction).toBe("lower"); else expect(Number.isFinite(prediction)).toBe(true);
}, 60_000);
