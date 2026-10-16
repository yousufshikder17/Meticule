import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import type { MlBackend } from "../../src/ml/backend.js";
import { ForecastingBackend } from "../../src/ml/forecasting-backend.js";
import { LightgbmBackend } from "../../src/ml/gbdt-backend.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { ANOMALY_ROWS, SERIES_ROWS } from "../conformance/modality-rows.js";

// Forecasting and anomaly detection through the processor, registry and endpoints with real runners.
const sklearnPython = process.env.ML_TEST_PYTHON, gbdtPython = process.env.ML_TEST_LIGHTGBM_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
const signal = () => new AbortController().signal;
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

async function lifecycle(backend: MlBackend, columns: { name: string; type: string; nullable?: boolean }[], rows: Record<string, unknown>[], target: string | undefined, features: string[], spec: Record<string, unknown>) {
  const map = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const dataset = await training.ingest(manager, { name: `d-${randomUUID().slice(0, 8)}`, schema: { id: randomUUID(), columns, ...(target ? { target } : {}) }, rows });
  const pipeline = await training.pipeline(manager, { name: "p", definition: { features, steps: [] } });
  const experiment = await training.experiment(manager, `e-${randomUUID().slice(0, 8)}`);
  const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id, seed: 2, ...spec } });
  const worker = new LifecycleWorker(pool, { workerId: "series-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  while (await worker.tick());
  return { result: await training.getJob(manager, job.id), training, models };
}
async function serve(models: MlModelService, attemptId: string) {
  const registry = await models.registry(manager, { name: `r-${randomUUID().slice(0, 8)}` }); const version = await models.register(manager, registry.id, { trainingRunId: attemptId });
  await models.transition(manager, version.id, "READY");
  return models.endpoint(manager, { name: `e-${randomUUID().slice(0, 8)}`, modelVersionId: version.id });
}
const SERIES_COLUMNS = [{ name: "at", type: "timestamp" }, { name: "y", type: "number" }];
const chronological = { id: randomUUID(), strategy: "chronological", timeColumn: "at", train: .6, validation: .2, test: .2 };

describe.runIf(Boolean(sklearnPython))("forecasting through the durable lifecycle (Phase 4 M7)", () => {
  it("trains a ridge forecaster on a chronological split and serves horizon forecasts from history rows", async () => {
    const { result, models } = await lifecycle(new ForecastingBackend(sklearnPython!), SERIES_COLUMNS, SERIES_ROWS, "y", ["at"],
      { algorithm: "ridge_forecaster", hyperparameters: { lags: 4, windows: [6], horizon: 5 }, split: chronological, evaluation: { metrics: ["mae", "rmse", "smape", "mase"] } });
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].environment).toMatchObject({ backend: "forecasting", artifact_format: "forecast-npz-v1" });
    expect(new Set(result.metrics.map((m: { partition: string }) => m.partition))).toEqual(new Set(["train", "validation", "test"]));
    const endpoint = await serve(models, result.attempts[0].id);
    const prediction = await models.predict(manager, endpoint.id, SERIES_ROWS.slice(-24), signal());
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toHaveLength(5);
    expect(prediction.output.map((p: { step: number }) => p.step)).toEqual([1, 2, 3, 4, 5]);
    expect(Date.parse(prediction.output[0].timestamp)).toBe(Date.parse(SERIES_ROWS.at(-1)!.at as string) + 3_600_000);
    await expect(models.predict(outsider, endpoint.id, SERIES_ROWS.slice(-24), signal())).rejects.toThrow(/not found/);
  }, 300_000);

  it("refuses a random split for forecasting before any runner starts", async () => {
    const training = new MlTrainingService(pool, store);
    const dataset = await training.ingest(manager, { name: "s", schema: { id: randomUUID(), columns: SERIES_COLUMNS, target: "y" }, rows: SERIES_ROWS });
    const pipeline = await training.pipeline(manager, { name: "p", definition: { features: ["at"], steps: [] } });
    const experiment = await training.experiment(manager, "random-split");
    const backend = new ForecastingBackend("definitely-not-a-python-interpreter");
    const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "forecasting", algorithm: "ridge_forecaster", seed: 1,
      split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: false } } });
    const worker = new LifecycleWorker(pool, { workerId: "series-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, new Map([["forecasting", backend]])));
    while (await worker.tick());
    const failed = await training.getJob(manager, job.id);
    expect(failed.job.status).toBe("FAILED"); expect(failed.attempts[0].failure.category).toBe("invalid_request"); // refused by validation; the bogus interpreter never ran
  }, 120_000);
});

describe.runIf(Boolean(sklearnPython && gbdtPython))("LightGBM forecasting through the durable lifecycle", () => {
  it("trains lightgbm_forecaster and serves horizon forecasts", async () => {
    const { result, models } = await lifecycle(new LightgbmBackend(gbdtPython!), SERIES_COLUMNS, SERIES_ROWS, "y", ["at"],
      { algorithm: "lightgbm_forecaster", hyperparameters: { lags: 4, windows: [6], horizon: 3, n_estimators: 40, min_child_samples: 5 }, split: chronological, evaluation: { metrics: ["mae", "mase"] } });
    expect(result.job.status).toBe("COMPLETED"); expect(result.attempts[0].environment.artifact_format).toBe("lightgbm-text-v1");
    const prediction = await models.predict(manager, (await serve(models, result.attempts[0].id)).id, SERIES_ROWS.slice(-24), signal());
    expect(prediction.output).toHaveLength(3); expect(prediction.output.every((p: { value: number }) => Number.isFinite(p.value))).toBe(true);
  }, 300_000);
});

describe.runIf(Boolean(sklearnPython))("anomaly detection through the durable lifecycle", () => {
  const columns = [{ name: "a", type: "number" }, { name: "b", type: "number" }];
  it("fits without labels, serves {score, anomaly} per row, and reports only the label-free diagnostic", async () => {
    const { result, models } = await lifecycle(new SklearnBackend(sklearnPython!), columns, ANOMALY_ROWS.map(r => ({ a: r.a, b: r.b })), undefined, ["a", "b"],
      { algorithm: "isolation_forest", hyperparameters: { contamination: 0.1 }, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: false } });
    expect(result.job.status).toBe("COMPLETED"); expect(result.attempts[0].environment.artifact_schema).toBe("2");
    expect(new Set(result.metrics.map((m: { name: string }) => m.name))).toEqual(new Set(["threshold_calibration_error"]));
    const prediction = await models.predict(manager, (await serve(models, result.attempts[0].id)).id, [{ a: 0.1, b: 0.2 }, { a: 9, b: -9 }], signal());
    expect(prediction.status).toBe("COMPLETED");
    expect(prediction.output[1]).toMatchObject({ anomaly: true }); expect(prediction.output[0].anomaly).toBe(false);
    expect(prediction.output[1].score).toBeGreaterThan(prediction.output[0].score);
  }, 300_000);

  it("evaluates labelled data with the label-based metrics, using labels for scoring only", async () => {
    const { result } = await lifecycle(new SklearnBackend(sklearnPython!), [...columns, { name: "outlier", type: "boolean" }], ANOMALY_ROWS, "outlier", ["a", "b"],
      { algorithm: "robust_covariance", split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics: ["average_precision", "roc_auc", "anomaly_f1"] } });
    expect(result.job.status).toBe("COMPLETED");
    expect(result.metrics.find((m: { name: string; partition: string }) => m.name === "average_precision" && m.partition === "test")!.value).toBeGreaterThan(0.8);
  }, 300_000);
});
