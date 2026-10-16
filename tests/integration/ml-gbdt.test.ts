import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlExperimentService } from "../../src/ml/experiments.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { LightgbmBackend, XgboostBackend } from "../../src/ml/gbdt-backend.js";
import type { MlBackend } from "../../src/ml/backend.js";

// The real XGBoost and LightGBM runners through the durable lifecycle: train -> register -> READY -> endpoint predict ->
// monitor, artifact refusals, tenant isolation, and a benchmark ranking both against scikit-learn on identical splits.
const sklearnPython = process.env.ML_TEST_PYTHON;
const gbdtPython = process.env.ML_TEST_XGBOOST_PYTHON && process.env.ML_TEST_LIGHTGBM_PYTHON ? process.env.ML_TEST_XGBOOST_PYTHON : undefined;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const columns = [{ name: "a", type: "number", nullable: true }, { name: "b", type: "number" }, { name: "kind", type: "string", nullable: true }, { name: "label", type: "string" }];
// Deterministic fixture with missing numbers (passed through as NaN, no imputer) and a categorical feature.
const rows = Array.from({ length: 150 }, (_, i) => ({ a: i % 9 === 0 ? null : (i % 2 ? 2 : -2) + (i % 7) * 0.1, b: (i * 13) % 11 / 10, kind: i % 5 === 0 ? null : ["red", "blue"][i % 2]!, label: i % 2 ? "yes" : "no" }));
const steps = [{ operation: "impute_most_frequent" }, { operation: "one_hot_encode" }];

async function fixture() {
  const map = new Map<string, MlBackend>([new SklearnBackend(sklearnPython!), new XgboostBackend(gbdtPython!), new LightgbmBackend(gbdtPython!)].map(b => [b.id, b]));
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map), experiments = new MlExperimentService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "gbdt", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "gbdt", definition: { features: ["a", "b", "kind"], steps } });
  const experiment = await training.experiment(manager, "gbdt");
  const worker = new LifecycleWorker(pool, { workerId: "gbdt-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const drain = async () => { while (await worker.tick()); };
  const queue = (backend: string, hyperparameters: Record<string, unknown> = {}) => training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id,
    backend, algorithm: `${backend}_classifier`, hyperparameters: { n_estimators: 40, ...(backend === "lightgbm" ? { min_child_samples: 5 } : {}), ...hyperparameters }, seed: 3,
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics: ["accuracy", "roc_auc", "log_loss"] } } });
  return { training, models, experiments, dataset, pipeline, drain, queue, map };
}

describe.runIf(Boolean(sklearnPython && gbdtPython))("real XGBoost and LightGBM backends through the durable lifecycle", () => {
  for (const backend of ["xgboost", "lightgbm"]) {
    it(`${backend}: trains natively with early stopping, registers, promotes and serves version-pinned predictions`, async () => {
      const f = await fixture(); const job = await f.queue(backend, { n_estimators: 300, early_stopping_rounds: 10 }); await f.drain();
      const result = await f.training.getJob(manager, job.id); const attempt = result.attempts[0];
      expect(result.job.status).toBe("COMPLETED");
      expect(attempt.environment).toMatchObject({ backend, artifact_format: backend === "xgboost" ? "xgboost-json-v1" : "lightgbm-text-v1", protocol: `meticule-${backend}-v1` });
      expect(attempt.resolved_hyperparameters.best_iteration).toBe(attempt.performance.bestIteration);
      expect(attempt.performance.bestIteration).toBeGreaterThanOrEqual(1); expect(attempt.performance.bestIteration).toBeLessThanOrEqual(300);
      expect(result.metrics.find((m: { name: string; partition: string }) => m.name === "accuracy" && m.partition === "test")!.value).toBeGreaterThan(0.9);
      const registry = await f.models.registry(manager, { name: `${backend}-classifier` }); const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
      await f.models.transition(manager, version.id, "READY");
      const endpoint = await f.models.endpoint(manager, { name: `${backend}-live`, modelVersionId: version.id });
      const prediction = await f.models.predict(manager, endpoint.id, [{ a: 2.2, b: 0.4, kind: "red" }, { a: -2.2, b: 0.4, kind: "blue" }, { a: null, b: 0.9, kind: null }], new AbortController().signal);
      expect(prediction.status).toBe("COMPLETED"); expect(prediction.output.slice(0, 2)).toEqual(["yes", "no"]); expect(prediction.model_version_id).toBe(version.id);
      expect(await f.models.monitoring(manager, endpoint.id)).toMatchObject({ requests: 1, failed: 0, pending: 0 });
      await expect(f.models.predict(outsider, endpoint.id, [{ a: 2.2, b: 0.4, kind: "red" }], new AbortController().signal)).rejects.toThrow(/not found/);
      const bytes = await store.get(manager.tenantId, result.artifacts[0].content);
      await expect(f.map.get(backend)!.predict(result.job.snapshot, bytes, rows.slice(0, 2), { ...attempt.environment, [backend]: "0.0.1" }, new AbortController().signal)).rejects.toThrow();
    }, 300_000);
  }

  it("refuses another framework's artifact in each direction", async () => {
    const f = await fixture(); const [x, l] = [await f.queue("xgboost"), await f.queue("lightgbm")]; await f.drain();
    const xr = await f.training.getJob(manager, x.id), lr = await f.training.getJob(manager, l.id);
    const xb = await store.get(manager.tenantId, xr.artifacts[0].content), lb = await store.get(manager.tenantId, lr.artifacts[0].content);
    await expect(f.map.get("lightgbm")!.predict(lr.job.snapshot, xb, rows.slice(0, 2), xr.attempts[0].environment, new AbortController().signal)).rejects.toThrow();
    await expect(f.map.get("xgboost")!.predict(xr.job.snapshot, lb, rows.slice(0, 2), lr.attempts[0].environment, new AbortController().signal)).rejects.toThrow();
  }, 300_000);

  it("benchmarks scikit-learn, XGBoost and LightGBM candidates on the same split and folds and ranks them", async () => {
    const f = await fixture();
    const created = await f.experiments.benchmark(manager, {
      name: "trees", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, taskType: "classification", seed: 5, backend: "sklearn",
      evaluation: { primaryMetric: "accuracy", secondaryMetrics: ["log_loss"], cv: { strategy: "stratified_kfold", folds: 3 } },
      candidates: [{ algorithm: "random_forest_classifier" }, { algorithm: "xgboost_classifier", backend: "xgboost", hyperparameters: { n_estimators: 40 } },
        { algorithm: "lightgbm_classifier", backend: "lightgbm", hyperparameters: { n_estimators: 40, min_child_samples: 5 } }],
    });
    await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id, "benchmark");
    expect(done.orchestration.status).toBe("COMPLETED");
    expect(done.candidates.map(c => c.lineage!.backend).sort()).toEqual(["lightgbm", "sklearn", "xgboost"]);
    const splits = await Promise.all(done.candidates.map(async c => (await f.training.getJob(manager, c.jobId)).attempts[0].split_indices));
    expect(splits[1]).toEqual(splits[0]); expect(splits[2]).toEqual(splits[0]);
    expect(done.ranking.entries).toHaveLength(3); expect(done.recommendation).not.toBeNull();
  }, 600_000);
});
