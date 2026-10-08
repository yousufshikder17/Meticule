import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";

const python = process.env.ML_TEST_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_jobs LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const columns = [{ name: "a", type: "number" }, { name: "b", type: "number", nullable: true }, { name: "kind", type: "string", nullable: true }, { name: "flag", type: "boolean" }, { name: "label", type: "string" }];
const rows = Array.from({ length: 120 }, (_, i) => ({ a: i % 12, b: i % 7 === 0 ? null : (i * 7) % 11, kind: i % 5 === 0 ? null : ["red", "blue", "green"][i % 3]!, flag: i % 2 === 0, label: i % 12 >= 6 ? "hi" : "lo" }));

class LeakyFoldsBackend extends SklearnBackend {
  override async prepare(...args: Parameters<SklearnBackend["prepare"]>) { const p = await super.prepare(...args); p.folds![0]![0] = p.indices.test[0]!; return p; }
}
async function fixture(backend: SklearnBackend = new SklearnBackend(python!)) {
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, new Map([[backend.id, backend]]));
  const dataset = await training.ingest(manager, { name: "mixed", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "mixed", definition: { features: ["a", "b", "kind", "flag"], steps: [{ operation: "impute_median" }, { operation: "standard_scale" }, { operation: "impute_most_frequent" }, { operation: "one_hot_encode" }] } });
  const experiment = await training.experiment(manager, "cv");
  const spec = { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "sklearn", algorithm: "random_forest_classifier", hyperparameters: { n_estimators: 20 }, seed: 11, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics: ["accuracy", "f1", "roc_auc", "log_loss"], cv: { strategy: "stratified_kfold", folds: 3 } } };
  const worker = new LifecycleWorker(pool, { workerId: "cv-test", leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, new Map([[backend.id, backend]])));
  return { training, models, experiment, spec, worker, dataset };
}

describe.runIf(Boolean(python))("cross-validated training through the durable lifecycle", () => {
  it("persists fold indices, per-fold and aggregate metrics, and timings; reproduces identically", async () => {
    const f = await fixture();
    const first = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec }), second = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec });
    await f.worker.tick(); await f.worker.tick();
    const a = await f.training.getJob(manager, first.id), b = await f.training.getJob(manager, second.id);
    expect(a.job.status).toBe("COMPLETED");
    const attempt = a.attempts[0], indices = attempt.split_indices as Record<string, number[]>;
    expect(Object.keys(indices).sort()).toEqual(["fold_0", "fold_1", "fold_2", "test", "train", "validation"]);
    expect([...indices.fold_0!, ...indices.fold_1!, ...indices.fold_2!].sort((x, y) => x - y)).toEqual([...indices.train!].sort((x, y) => x - y));
    const metrics = a.metrics as { name: string; partition: string; fold: number | null; std: number | null; value: number; direction: string }[];
    expect(metrics.filter(m => m.partition === "cv_fold")).toHaveLength(12);
    const cv = metrics.filter(m => m.partition === "cv"); expect(cv.map(m => m.name).sort()).toEqual(["accuracy", "f1", "log_loss", "roc_auc"]);
    for (const m of cv) {
      const folds = metrics.filter(x => x.partition === "cv_fold" && x.name === m.name).map(x => x.value);
      expect(m.value).toBeCloseTo(folds.reduce((s, v) => s + v, 0) / 3, 9); expect(m.std).toBeGreaterThanOrEqual(0);
    }
    expect(cv.find(m => m.name === "log_loss")!.direction).toBe("lower"); expect(cv.find(m => m.name === "f1")!.direction).toBe("higher");
    expect(Object.keys(attempt.performance).sort()).toEqual(["cvFitSeconds", "fitSeconds", "predictSeconds"]);
    expect(attempt.resolved_hyperparameters.n_estimators).toBe(20);
    // Same data, pipeline, seed and CV definition: identical folds and materially equal scores.
    expect(b.attempts[0].split_indices).toEqual(attempt.split_indices);
    const deltas = await f.models.compare(manager, attempt.id, b.attempts[0].id);
    expect(deltas.every(d => !d.regressed && Math.abs(d.delta) < 1e-9)).toBe(true);
    await expect(pool.query("UPDATE ml_training_runs SET performance='{}' WHERE id=$1", [attempt.id])).rejects.toThrow(/immutable/);
  }, 120_000);
  it("serves categorical inference from the persisted transformation and rejects schema drift", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec }); await f.worker.tick();
    const attempt = (await f.training.getJob(manager, job.id)).attempts[0];
    const registry = await f.models.registry(manager, { name: "mixed" }); const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "mixed", modelVersionId: version.id });
    const predict = (batch: unknown[]) => f.models.predict(manager, endpoint.id, batch, new AbortController().signal);
    const ok = await predict([{ a: 11, b: null, kind: "never-seen-at-training", flag: true }, { a: 0, b: 3, kind: null, flag: false }]);
    expect(ok.status).toBe("COMPLETED"); expect(ok.output).toEqual(["hi", "lo"]);
    await expect(predict([{ a: 1, b: 1, kind: "red" }])).rejects.toThrow(/Row 0/);
    await expect(predict([{ a: 1, b: 1, kind: "red", flag: true, label: "lo" }])).rejects.toThrow(/Row 0/);
    await expect(predict([{ a: "1", b: 1, kind: "red", flag: true }])).rejects.toThrow(/invalid a/);
  }, 120_000);
  it("fails the job instead of training when a backend returns folds that touch held-out rows", async () => {
    const f = await fixture(new LeakyFoldsBackend(python!)); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec }); await f.worker.tick();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED"); expect(result.attempts[0].failure.message).toMatch(/invalid CV folds/); expect(result.metrics).toHaveLength(0); expect(result.artifacts).toHaveLength(0);
  }, 60_000);
});
