import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import type { MlBackend } from "../../src/ml/backend.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlExperimentService } from "../../src/ml/experiments.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";

// Cross-framework matrix (Phase 3E): every framework trains on identical data, splits and folds; each framework reproduces
// itself; and no framework's artifact is served by another framework's runner.
const sklearnPython = process.env.ML_TEST_PYTHON;
const torchPython = process.env.ML_TEST_PYTORCH_PYTHON;
const tensorflowPython = process.env.ML_TEST_TENSORFLOW_PYTHON;
const runnable = Boolean(sklearnPython && torchPython && tensorflowPython);
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const columns = [{ name: "a", type: "number", nullable: true }, { name: "b", type: "number" }, { name: "kind", type: "string", nullable: true }, { name: "label", type: "string" }];
const rows = Array.from({ length: 120 }, (_, i) => ({ a: i % 9 === 0 ? null : (i % 2 ? 2.5 : -2.5) + (i % 7) * 0.1, b: (i * 13) % 11 / 10, kind: i % 5 === 0 ? null : ["red", "blue"][i % 2]!, label: i % 2 ? "yes" : "no" }));
const steps = [{ operation: "impute_median" }, { operation: "standard_scale" }, { operation: "impute_most_frequent" }, { operation: "one_hot_encode" }];
const mlpHyper = { hiddenLayers: [16], epochs: 20, batchSize: 16, learningRate: 0.01 };
const metrics = ["accuracy", "f1", "roc_auc", "log_loss"];
const split = { id: "00000000-0000-4000-8000-0000000000aa", strategy: "random", train: .6, validation: .2, test: .2, stratify: true };
const signal = () => new AbortController().signal;

async function fixture() {
  const backends: MlBackend[] = [new SklearnBackend(sklearnPython!), new PytorchBackend(torchPython!), new TensorflowBackend(tensorflowPython!)];
  const map = new Map(backends.map(b => [b.id, b]));
  const training = new MlTrainingService(pool, store), experiments = new MlExperimentService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "matrix", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "matrix", definition: { features: ["a", "b", "kind"], steps } });
  const experiment = await training.experiment(manager, "matrix");
  const worker = new LifecycleWorker(pool, { workerId: "matrix-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const drain = async () => { while (await worker.tick()); };
  /** One job per framework, with the algorithm that framework offers and a fixed split id, so repeats share one split. */
  const spec = (backend: string) => ({ datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend,
    algorithm: backend === "sklearn" ? "logistic_regression" : "mlp_classifier", hyperparameters: backend === "sklearn" ? {} : mlpHyper,
    seed: 11, split, evaluation: { metrics } });
  return { training, experiments, dataset, pipeline, experiment, drain, spec, map };
}

/** Trains one job for a framework and returns what the matrix needs: its snapshot, runtime record and artifact bytes. */
async function trainFor(f: Awaited<ReturnType<typeof fixture>>, backend: string) {
  const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec(backend) });
  await f.drain();
  const result = await f.training.getJob(manager, job.id);
  expect(result.job.status, backend).toBe("COMPLETED");
  return { result, snapshot: result.job.snapshot, environment: result.attempts[0].environment as Record<string, string>,
    artifact: await store.get(manager.tenantId, result.artifacts[0].content), metrics: result.metrics };
}

describe.runIf(runnable)("cross-framework matrix (Phase 3E)", () => {
  it("benchmarks sklearn, PyTorch and TensorFlow on identical splits and folds, and ranks them with per-framework lineage", async () => {
    const f = await fixture();
    const created = await f.experiments.benchmark(manager, {
      name: "three frameworks", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, taskType: "classification", seed: 5, backend: "sklearn",
      evaluation: { primaryMetric: "f1", secondaryMetrics: ["accuracy"], cv: { strategy: "stratified_kfold", folds: 3 } },
      candidates: [{ algorithm: "logistic_regression" }, { algorithm: "mlp_classifier", label: "pytorch mlp", backend: "pytorch", hyperparameters: mlpHyper }, { algorithm: "mlp_classifier", label: "tensorflow mlp", backend: "tensorflow", hyperparameters: mlpHyper }],
    });
    await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id, "benchmark");
    expect(done.orchestration.status).toBe("COMPLETED");
    const byBackend = Object.fromEntries(["sklearn", "pytorch", "tensorflow"].map(b => [b, done.candidates.find(c => c.lineage?.backend === b)!]));
    expect(Object.values(byBackend).every(Boolean)).toBe(true);
    const splits = await Promise.all(Object.values(byBackend).map(async c => (await f.training.getJob(manager, c.jobId)).attempts[0].split_indices));
    expect(splits[1]).toEqual(splits[0]); expect(splits[2]).toEqual(splits[0]);
    expect(done.ranking.entries).toHaveLength(3); expect(done.recommendation).not.toBeNull();
    for (const neural of [byBackend.pytorch, byBackend.tensorflow]) expect((await f.training.getJob(manager, neural!.jobId)).attempts[0].performance.cvModelsBuilt).toBe(3);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
  }, 600_000);

  it("reproduces each framework's own metrics exactly across two separate jobs with the same seed and split", async () => {
    const f = await fixture();
    for (const backend of ["sklearn", "pytorch", "tensorflow"]) {
      const first = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec(backend) });
      const second = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec(backend) });
      await f.drain();
      const [a, b] = [await f.training.getJob(manager, first.id), await f.training.getJob(manager, second.id)];
      expect(a.job.status, backend).toBe("COMPLETED"); expect(b.job.status, backend).toBe("COMPLETED");
      expect(b.attempts[0].split_indices, backend).toEqual(a.attempts[0].split_indices);
      expect(a.metrics.length, backend).toBeGreaterThan(0);
      for (const [i, metric] of a.metrics.entries()) expect(Math.abs(metric.value - b.metrics[i].value), `${backend} ${metric.name}`).toBeLessThan(1e-9);
    }
  }, 600_000);

  it("refuses every other framework's artifact at inference, and each framework serves its own", async () => {
    const f = await fixture();
    const trained = { sklearn: await trainFor(f, "sklearn"), pytorch: await trainFor(f, "pytorch"), tensorflow: await trainFor(f, "tensorflow") };
    const formats = Object.values(trained).map(t => t.environment.artifact_format);
    expect(new Set(formats).size).toBe(3); // each framework writes a distinct, recognisable format
    const sample = rows.slice(0, 20);
    for (const consumer of Object.keys(trained) as (keyof typeof trained)[]) {
      const backend = f.map.get(consumer)!;
      const own = await backend.predict(trained[consumer].snapshot, trained[consumer].artifact, sample, trained[consumer].environment, signal());
      expect(own, consumer).toHaveLength(sample.length);
      for (const producer of Object.keys(trained) as (keyof typeof trained)[]) {
        if (producer === consumer) continue;
        const attempt = backend.predict(trained[consumer].snapshot, trained[producer].artifact, sample, trained[producer].environment, signal());
        await expect(attempt, `${producer} artifact on ${consumer}`).rejects.toThrow();
        // Refusals must not expose paths or stack traces, whichever framework refused.
        await attempt.catch((error: Error) => expect(error.message, `${producer} on ${consumer}`).not.toMatch(/Traceback|[A-Za-z]:\\|\/Users\//));
      }
    }
  }, 600_000);
});
