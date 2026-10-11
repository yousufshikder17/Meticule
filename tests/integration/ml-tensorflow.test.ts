import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";

const tensorflowPython = process.env.ML_TEST_TENSORFLOW_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 6 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_checkpoints LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const columns = [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "label", type: "string" }];
const rows = Array.from({ length: 96 }, (_, i) => ({ a: (i % 2 ? 2 : -2) + (i % 7) * 0.05, b: (i * 7) % 9 / 9, label: i % 2 ? "yes" : "no" }));

async function fixture(device: "cpu" | "cuda" = "cpu", precision: "fp32" | "mixed" = "fp32") {
  const backend = new TensorflowBackend(tensorflowPython!);
  const training = new MlTrainingService(pool, store);
  const dataset = await training.ingest(manager, { name: "tensorflow", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "tensorflow", definition: { features: ["a", "b"], steps: [{ operation: "standard_scale" }] } });
  const experiment = await training.experiment(manager, "tensorflow");
  const job = (extra: Record<string, unknown> = {}) => training.queue(manager, { experimentId: experiment.id, spec: {
    datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "tensorflow", algorithm: "mlp_classifier", seed: 21,
    hyperparameters: { hiddenLayers: [8], epochs: 4, batchSize: 16, learningRate: 0.01, dropout: 0, optimizer: "adam" }, evaluation: { metrics: ["accuracy"] },
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true },
    training: { checkpointEveryEpochs: 2, earlyStopping: null, device, precision }, ...extra } });
  const drain = async () => { const worker = new LifecycleWorker(pool, { workerId: "tensorflow-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, new Map([[backend.id, backend]]))); while (await worker.tick()); };
  return { training, job, drain };
}

describe.runIf(Boolean(tensorflowPython))("TensorFlow backend through the durable lifecycle (Phase 3D)", () => {
  it("trains in bounded chunks on CPU with TensorFlow provenance and completes", async () => {
    const f = await fixture(); const job = await f.job(); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].environment).toMatchObject({ backend: "tensorflow", artifact_format: "tensorflow-variables-v1", protocol: "meticule-tensorflow-v1", device_requested: "cpu", device_actual: "cpu", cuda_runtime: "none" });
    expect(result.metrics.length).toBeGreaterThan(0);
    expect(result.artifacts).toHaveLength(1);
  }, 240_000);

  it("fails a CUDA request explicitly: the TensorFlow runtime has no CUDA path, so the job never runs on CPU instead", async () => {
    const f = await fixture("cuda"); const job = await f.job(); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED");
    expect(result.attempts[0].failure.message).toMatch(/does not support device cuda/);
    expect(result.metrics).toHaveLength(0); expect(result.artifacts).toHaveLength(0);
  }, 240_000);

  it("refuses mixed precision at queueing", async () => {
    const f = await fixture("cpu", "mixed"); const count = async () => (await pool.query("SELECT count(*)::int AS n FROM ml_training_jobs")).rows[0].n;
    await expect(f.job()).rejects.toThrow();
    expect(await count()).toBe(0);
  }, 60_000);
});
