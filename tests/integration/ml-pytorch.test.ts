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
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import type { MlBackend } from "../../src/ml/backend.js";

const sklearnPython = process.env.ML_TEST_PYTHON;
const torchPython = process.env.ML_TEST_PYTORCH_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

/**
 * Simulates storage corruption beneath the append-only blob table, as a disk or operator fault would. The trigger is lifted
 * only inside this transaction; the artifact store must still detect the change through its content hash.
 */
async function corruptBlob(tenantId: string, key: string, content: string): Promise<void> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("ALTER TABLE ml_blobs DISABLE TRIGGER ml_blob_append_only");
    const updated = await client.query("UPDATE ml_blobs SET content=$3 WHERE tenant_id=$1 AND content_hash=$2", [tenantId, key, Buffer.from(content)]);
    if (updated.rowCount !== 1) throw new Error("blob to corrupt was not found");
    await client.query("ALTER TABLE ml_blobs ENABLE TRIGGER ml_blob_append_only");
    await client.query("COMMIT");
  } catch (error) { await client.query("ROLLBACK"); throw error; } finally { client.release(); }
}

const columns = [{ name: "a", type: "number", nullable: true }, { name: "b", type: "number" }, { name: "kind", type: "string", nullable: true }, { name: "label", type: "string" }];
// Deterministic, linearly separable fixture with a categorical feature and missing numbers. No downloads.
const rows = Array.from({ length: 120 }, (_, i) => ({ a: i % 9 === 0 ? null : (i % 2 ? 2.5 : -2.5) + (i % 7) * 0.1, b: (i * 13) % 11 / 10, kind: i % 5 === 0 ? null : ["red", "blue"][i % 2]!, label: i % 2 ? "yes" : "no" }));
const steps = [{ operation: "impute_median" }, { operation: "standard_scale" }, { operation: "impute_most_frequent" }, { operation: "one_hot_encode" }];
const mlpHyper = { hiddenLayers: [16], epochs: 30, batchSize: 16, learningRate: 0.01 };
const metrics = ["accuracy", "f1", "roc_auc", "log_loss"];

async function fixture(backends: MlBackend[] = [new SklearnBackend(sklearnPython!), new PytorchBackend(torchPython!)]) {
  const map = new Map(backends.map(b => [b.id, b]));
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map), experiments = new MlExperimentService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "mixed", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "mixed", definition: { features: ["a", "b", "kind"], steps } });
  const experiment = await training.experiment(manager, "pytorch");
  const worker = new LifecycleWorker(pool, { workerId: "torch-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const drain = async () => { while (await worker.tick()); };
  const spec = (overrides: Record<string, unknown> = {}) => ({ datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "pytorch", algorithm: "mlp_classifier", hyperparameters: mlpHyper, seed: 11,
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics }, ...overrides });
  return { training, models, experiments, dataset, pipeline, experiment, worker, drain, spec, map };
}

describe.runIf(Boolean(torchPython))("real PyTorch backend through the durable lifecycle", () => {
  it("trains a real MLP on CPU, persists provenance and metrics, and records final losses and timings", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    const attempt = result.attempts[0];
    expect(attempt.environment).toMatchObject({ backend: "pytorch", artifact_format: "pytorch-state-dict-v1", artifact_schema: "1", protocol: "meticule-pytorch-v1" });
    expect(attempt.environment.torch).toMatch(/^2\.9\.1/);
    expect(attempt.resolved_hyperparameters).toMatchObject({ hiddenLayers: [16], epochs: 30, optimizer: "adam", loss: "cross_entropy" });
    expect(attempt.performance).toMatchObject({ epochsCompleted: 30 }); expect(attempt.performance.finalTrainLoss).toBeGreaterThan(0); expect(attempt.performance.finalValidationLoss).toBeGreaterThan(0); expect(attempt.performance.fitSeconds).toBeGreaterThan(0);
    expect(result.metrics).toHaveLength(12);
    const accuracy = result.metrics.find((m: { name: string; partition: string }) => m.name === "accuracy" && m.partition === "test")!;
    expect(accuracy.value).toBeGreaterThan(0.85);
    expect(result.artifacts).toHaveLength(1);
  }, 120_000);

  it("registers and promotes the trained model explicitly, then serves it through version-pinned inference", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }); await f.drain();
    const attempt = (await f.training.getJob(manager, job.id)).attempts[0];
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
    const registry = await f.models.registry(manager, { name: "torch-classifier" }); const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
    await expect(f.models.endpoint(manager, { name: "too-early", modelVersionId: version.id })).rejects.toThrow(/READY/);
    await f.models.transition(manager, version.id, "READY");
    const endpoint = await f.models.endpoint(manager, { name: "torch-live", modelVersionId: version.id });
    const batch = [{ a: 2.5, b: 0.4, kind: "red" }, { a: -2.5, b: 0.4, kind: "blue" }, { a: 2.5, b: 0.9, kind: null }];
    const prediction = await f.models.predict(manager, endpoint.id, batch, new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toHaveLength(3);
    expect(prediction.output.every((label: unknown) => label === "yes" || label === "no")).toBe(true);
    expect(prediction.output[0]).toBe("yes"); expect(prediction.output[1]).toBe("no");
    expect(prediction.model_version_id).toBe(version.id);
  }, 120_000);

  it("benchmarks LogisticRegression against an MLP on identical splits and folds with per-candidate backends", async () => {
    const f = await fixture();
    const created = await f.experiments.benchmark(manager, {
      name: "linear vs neural", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, taskType: "classification", seed: 5, backend: "sklearn",
      evaluation: { primaryMetric: "f1", secondaryMetrics: ["accuracy"], cv: { strategy: "stratified_kfold", folds: 3 } },
      candidates: [{ algorithm: "logistic_regression" }, { algorithm: "mlp_classifier", backend: "pytorch", hyperparameters: mlpHyper }],
    });
    await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id, "benchmark");
    expect(done.orchestration.status).toBe("COMPLETED");
    const [linear, neural] = [done.candidates.find(c => c.algorithm === "logistic_regression")!, done.candidates.find(c => c.algorithm === "mlp_classifier")!];
    expect(linear.lineage!.backend).toBe("sklearn"); expect(neural.lineage!.backend).toBe("pytorch");
    expect((await f.training.getJob(manager, linear.jobId)).attempts[0].split_indices).toEqual((await f.training.getJob(manager, neural.jobId)).attempts[0].split_indices);
    expect(done.ranking.entries).toHaveLength(2); expect(done.recommendation).not.toBeNull();
    const neuralAttempt = (await f.training.getJob(manager, neural.jobId)).attempts[0];
    expect(neuralAttempt.performance.cvModelsBuilt).toBe(3);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
  }, 300_000);

  it("reproduces predictions and metrics across separate runner processes with the same seed and data", async () => {
    const f = await fixture(); const first = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }), second = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() });
    await f.drain();
    const a = await f.training.getJob(manager, first.id), b = await f.training.getJob(manager, second.id);
    expect(b.attempts[0].split_indices).toEqual(a.attempts[0].split_indices);
    expect(a.metrics.map((m: { value: number }) => m.value).every((v: number, i: number) => Math.abs(v - b.metrics[i].value) < 1e-9)).toBe(true);
    const artifactBytes = async (job: typeof a) => store.get(manager.tenantId, job.artifacts[0].content);
    const backend = f.map.get("pytorch")!; const sample = rows.slice(0, 25);
    const predA = await backend.predict(a.job.snapshot, await artifactBytes(a), sample, a.attempts[0].environment, new AbortController().signal);
    const predB = await backend.predict(b.job.snapshot, await artifactBytes(b), sample, b.attempts[0].environment, new AbortController().signal);
    expect(predA).toEqual(predB);
  }, 300_000);

  it("refuses artifacts whose recorded runtime differs, before reading their bytes", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }); await f.drain();
    const result = await f.training.getJob(manager, job.id); const backend = f.map.get("pytorch")!;
    const bytes = await store.get(manager.tenantId, result.artifacts[0].content);
    const env = result.attempts[0].environment;
    await expect(backend.predict(result.job.snapshot, bytes, rows.slice(0, 2), { ...env, torch: "0.0.1" }, new AbortController().signal)).rejects.toThrow();
    await expect(backend.predict(result.job.snapshot, Buffer.from("not an artifact"), rows.slice(0, 2), { ...env, sklearn: "0.0.1" }, new AbortController().signal)).rejects.toThrow();
    // Provenance drift alone (adapter checksum) is tolerated: the same artifact still serves.
    const drifted = await backend.predict(result.job.snapshot, bytes, rows.slice(0, 2), { ...env, adapter_sha256: "0".repeat(64) }, new AbortController().signal);
    expect(drifted).toHaveLength(2);
  }, 120_000);

  it("refuses tampered artifact bytes through the content check and reports a generic failure", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    const registry = await f.models.registry(manager, { name: "tamper" }); const version = await f.models.register(manager, registry.id, { trainingRunId: result.attempts[0].id });
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "tamper", modelVersionId: version.id });
    await corruptBlob(manager.tenantId, result.artifacts[0].content.key, "tampered");
    const failed = await f.models.predict(manager, endpoint.id, [{ a: 1, b: 0.2, kind: "red" }], new AbortController().signal);
    expect(failed.status).toBe("FAILED"); expect(JSON.stringify(failed)).not.toMatch(/tampered|checksum|pytorch|torch/i);
  }, 120_000);

  it("rejects invalid architectures, options and unsupported preprocessing at benchmark creation, and fails them durably as single jobs", async () => {
    const f = await fixture(); const count = async () => (await pool.query("SELECT count(*)::int AS n FROM ml_training_jobs")).rows[0].n;
    const benchmark = (hyperparameters: Record<string, unknown>, pipelineId = f.pipeline.id) => f.experiments.benchmark(manager, {
      name: "invalid", datasetVersionId: f.dataset.id, featurePipelineId: pipelineId, taskType: "classification", seed: 1, backend: "sklearn",
      evaluation: { primaryMetric: "f1", cv: { strategy: "stratified_kfold", folds: 3 } },
      candidates: [{ algorithm: "logistic_regression" }, { algorithm: "mlp_classifier", backend: "pytorch", hyperparameters }],
    });
    await expect(benchmark({ hiddenLayers: [512, 512] })).rejects.toThrow();
    await expect(benchmark({ optimizer: "lbfgs" })).rejects.toThrow();
    await expect(benchmark({ epochs: 0 })).rejects.toThrow();
    const variance = await f.training.pipeline(manager, { name: "variance", definition: { features: ["a", "b"], steps: [{ operation: "variance_threshold" }] } });
    await expect(benchmark(mlpHyper, variance.id)).rejects.toThrow(/does not support preprocessing/);
    expect(await count()).toBe(0);
    // A single job is accepted at queueing (as in V1) and fails durably during preparation, with no metrics or artifact.
    const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec({ hyperparameters: { optimizer: "lbfgs" } }) }); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED"); expect(result.attempts[0].failure.code).toBe("ml_training_failed");
    expect(result.metrics).toHaveLength(0); expect(result.artifacts).toHaveLength(0);
  }, 120_000);

  it("isolates PyTorch jobs, artifacts and endpoints by tenant", async () => {
    const outsider: Principal = { tenantId: "33333333-3333-4333-8333-333333333333", userId: "44444444-4444-4444-8444-444444444444", roles: ["tenant_admin"] };
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec() }); await f.drain();
    const attempt = (await f.training.getJob(manager, job.id)).attempts[0];
    const registry = await f.models.registry(manager, { name: "isolated" }); const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "isolated", modelVersionId: version.id });
    await expect(f.training.getJob(outsider, job.id)).rejects.toThrow(/not found/);
    await expect(f.models.predict(outsider, endpoint.id, [{ a: 2.5, b: 0.4, kind: "red" }], new AbortController().signal)).rejects.toThrow(/not found/);
    await expect(f.models.register(outsider, registry.id, { trainingRunId: attempt.id })).rejects.toThrow(/not found/);
    // The owner's endpoint still serves after the foreign attempts above.
    expect((await f.models.predict(manager, endpoint.id, [{ a: 2.5, b: 0.4, kind: "red" }], new AbortController().signal)).status).toBe("COMPLETED");
  }, 120_000);

  it("keeps the sklearn backend working alongside PyTorch in one deployment", async () => {
    const f = await fixture(); const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: f.spec({ backend: "sklearn", algorithm: "logistic_regression", hyperparameters: {} }) }); await f.drain();
    expect((await f.training.getJob(manager, job.id)).job.status).toBe("COMPLETED");
  }, 120_000);
});
