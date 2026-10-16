import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { RunRepository } from "../../src/db/repositories.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { JaxBackend } from "../../src/ml/jax-backend.js";
import type { ChunkResult, MlBackend, PreparedData } from "../../src/ml/backend.js";
import type { Row, Snapshot } from "../../src/ml/domain.js";

// The real JAX runner through the durable lifecycle: chunked training with persisted checkpoints, recovery after a lost
// lease, fencing, early stopping, registry, version-pinned inference and tenant isolation. No downloads, CPU only.
const jaxPython = process.env.ML_TEST_JAX_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_checkpoints LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const columns = [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "label", type: "string" }];
const rows = Array.from({ length: 120 }, (_, i) => ({ a: (i % 2 ? 2 : -2) + (i % 7) * 0.05, b: (i * 13) % 11 / 10, label: i % 2 ? "yes" : "no" }));
// Every fifth label flipped: a wide network overfits it, so the validation loss genuinely turns upward.
const noisyRows = rows.map((r, i) => ({ ...r, a: r.a * 0.3, label: i % 5 === 0 ? (r.label === "yes" ? "no" : "yes") : r.label }));
const base = { algorithm: "mlp_classifier", hyperparameters: { hiddenLayers: [8], epochs: 6, batchSize: 16, learningRate: 0.02, dropout: 0.2 }, seed: 17, evaluation: { metrics: ["accuracy", "log_loss"] } };

/** Runs a hook after each chunk, to inject the failures the durable path must survive. */
class HookedJax extends JaxBackend {
  private calls = 0;
  constructor(python: string, private readonly hook: (call: number) => Promise<void>) { super(python); }
  override async trainChunk(snapshot: Snapshot, rowsIn: Row[], prepared: PreparedData, state: Buffer | null, epochs: number, signal: AbortSignal): Promise<ChunkResult> {
    const result = await super.trainChunk(snapshot, rowsIn, prepared, state, epochs, signal);
    await this.hook(++this.calls);
    return result;
  }
}

async function fixture(backend: MlBackend = new JaxBackend(jaxPython!), data: Record<string, unknown>[] = rows) {
  const map = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "jax", schema: { id: randomUUID(), columns, target: "label" }, rows: data });
  const pipeline = await training.pipeline(manager, { name: "jax", definition: { features: ["a", "b"], steps: [{ operation: "standard_scale" }] } });
  const experiment = await training.experiment(manager, "jax");
  const job = (extra: Record<string, unknown> = {}) => training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "jax",
    ...base, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, training: { checkpointEveryEpochs: 2, earlyStopping: null }, ...extra } });
  const worker = (id: string) => new LifecycleWorker(pool, { workerId: id, leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const drain = async (id = "jax-main") => { const w = worker(id); while (await w.tick()); };
  return { training, models, job, worker, drain, map };
}

const sameMetrics = (a: { name: string; partition: string; value: number }[], b: { name: string; partition: string; value: number }[]) =>
  a.length > 0 && a.length === b.length && a.every(m => Math.abs(m.value - b.find(r => r.name === m.name && r.partition === m.partition)!.value) < 1e-9);

describe.runIf(Boolean(jaxPython))("real JAX backend through the durable lifecycle", () => {
  it("trains in bounded chunks, persists every epoch and a JAX checkpoint per chunk, and records real JAX provenance", async () => {
    const f = await fixture(); const job = await f.job(); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    expect((await f.training.history(manager, job.id)).map((e: { epoch: number }) => e.epoch)).toEqual([1, 2, 3, 4, 5, 6]);
    const checkpoints = await pool.query("SELECT epoch,format FROM ml_training_checkpoints ORDER BY epoch");
    expect(checkpoints.rows).toEqual([2, 4, 6].map(epoch => ({ epoch, format: "jax-checkpoint-v1" })));
    expect(result.attempts[0].environment).toMatchObject({ backend: "jax", artifact_format: "jax-params-v1", protocol: "meticule-jax-v1", jax: "0.11.2", jax_platform: "cpu", device_actual: "cpu" });
    expect(result.metrics.find((m: { name: string; partition: string }) => m.name === "accuracy" && m.partition === "test")!.value).toBeGreaterThan(0.9);
  }, 300_000);

  it("resumes after a lost lease from the durable checkpoint and reproduces the uninterrupted run exactly (dropout on)", async () => {
    const reference = await fixture(); const refJob = await reference.job(); await reference.drain();
    const refResult = await reference.training.getJob(manager, refJob.id);

    const f = await fixture(); const job = await f.job();
    // The second chunk finishes, then the lease expires before its checkpoint is written: only epoch 2 survives the crash.
    f.map.set("jax", new HookedJax(jaxPython!, async call => { if (call === 2) await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [job.run_id]); }));
    await f.worker("dead-worker").tick();
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);
    f.map.set("jax", new JaxBackend(jaxPython!));
    await f.drain("recovered-worker");
    const result = await f.training.getJob(manager, job.id);
    expect(result.attempts.map(a => a.status)).toEqual(["FAILED", "COMPLETED"]);
    expect(result.attempts[1].resumed_from_epoch).toBe(2);
    expect(sameMetrics(result.metrics, refResult.metrics)).toBe(true);
  }, 600_000);

  it("fences a worker that lost its lease: its JAX chunk result is never recorded", async () => {
    const f = await fixture(); const job = await f.job();
    f.map.set("jax", new HookedJax(jaxPython!, async call => { if (call === 1) { await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [job.run_id]); await new RunRepository(pool).recoverExpired(); } }));
    await f.worker("stale-worker").tick();
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints")).rows[0].n).toBe(0);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_epochs")).rows[0].n).toBe(0);
  }, 300_000);

  it("stops early on the validation loss and exports the best epoch", async () => {
    const f = await fixture(new JaxBackend(jaxPython!), noisyRows); const job = await f.job({ hyperparameters: { ...base.hyperparameters, hiddenLayers: [64, 64], batchSize: 8, epochs: 15, learningRate: 0.05, dropout: 0 }, training: { checkpointEveryEpochs: 5, earlyStopping: { patience: 2, minDelta: 0 } } });
    await f.drain();
    const result = await f.training.getJob(manager, job.id); const attempt = result.attempts[0];
    expect(result.job.status).toBe("COMPLETED");
    const history = await f.training.history(manager, job.id) as { epoch: number; validation_loss: number }[];
    expect(attempt.stop_reason).toBe("early_stopping");
    expect(history.length).toBeLessThan(15);
    // Early stopping forces one epoch per chunk, so every recorded epoch has its checkpoint.
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints WHERE training_run_id=$1", [attempt.id])).rows[0].n).toBe(history.length);
    expect(attempt.best_epoch).toBe(history.reduce((x, y) => (y.validation_loss < x.validation_loss ? y : x)).epoch);
  }, 600_000);

  it("registers, promotes and serves the JAX model through version-pinned inference, isolated by tenant", async () => {
    const f = await fixture(); const job = await f.job({ training: undefined }); await f.drain();
    const attempt = (await f.training.getJob(manager, job.id)).attempts[0];
    const registry = await f.models.registry(manager, { name: "jax-classifier" }); const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
    await f.models.transition(manager, version.id, "READY");
    const endpoint = await f.models.endpoint(manager, { name: "jax-live", modelVersionId: version.id });
    const prediction = await f.models.predict(manager, endpoint.id, [{ a: 2.1, b: 0.3 }, { a: -2.1, b: 0.3 }], new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toEqual(["yes", "no"]); expect(prediction.model_version_id).toBe(version.id);
    await expect(f.models.predict(outsider, endpoint.id, [{ a: 2.1, b: 0.3 }], new AbortController().signal)).rejects.toThrow(/not found/);
    await expect(f.training.getJob(outsider, job.id)).rejects.toThrow(/not found/);
    // A recorded runtime that differs is refused before the artifact is read; provenance drift alone is tolerated.
    const result = await f.training.getJob(manager, job.id); const bytes = await store.get(manager.tenantId, result.artifacts[0].content);
    await expect(f.map.get("jax")!.predict(result.job.snapshot, bytes, rows.slice(0, 2), { ...attempt.environment, jaxlib: "0.0.1" }, new AbortController().signal)).rejects.toThrow();
    expect(await f.map.get("jax")!.predict(result.job.snapshot, bytes, rows.slice(0, 2), { ...attempt.environment, adapter_sha256: "0".repeat(64) }, new AbortController().signal)).toHaveLength(2);
  }, 300_000);
});
