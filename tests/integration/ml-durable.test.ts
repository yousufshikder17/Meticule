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
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import type { ChunkResult, MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import type { Row, Snapshot } from "../../src/ml/domain.js";

const sklearnPython = process.env.ML_TEST_PYTHON;
const torchPython = process.env.ML_TEST_PYTORCH_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_checkpoints LIMIT 1"); store = new PostgresArtifactStore(pool); });
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

const columns = [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "label", type: "string" }];
const rows = Array.from({ length: 120 }, (_, i) => ({ a: (i % 2 ? 2 : -2) + (i % 7) * 0.05, b: (i * 13) % 11 / 10, label: i % 2 ? "yes" : "no" }));
const base = { algorithm: "mlp_classifier", hyperparameters: { hiddenLayers: [8], epochs: 6, batchSize: 16, learningRate: 0.02, dropout: 0 }, seed: 17, evaluation: { metrics: ["accuracy"] } };

/** A backend whose hook runs after each chunk. Used to inject the exact failures the durable path must survive. */
class HookedBackend extends PytorchBackend {
  private calls = 0;
  constructor(python: string, private readonly hook?: (call: number) => Promise<void> | void) { super(python); }
  override async trainChunk(snapshot: Snapshot, rowsIn: Row[], prepared: PreparedData, state: Buffer | null, epochs: number, signal: AbortSignal): Promise<ChunkResult> {
    const result = await super.trainChunk(snapshot, rowsIn, prepared, state, epochs, signal);
    const call = ++this.calls;
    await this.hook?.(call);
    return result;
  }
}

async function fixture(backend: MlBackend, spec: Record<string, unknown> = {}) {
  const map = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "durable", schema: { id: randomUUID(), columns, target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: "durable", definition: { features: ["a", "b"], steps: [{ operation: "standard_scale" }] } });
  const experiment = await training.experiment(manager, "durable");
  const job = (extra: Record<string, unknown> = {}) => training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id,
    ...base, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, training: { checkpointEveryEpochs: 2, earlyStopping: null }, ...spec, ...extra } });
  const worker = (id: string) => new LifecycleWorker(pool, { workerId: id, leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const drain = async (id = "durable-main") => { const w = worker(id); while (await w.tick()); };
  return { training, models, dataset, pipeline, experiment, job, worker, drain, map };
}

/** Simulates a worker that died: its lease is expired, and the canonical recovery requeues the job. */
async function killLease(runId: string) {
  await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [runId]);
  expect(await new RunRepository(pool).recoverExpired()).toBe(1);
}

describe.runIf(Boolean(torchPython))("durable deep-learning training (Phase 3B)", () => {
  it("persists bounded epoch summaries and checkpoints for an uninterrupted run", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job(); await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    const history = await f.training.history(manager, job.id);
    expect(history.map((e: { epoch: number }) => e.epoch)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(history.every((e: { train_loss: number }) => Number.isFinite(e.train_loss))).toBe(true);
    const checkpoints = await pool.query("SELECT epoch,format FROM ml_training_checkpoints ORDER BY epoch");
    expect(checkpoints.rows.map(r => r.epoch)).toEqual([2, 4, 6]);
    expect(new Set(checkpoints.rows.map(r => r.format))).toEqual(new Set(["pytorch-checkpoint-v1"]));
    expect(result.attempts[0].resumed_from_epoch).toBeNull(); expect(result.attempts[0].stop_reason).toBe("completed");
  }, 180_000);

  it("resumes an interrupted attempt from its checkpoint and matches an uninterrupted run exactly", async () => {
    const reference = await fixture(new PytorchBackend(torchPython!)); const refJob = await reference.job(); await reference.drain();
    const refResult = await reference.training.getJob(manager, refJob.id);

    // The worker's second chunk completes, then its lease expires before the checkpoint can be written.
    // The first chunk's checkpoint is durable; the second is lost with the dead worker, as a real crash would lose it.
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job();
    f.map.set("pytorch", new HookedBackend(torchPython!, async call => {
      if (call === 2) await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [job.run_id]);
    }));
    await f.worker("dead-worker").tick();
    expect((await f.training.getJob(manager, job.id)).job.status).toBe("TRAINING");
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);
    expect((await f.training.getJob(manager, job.id)).job.status).toBe("QUEUED");
    f.map.set("pytorch", new PytorchBackend(torchPython!));
    await f.drain("durable-recovered");

    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts.map(a => a.status)).toEqual(["FAILED", "COMPLETED"]);
    expect(result.attempts[1].resumed_from_epoch).toBe(2);
    expect(result.attempts[1].split_indices).toEqual(result.attempts[0].split_indices);
    const history = await f.training.history(manager, job.id) as { epoch: number }[];
    expect(history.map(e => e.epoch)).toEqual([1, 2, 3, 4, 5, 6]);
    // Chunked and resumed training must reproduce the uninterrupted run: identical metrics for every partition.
    const pairs = result.metrics.map((m: { name: string; partition: string; value: number }) => [m.value, refResult.metrics.find((r: { name: string; partition: string }) => r.name === m.name && r.partition === m.partition)!.value]);
    expect(pairs.length).toBeGreaterThan(0);
    expect(pairs.every(([a, b]) => Math.abs(a - b) < 1e-9)).toBe(true);
  }, 300_000);

  it("fences a stale worker: it cannot write checkpoints or progress after losing ownership", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); await f.job();
    const runs = new RunRepository(pool);
    const stale = (await runs.claimNext("old-owner", 60, "ml"))!; await runs.workerTransition(stale.id, "old-owner", "running");
    const attempt = await f.training.begin(stale.id, "old-owner");
    await f.training.phase(stale.id, "old-owner", attempt.id, "TRAINING");
    await killLease(stale.id);
    const dummy = { epoch: 1, content: { key: "a".repeat(64), sha256: "a".repeat(64), size: 1, mediaType: "application/octet-stream" }, format: "pytorch-checkpoint-v1", schemaVersion: 1 };
    await expect(f.training.checkpoint(stale.id, "old-owner", attempt.id, [{ epoch: 1, trainLoss: 0.5, validationLoss: 0.5, elapsedSeconds: 0.1 }], dummy)).rejects.toThrow(/does not own|lease/i);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_epochs WHERE training_run_id=$1", [attempt.id])).rows[0].n).toBe(0);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints WHERE training_run_id=$1", [attempt.id])).rows[0].n).toBe(0);
  }, 120_000);

  it("fences a worker whose lease expired before recovery ran, while its attempt is still TRAINING", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); await f.job();
    const runs = new RunRepository(pool);
    const stale = (await runs.claimNext("expired-owner", 60, "ml"))!; await runs.workerTransition(stale.id, "expired-owner", "running");
    const attempt = await f.training.begin(stale.id, "expired-owner");
    await f.training.phase(stale.id, "expired-owner", attempt.id, "TRAINING");
    // Only the lease has expired. Recovery has not requeued the run, so the database trigger would still accept progress.
    await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [stale.id]);
    expect((await pool.query("SELECT status FROM ml_training_runs WHERE id=$1", [attempt.id])).rows[0].status).toBe("TRAINING");
    const dummy = { epoch: 1, content: { key: "a".repeat(64), sha256: "a".repeat(64), size: 1, mediaType: "application/octet-stream" }, format: "pytorch-checkpoint-v1", schemaVersion: 1 };
    await expect(f.training.checkpoint(stale.id, "expired-owner", attempt.id, [{ epoch: 1, trainLoss: 0.5, validationLoss: 0.5, elapsedSeconds: 0.1 }], dummy)).rejects.toThrow(/does not own/);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints WHERE training_run_id=$1", [attempt.id])).rows[0].n).toBe(0);
  }, 120_000);

  it("refuses to resume from a corrupted checkpoint and fails the job explicitly instead of restarting", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job();
    f.map.set("pytorch", new HookedBackend(torchPython!, async call => {
      if (call === 2) await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [job.run_id]);
    }));
    await f.worker("dead-worker").tick();
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);
    const checkpoint = (await pool.query("SELECT content FROM ml_training_checkpoints ORDER BY epoch DESC LIMIT 1")).rows[0].content;
    await corruptBlob(manager.tenantId, checkpoint.key, "corrupted checkpoint");
    f.map.set("pytorch", new PytorchBackend(torchPython!));
    await f.drain("durable-recovered");
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED");
    expect(result.attempts[1].failure.code).toBe("ml_training_failed");
    expect(result.attempts[1].failure.category).toBe("invalid_checkpoint");
    expect(result.attempts[1].resumed_from_epoch).toBe(2);
    expect(result.metrics).toHaveLength(0); expect(result.artifacts).toHaveLength(0);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints WHERE training_run_id=$1", [result.attempts[1].id])).rows[0].n).toBe(0);
  }, 240_000);

  it("refuses to resume a checkpoint whose format the backend does not declare, before reading its bytes", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job();
    f.map.set("pytorch", new HookedBackend(torchPython!, async call => {
      if (call === 2) await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [job.run_id]);
    }));
    await f.worker("dead-worker").tick();
    expect(await new RunRepository(pool).recoverExpired()).toBe(1);
    // The resuming backend declares only a newer checkpoint format, as after an incompatible checkpoint upgrade.
    const upgraded = new PytorchBackend(torchPython!);
    f.map.set("pytorch", Object.assign(upgraded, { capabilities: { ...upgraded.capabilities, checkpointFormats: ["pytorch-checkpoint-v2"] } }));
    await f.drain("durable-recovered");
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED");
    expect(result.attempts[1].failure.category).toBe("incompatible_checkpoint");
    expect(result.artifacts).toHaveLength(0);
  }, 240_000);

  it("cancels between chunks; the saved checkpoint stays, and the run is never promoted to completed", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job();
    f.map.set("pytorch", new HookedBackend(torchPython!, async call => {
      if (call === 2) await new RunRepository(pool).requestCancellation(manager, job.run_id);
    }));
    await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("CANCELLED");
    expect(result.artifacts).toHaveLength(0);
    expect((await pool.query("SELECT epoch FROM ml_training_checkpoints ch JOIN ml_training_runs t ON t.id=ch.training_run_id WHERE t.job_id=$1 ORDER BY epoch", [job.id])).rows.map(r => r.epoch)).toEqual([2]);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_runs WHERE job_id=$1 AND status='COMPLETED'", [job.id])).rows[0].n).toBe(0);
  }, 180_000);

  it("stops early on validation loss, restores the best epoch, and records the reason", async () => {
    const f = await fixture(new PytorchBackend(torchPython!));
    const job = await f.job({ hyperparameters: { ...base.hyperparameters, epochs: 12, learningRate: 0.5, dropout: 0 }, training: { checkpointEveryEpochs: 5, earlyStopping: { patience: 2, minDelta: 0 } } });
    await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("COMPLETED");
    const attempt = result.attempts[0];
    expect(attempt.stop_reason).toBe("early_stopping");
    const history = await f.training.history(manager, job.id) as { epoch: number; validation_loss: number }[];
    expect(history.length).toBeLessThan(12);
    const best = history.reduce((x, y) => (y.validation_loss < x.validation_loss ? y : x));
    expect(attempt.best_epoch).toBe(best.epoch);
    expect(attempt.best_epoch).toBeLessThan(history.length);
    // Early stopping forces one epoch per chunk, so there is one checkpoint per recorded epoch.
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_checkpoints WHERE training_run_id=$1", [attempt.id])).rows[0].n).toBe(history.length);
  }, 240_000);

  it("keeps history and checkpoints tenant-isolated and rejects unsupported training policies", async () => {
    const f = await fixture(new PytorchBackend(torchPython!)); const job = await f.job(); await f.drain();
    expect(await f.training.history(outsider, job.id)).toEqual([]);
    await expect(f.training.getJob(outsider, job.id)).rejects.toThrow(/not found/);
    await expect(f.job({ training: { checkpointEveryEpochs: 0, earlyStopping: null } })).rejects.toThrow();
    await expect(f.job({ training: { checkpointEveryEpochs: 2, earlyStopping: { patience: 0, minDelta: 0 } } })).rejects.toThrow();
  }, 180_000);
});

describe.runIf(Boolean(sklearnPython))("training policy is rejected for non-iterative models", () => {
  it("fails a sklearn job that carries a checkpoint or early-stopping policy, explicitly", async () => {
    const f = await fixture(new SklearnBackend(sklearnPython!));
    const job = await f.job({ backend: "sklearn", algorithm: "logistic_regression", hyperparameters: {}, training: { checkpointEveryEpochs: 2, earlyStopping: null } });
    await f.drain();
    const result = await f.training.getJob(manager, job.id);
    expect(result.job.status).toBe("FAILED");
    expect(result.attempts[0].failure.message).toMatch(/iterative/);
  }, 60_000);
});
