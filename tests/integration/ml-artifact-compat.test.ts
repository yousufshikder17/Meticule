import pg from "pg";
import { copyFile, appendFile, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";

const python = process.env.ML_TEST_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 6 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
let store: PostgresArtifactStore; let scripts: string;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_jobs LIMIT 1"); scripts = await mkdtemp(join(tmpdir(), "meticule-ml-adapter-")); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); if (scripts) await rm(scripts, { recursive: true, force: true }); });

const timeout = 120_000;
/** ml_blobs is append-only by trigger; the table owner disables it only to simulate storage corruption. */
async function tamper(key: string) {
  const c = await pool.connect();
  try { await c.query("ALTER TABLE ml_blobs DISABLE TRIGGER ml_blob_append_only"); await c.query("UPDATE ml_blobs SET content=$1 WHERE tenant_id=$2 AND content_hash=$3", [Buffer.from("tampered"), manager.tenantId, key]); }
  finally { await c.query("ALTER TABLE ml_blobs ENABLE TRIGGER ml_blob_append_only").catch(() => undefined); c.release(); }
}
const legacyScript = resolve("tests/python/legacy/sklearn_runner_phase1.py");
const sha = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");

/** Trains with `trainer`, registers and serves with `server`; returns what is needed to predict and inspect. */
async function serve(trainer: SklearnBackend, server: SklearnBackend) {
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, new Map([[server.id, server]]));
  const dataset = await training.ingest(manager, { name: "linear", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "y", type: "number" }], target: "y" }, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, y: 2 * i + 1 })) });
  const pipeline = await training.pipeline(manager, { name: "numeric", definition: { features: ["x"], steps: [{ operation: "impute_median" }, { operation: "standard_scale" }] } });
  const experiment = await training.experiment(manager, "compat");
  const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "sklearn", algorithm: "ridge", hyperparameters: { alpha: .01 }, seed: 42, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2 } } });
  await new LifecycleWorker(pool, { workerId: "compat", leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, new Map([[trainer.id, trainer]]))).tick();
  const result = await training.getJob(manager, job.id); expect(result.job.status).toBe("COMPLETED");
  const registry = await models.registry(manager, { name: "compat" }); const version = await models.register(manager, registry.id, { trainingRunId: result.attempts[0].id });
  await models.transition(manager, version.id, "READY"); const endpoint = await models.endpoint(manager, { name: "compat", modelVersionId: version.id });
  return { attempt: result.attempts[0], artifact: result.artifacts[0], endpoint, predict: () => models.predict(manager, endpoint.id, [{ x: 10 }], new AbortController().signal) };
}

describe.runIf(Boolean(python))("artifact compatibility through registry and inference", () => {
  it("keeps serving after an implementation-only adapter change and preserves the original checksum", async () => {
    const original = new SklearnBackend(python!); const edited = join(scripts, "sklearn_runner_edited.py");
    await copyFile(resolve("ml/sklearn_runner.py"), edited); await appendFile(edited, "\n# harmless implementation edit: changes the checksum, not the artifact contract\n");
    const drifted = new SklearnBackend(python!, 120_000, edited);
    const s = await serve(original, drifted);
    expect(s.attempt.environment.adapter_sha256).toBe(await sha(resolve("ml/sklearn_runner.py"))); expect(await sha(edited)).not.toBe(s.attempt.environment.adapter_sha256);
    expect(s.attempt.environment).toMatchObject({ backend: "sklearn", artifact_format: "sklearn-pickle-v1", artifact_schema: "1", protocol: "meticule-sklearn-v2" });
    const prediction = await s.predict(); expect(prediction.status).toBe("COMPLETED"); expect(prediction.output[0]).toBeCloseTo(21, 0);
    expect((await pool.query("SELECT environment FROM ml_training_runs WHERE id=$1", [s.attempt.id])).rows[0].environment.adapter_sha256).toBe(s.attempt.environment.adapter_sha256);
  }, timeout);
  it("serves a model trained and registered under the unmodified Phase 1 adapter, retaining its evidence", async () => {
    const s = await serve(new SklearnBackend(python!, 120_000, legacyScript), new SklearnBackend(python!));
    expect(s.attempt.environment.protocol).toBe("meticule-sklearn-v1"); expect(s.attempt.environment).not.toHaveProperty("artifact_schema");
    expect(s.attempt.environment.adapter_sha256).toBe(await sha(legacyScript));
    const prediction = await s.predict(); expect(prediction.status).toBe("COMPLETED"); expect(prediction.output[0]).toBeCloseTo(21, 0);
  }, timeout);
  it("still refuses artifacts when integrity, format or runtime guarantees fail", async () => {
    const backend = new SklearnBackend(python!); const s = await serve(backend, backend);
    expect((await s.predict()).status).toBe("COMPLETED");
    // Unsupported stored format: refused before the bytes are read.
    const strict = Object.assign(Object.create(backend) as SklearnBackend, { artifactFormats: ["something-else"] });
    const strictModels = new MlModelService(pool, store, new Map([["sklearn", strict]]));
    const wrongFormat = await strictModels.predict(manager, s.endpoint.id, [{ x: 10 }], new AbortController().signal);
    expect(wrongFormat.status).toBe("FAILED"); expect(JSON.stringify(wrongFormat)).not.toMatch(/pickle|something-else|Traceback/);
    // Runtime mismatch recorded for the artifact: refused by the contract.
    const mismatch = await backend.predict(s.attempt.snapshot, await store.get(manager.tenantId, s.artifact.content), [{ x: 10 }], { ...s.attempt.environment, sklearn: "0.0.1" }, new AbortController().signal).then(() => "accepted", () => "refused");
    expect(mismatch).toBe("refused");
    // Corrupted bytes: the content hash check fails before anything is unpickled.
    await tamper(s.artifact.content.key);
    const corrupted = await s.predict(); expect(corrupted.status).toBe("FAILED"); expect(corrupted.failure.message).not.toMatch(/tampered|checksum/);
  }, timeout);
});
