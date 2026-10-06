import pg from "pg";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { SignJWT } from "jose";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { RunRepository } from "../../src/db/repositories.js";
import { withTenantSession } from "../../src/db/tenant-session.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { MlModelService } from "../../src/ml/model-service.js";
import type { MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import type { MetricValue, Row, Snapshot } from "../../src/ml/domain.js";
import { createApp } from "../../src/api/app.js";
import { JwtAuthenticator } from "../../src/auth/authentication.js";

const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
class TestBackend implements MlBackend {
  readonly id = "test-ml";
  validate(): void {}
  async prepare(_s: Snapshot, rows: Row[]): Promise<PreparedData> { return { indices: { train: rows.map((_, i) => i).slice(0, 36), validation: rows.map((_, i) => i).slice(36, 48), test: rows.map((_, i) => i).slice(48) }, environment: { protocol: "test" } }; }
  async train(): Promise<TrainedModel> { return { bytes: Buffer.from("test-model"), format: "test-v1", environment: { protocol: "test" }, resolvedHyperparameters: { alpha: 1 } }; }
  async evaluate(): Promise<MetricValue[]> { return ["train", "validation", "test"].map(partition => ({ name: "mae", partition: partition as MetricValue["partition"], value: .1, direction: "lower" })); }
  async predict(_s: Snapshot, _a: Buffer, rows: Row[]): Promise<unknown[]> { return rows.map(r => 2 * Number(r.x) + 1); }
}
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_training_jobs LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE revoked_tokens,tenant_memberships,ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

async function fixture(backend: MlBackend = new TestBackend()) {
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, new Map([[backend.id, backend]]));
  const dataset = await training.ingest(manager, { name: "linear", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "y", type: "number" }], target: "y" }, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, y: 2 * i + 1 })) });
  const pipeline = await training.pipeline(manager, { name: "numeric", definition: { features: ["x"], steps: [{ operation: "standard_scale" }] } });
  const experiment = await training.experiment(manager, "regression");
  const input = { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id, algorithm: "ridge", hyperparameters: { alpha: .01 }, seed: 42, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2 } } };
  const job = await training.queue(manager, input);
  const worker = new LifecycleWorker(pool, { workerId: "ml-test", leaseSeconds: 30, kind: "ml" }, new TrainingProcessor(pool, store, new Map([[backend.id, backend]])));
  return { training, models, dataset, pipeline, experiment, input, job, worker, backend };
}

describe("durable ML subsystem", () => {
  it("freezes versions, isolates workloads and completes through shared runtime/usage/audit", async () => {
    const f = await fixture();
    const revision = await f.training.ingest(manager, { name: "linear v2", schema: f.dataset.schema, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, y: 3 * i + 1 })) }, f.dataset.datasetId);
    expect(revision.version).toBe(2); expect(revision.content.sha256).not.toBe(f.dataset.content.sha256);
    expect((await f.training.getJob(manager, f.job.id)).job.snapshot.dataset.id).toBe(f.dataset.id);
    await f.training.pipeline(manager, { name: "changed", definition: { features: ["x"], steps: [] } }, f.pipeline.logicalId);
    expect((await f.training.getJob(manager, f.job.id)).job.snapshot.pipeline.id).toBe(f.pipeline.id);
    expect(await new RunRepository(pool).claimNext("agent-worker", 30)).toBeNull();
    expect(await f.worker.tick()).toBe(true);
    const result = await new MlTrainingService(pool, store).getJob(manager, f.job.id);
    expect(result.job.runtime_status).toBe("completed"); expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts).toHaveLength(1); expect(result.attempts[0].ended_at).not.toBeNull();
    expect(result.attempts[0].split_indices.train).toHaveLength(36); expect(result.metrics).toHaveLength(3); expect(result.artifacts).toHaveLength(1);
    expect((await pool.query("SELECT usage_kind,duration_ms FROM usage_records WHERE run_id=$1", [f.job.run_id])).rows[0].usage_kind).toBe("ml_training");
    expect((await pool.query("SELECT 1 FROM worker_leases WHERE run_id=$1", [f.job.run_id])).rowCount).toBe(0);
    expect((await pool.query("SELECT event_type FROM audit_events WHERE run_id=$1 AND event_type='ml.training_completed'", [f.job.run_id])).rowCount).toBe(1);
    expect((await pool.query("SELECT actor_type FROM audit_events WHERE run_id=$1 AND event_type='ml.training_completed'", [f.job.run_id])).rows[0].actor_type).toBe("worker");
  });
  it("enforces state, provenance and tenant ownership in PostgreSQL", async () => {
    const f = await fixture();
    await expect(pool.query("UPDATE ml_training_jobs SET status='COMPLETED' WHERE id=$1", [f.job.id])).rejects.toThrow(/invalid ML training transition/);
    await expect(pool.query("UPDATE ml_dataset_versions SET document='{}' WHERE id=$1", [f.dataset.id])).rejects.toThrow();
    await expect(pool.query("UPDATE ml_training_jobs SET snapshot='{}' WHERE id=$1", [f.job.id])).rejects.toThrow(/immutable/);
    await expect(f.training.ingest(manager, { name: "changed schema", schema: { ...f.dataset.schema, columns: [{ name: "x", type: "string" }, { name: "y", type: "number" }] }, rows: [{ x: "changed", y: 1 }] }, f.dataset.datasetId)).rejects.toThrow(/different definition/);
    await expect(f.training.getJob(outsider, f.job.id)).rejects.toThrow(/not found/);
    await expect(f.training.queue(outsider, f.input)).rejects.toThrow(/not found/);
    await withTenantSession(pool, outsider.tenantId, async db => {
      expect((await db.query("SELECT * FROM ml_training_jobs")).rowCount).toBe(0);
      expect((await db.query("SELECT * FROM ml_dataset_versions")).rowCount).toBe(0);
      await expect(new MlTrainingService(db, store).getJob(outsider, f.job.id)).rejects.toThrow();
    });
    await expect(f.training.experiment({ ...manager, roles: [] }, "unauthorized")).rejects.toThrow(/role/);
  });
  it("uses canonical cancellation and synchronizes queued ML evidence under API RLS", async () => {
    const f = await fixture();
    await withTenantSession(pool, manager.tenantId, db => new RunRepository(db).requestCancellation(manager, f.job.run_id));
    await new RunRepository(pool).finalizeUnleasedCancellations();
    expect((await f.training.getJob(manager, f.job.id)).job.status).toBe("CANCELLED"); expect(await f.worker.tick()).toBe(false);
  });
  it("closes expired attempts and restarts with frozen provenance; fences old workers", async () => {
    const f = await fixture(); const runs = new RunRepository(pool);
    await runs.claimNext("dead", 30, "ml"); await runs.workerTransition(f.job.run_id, "dead", "running");
    const first = await f.training.begin(f.job.run_id, "dead"); await f.training.phase(f.job.run_id, "dead", first.id, "TRAINING");
    await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [f.job.run_id]);
    expect(await runs.recoverExpired()).toBe(1);
    await expect(f.training.phase(f.job.run_id, "dead", first.id, "EVALUATING")).rejects.toThrow(/lease/);
    expect(await f.worker.tick()).toBe(true);
    const result = await f.training.getJob(manager, f.job.id);
    expect(result.attempts.map(t => t.status)).toEqual(["FAILED", "COMPLETED"]);
    expect(result.attempts[0].failure.code).toBe("lease_lost"); expect(result.attempts[0].snapshot).toEqual(result.attempts[1].snapshot);
    await expect(pool.query("UPDATE ml_training_runs SET environment='{}' WHERE id=$1", [result.attempts[1].id])).rejects.toThrow(/immutable/);
    await expect(pool.query("INSERT INTO ml_metrics(tenant_id,training_run_id,name,partition,value,direction) VALUES($1,$2,'late','test',1,'lower')", [manager.tenantId, result.attempts[1].id])).rejects.toThrow(/saving attempt/);
  });
  it("cancels in-flight training without publishing metrics or artifacts", async () => {
    const backend = new TestBackend(); let started!: () => void; const trainingStarted = new Promise<void>(r => { started = r; });
    backend.train = async (...args: unknown[]) => { const signal = args[3] as AbortSignal; started(); return new Promise<TrainedModel>((_resolve, reject) => signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true })); };
    const f = await fixture(backend); const tick = f.worker.tick(); await trainingStarted;
    await new RunRepository(pool).requestCancellation(manager, f.job.run_id); await tick;
    const result = await f.training.getJob(manager, f.job.id); expect(result.job.status).toBe("CANCELLED"); expect(result.attempts[0].status).toBe("CANCELLED"); expect(result.artifacts).toHaveLength(0);
  });
  it("fences stale failure handling when the same worker ID reclaims a job", async () => {
    const f = await fixture(); const runs = new RunRepository(pool);
    await runs.claimNext("same-worker", 30, "ml"); await runs.workerTransition(f.job.run_id, "same-worker", "running");
    const abandoned = await f.training.begin(f.job.run_id, "same-worker");
    await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [f.job.run_id]); await runs.recoverExpired();
    await runs.claimNext("same-worker", 30, "ml"); await runs.workerTransition(f.job.run_id, "same-worker", "running");
    const current = await f.training.begin(f.job.run_id, "same-worker");
    await f.training.fail(f.job.run_id, "same-worker", abandoned.id, "stale failure");
    await expect(f.training.phase(f.job.run_id, "same-worker", abandoned.id, "TRAINING")).rejects.toThrow();
    const result = await f.training.getJob(manager, f.job.id); expect(result.job.runtime_status).toBe("running"); expect(result.attempts[1].id).toBe(current.id); expect(result.attempts[1].status).toBe("PREPARING");
    await f.training.fail(f.job.run_id, "same-worker", current.id, "current failure");
    expect((await f.training.getJob(manager, f.job.id)).job.status).toBe("FAILED");
  });
  it("deduplicates durable artifacts, isolates tenants and verifies load integrity", async () => {
    const reference = await store.put(manager.tenantId, Buffer.from("model-bytes"), "application/octet-stream");
    expect(await store.put(manager.tenantId, Buffer.from("model-bytes"), reference.mediaType)).toEqual(reference);
    expect((await new PostgresArtifactStore(pool).get(manager.tenantId, reference)).toString()).toBe("model-bytes");
    expect((await pool.query("SELECT 1 FROM ml_blobs WHERE tenant_id=$1", [manager.tenantId])).rowCount).toBe(1);
    await expect(store.get(outsider.tenantId, reference)).rejects.toThrow(/not found/);
    await expect(store.get(manager.tenantId, { ...reference, size: 999 })).rejects.toThrow(/checksum/);
    await expect(pool.query("UPDATE ml_blobs SET content=$1 WHERE tenant_id=$2", [Buffer.from("tampered"), manager.tenantId])).rejects.toThrow(/immutable|append/);
    await withTenantSession(pool, outsider.tenantId, async db => { expect((await db.query("SELECT * FROM ml_blobs")).rowCount).toBe(0); });
  });
  it("rolls back finalization on invalid metric evidence", async () => {
    const backend = new TestBackend(); const evaluate = backend.evaluate.bind(backend);
    backend.evaluate = async () => { const metrics = await evaluate(); return [...metrics, metrics[0]!]; };
    const f = await fixture(backend); await f.worker.tick(); const result = await f.training.getJob(manager, f.job.id);
    expect(result.job.status).toBe("FAILED"); expect(result.attempts[0].failure.code).toBe("ml_training_failed"); expect(result.metrics).toHaveLength(0); expect(result.artifacts).toHaveLength(0);
  });
  it("registers, promotes, predicts and retires version-pinned models", async () => {
    const f = await fixture(); await f.worker.tick(); const attempt = (await f.training.getJob(manager, f.job.id)).attempts[0];
    const registry = await withTenantSession(pool, manager.tenantId, db => new MlModelService(db, new PostgresArtifactStore(db), new Map()).registry(manager, { name: "linear" }));
    const version = await f.models.register(manager, registry.id, { trainingRunId: attempt.id });
    expect((await f.models.register(manager, registry.id, { trainingRunId: attempt.id })).id).toBe(version.id);
    await expect(f.models.endpoint(manager, { name: "not-ready", modelVersionId: version.id })).rejects.toThrow(/READY/);
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "prediction", modelVersionId: version.id });
    const prediction = await withTenantSession(pool, manager.tenantId, db => new MlModelService(db, store, new Map([[f.backend.id, f.backend]])).predict(manager, endpoint.id, [{ x: 10 }], new AbortController().signal));
    expect(prediction.output).toEqual([21]); expect(prediction.model_version_id).toBe(version.id);
    const nextJob = await f.training.queue(manager, f.input); await f.worker.tick();
    const nextVersion = await f.models.register(manager, registry.id, { trainingRunId: (await f.training.getJob(manager, nextJob.id)).attempts[0].id });
    expect(nextVersion.version).toBe(2); expect(nextVersion.status).toBe("REGISTERED");
    expect((await pool.query("SELECT model_version_id FROM ml_endpoints WHERE id=$1", [endpoint.id])).rows[0].model_version_id).toBe(version.id);
    await expect(pool.query("UPDATE ml_predictions SET output='[999]' WHERE id=$1", [prediction.id])).rejects.toThrow(/immutable/);
    await expect(f.models.predict(outsider, endpoint.id, [{ x: 10 }], new AbortController().signal)).rejects.toThrow(/not found/);
    await f.models.transition(manager, version.id, "RETIRED");
    await expect(f.models.predict(manager, endpoint.id, [{ x: 10 }], new AbortController().signal)).rejects.toThrow(/not found/);
    await expect(f.models.transition(manager, version.id, "READY")).rejects.toThrow();
    await expect(pool.query("UPDATE ml_model_versions SET status='READY' WHERE id=$1", [version.id])).rejects.toThrow(/invalid ML model transition/);
  });
  it("compares only reproducible cohorts and records inference failures", async () => {
    const f = await fixture(); await f.worker.tick(); const first = (await f.training.getJob(manager, f.job.id)).attempts[0];
    const second = await f.training.queue(manager, f.input); await f.worker.tick(); const other = (await f.training.getJob(manager, second.id)).attempts[0];
    expect((await f.models.compare(manager, first.id, other.id)).every(m => !m.regressed)).toBe(true);
    const changed = await f.training.queue(manager, { ...f.input, spec: { ...f.input.spec, seed: 123 } }); await f.worker.tick();
    await expect(f.models.compare(manager, first.id, (await f.training.getJob(manager, changed.id)).attempts[0].id)).rejects.toThrow(/matching/);
    const prepare = f.backend.prepare.bind(f.backend);
    f.backend.prepare = async (s, rows, signal) => { const p = await prepare(s, rows, signal); return { ...p, indices: { ...p.indices, validation: p.indices.test, test: p.indices.validation } }; };
    const differentSplit = await f.training.queue(manager, f.input); await f.worker.tick();
    await expect(f.models.compare(manager, first.id, (await f.training.getJob(manager, differentSplit.id)).attempts[0].id)).rejects.toThrow(/matching/);
    const registry = await f.models.registry(manager, { name: "fail-inference" }); const version = await f.models.register(manager, registry.id, { trainingRunId: first.id }); await f.models.transition(manager, version.id, "READY");
    const endpoint = await f.models.endpoint(manager, { name: "bad", modelVersionId: version.id }); f.backend.predict = async () => { throw new Error("secret data"); };
    const failed = await f.models.predict(manager, endpoint.id, [{ x: 1 }], new AbortController().signal);
    expect(failed.status).toBe("FAILED"); expect(JSON.stringify(failed)).not.toContain("secret data");
  });
  it("rejects invalid inference schemas before creating prediction records", async () => {
    const f = await fixture(); await f.worker.tick(); const attempt = (await f.training.getJob(manager, f.job.id)).attempts[0];
    const registry = await f.models.registry(manager, { name: "schema-check" }); const model = await f.models.register(manager, registry.id, { trainingRunId: attempt.id }); await f.models.transition(manager, model.id, "READY");
    const endpoint = await f.models.endpoint(manager, { name: "schema", modelVersionId: model.id });
    for (const rows of [[{ x: "bad" }], [{}], [{ x: 1, y: 3 }], [{ x: null }], [{ x: Infinity }]]) await expect(f.models.predict(manager, endpoint.id, rows, new AbortController().signal)).rejects.toThrow();
    expect((await pool.query("SELECT 1 FROM ml_predictions WHERE endpoint_id=$1", [endpoint.id])).rowCount).toBe(0);
  });
  it("uses public JWT membership/RLS for APIs and performs no training inline", async () => {
    const f = await fixture(); const secret = "public-ml-test-only-secret-at-least-32-characters", issuer = "public-ml-test", audience = "public-ml-api";
    for (const p of [manager, outsider]) await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,'user',$3)", [p.tenantId, p.userId, JSON.stringify(p.roles)]);
    const jwt = async (p: Principal, roles = p.roles) => new SignJWT({ tenant_id: p.tenantId, identity_type: "user", roles }).setProtectedHeader({ alg: "HS256" }).setSubject(p.userId).setIssuer(issuer).setAudience(audience).setJti(randomUUID()).setIssuedAt().setExpirationTime("5m").sign(new TextEncoder().encode(secret));
    const app = createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience), null, null, null, undefined, null, { enabled: true, backends: new Map([[f.backend.id, f.backend]]) });
    const request = async (path: string, token: string, body?: unknown) => app.request(path, { method: body === undefined ? "GET" : "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    expect((await app.request("/ml/datasets")).status).toBe(401);
    const admin = await jwt(manager), member = await jwt(manager, []), other = await jwt(outsider);
    expect((await request("/ml/training-jobs", member, f.input)).status).toBe(403);
    expect((await request(`/ml/training-jobs/${f.job.id}`, other)).status).toBe(404);
    expect((await request("/ml/datasets", member)).status).toBe(200);
    const ingested = await request("/ml/datasets", admin, { name: "http-dataset", schema: { ...f.dataset.schema, id: randomUUID() }, rows: [{ x: 1, y: 3 }] }); expect(ingested.status).toBe(201);
    expect((await request(`/ml/datasets/${f.dataset.datasetId}/versions`, admin, { name: "http-revision", schema: f.dataset.schema, rows: [{ x: 2, y: 5 }] })).status).toBe(201);
    expect((await request(`/ml/pipelines/${f.pipeline.logicalId}/versions`, admin, { name: "http-pipeline", definition: { features: ["x"], steps: [] } })).status).toBe(201);
    const response = await request("/ml/training-jobs", admin, f.input); expect(response.status).toBe(202); expect((await response.json()).status).toBe("QUEUED");
    expect((await pool.query("SELECT 1 FROM ml_training_runs")).rowCount).toBe(0);
    await f.worker.tick(); const attempt = (await f.training.getJob(manager, f.job.id)).attempts[0];
    const registryResponse = await request("/ml/registry", admin, { name: "http-model" }); expect(registryResponse.status).toBe(201); const registry = await registryResponse.json();
    const versionResponse = await request(`/ml/registry/${registry.id}/versions`, admin, { trainingRunId: attempt.id }); expect(versionResponse.status).toBe(201); const version = await versionResponse.json();
    expect((await request(`/ml/model-versions/${version.id}/transition`, admin, { status: "READY" })).status).toBe(200);
    const endpointResponse = await request("/ml/endpoints", admin, { name: "http", modelVersionId: version.id }); expect(endpointResponse.status).toBe(201); const endpoint = await endpointResponse.json();
    expect((await request(`/ml/endpoints/${endpoint.id}/predict`, member, { rows: [{ x: "bad" }] })).status).toBe(409);
    const prediction = await request(`/ml/endpoints/${endpoint.id}/predict`, member, { rows: [{ x: 10 }] }); expect(prediction.status).toBe(200); expect((await prediction.json()).output).toEqual([21]);
    const records = await request(`/ml/endpoints/${endpoint.id}/predictions`, member); expect((await records.json()).length).toBe(1);
    const disabled = createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience), null, null, null, undefined, null, { enabled: false, backends: new Map() });
    expect((await disabled.request("/ml/datasets", { headers: { authorization: `Bearer ${admin}` } })).status).toBe(503);
    f.backend.predict = async () => { throw new Error("internal details"); };
    expect((await request(`/ml/endpoints/${endpoint.id}/predict`, member, { rows: [{ x: 1 }] })).status).toBe(422);
    expect((await pool.query("SELECT 1 FROM ml_predictions WHERE status='FAILED'")).rowCount).toBe(1);
  });
  it("rejects the unreleased precursor migration marker without overwriting data", async () => {
    const f = await fixture(); const client = await pool.connect();
    try {
      await client.query("BEGIN"); await client.query("INSERT INTO schema_migrations(version) VALUES('002_ml')");
      await expect(client.query(await readFile("migrations/002_ml_foundation.sql", "utf8"))).rejects.toThrow(/precursor detected/);
    } finally { await client.query("ROLLBACK"); client.release(); }
    expect((await f.training.getJob(manager, f.job.id)).job.status).toBe("QUEUED");
    expect((await pool.query("SELECT 1 FROM schema_migrations WHERE version='002_ml'")).rowCount).toBe(0);
  });
  it.runIf(Boolean(process.env.ML_TEST_PYTHON))("executes real sklearn through the PostgreSQL lifecycle and registry", async () => {
    const f = await fixture(new SklearnBackend(process.env.ML_TEST_PYTHON!)); await f.worker.tick();
    const result = await f.training.getJob(manager, f.job.id); expect(result.job.status).toBe("COMPLETED"); expect(result.metrics).toHaveLength(6); expect(result.attempts[0].environment.sklearn).toBe("1.7.2");
    const registry = await f.models.registry(manager, { name: "real-sklearn" }); const model = await f.models.register(manager, registry.id, { trainingRunId: result.attempts[0].id }); await f.models.transition(manager, model.id, "READY");
    const endpoint = await f.models.endpoint(manager, { name: "real", modelVersionId: model.id });
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: 10 }], new AbortController().signal); expect(prediction.status).toBe("COMPLETED"); expect(prediction.output[0]).toBeCloseTo(21, 0);
  }, 60_000);
});
