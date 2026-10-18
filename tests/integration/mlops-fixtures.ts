// Shared fixtures for the Phase 5 MLOps integration tests: a deterministic in-process backend and a trained model.
import pg from "pg";
import { Hono } from "hono";
import { randomUUID } from "node:crypto";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { MlModelService } from "../../src/ml/model-service.js";
import type { MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import { defineModel, registerModels } from "../../src/ml/model-catalog.js";
import { registerMlRoutes } from "../../src/ml/routes.js";
import type { MetricValue, Row, Snapshot } from "../../src/ml/domain.js";

export const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
export const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
export const approver: Principal = { ...manager, userId: "44444444-4444-4444-8444-444444444444" };
// Public roles: writes need tenant_admin; any member of the tenant reads.
export const viewer: Principal = { ...manager, userId: "55555555-5555-4555-8555-555555555555", roles: [] };
export const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };

const unscored = { scores: "none", scaleSensitive: false, seeded: true } as const;
registerModels([defineModel({ backend: "test-p5", id: "linear", taskType: "regression", displayName: "Test linear", parameters: { alpha: { kind: "float", min: 0, max: 100, default: 1 } }, capabilities: unscored })]);

/** y = 2x + 1 exactly; metrics depend on alpha so comparisons and searches have something to rank. */
export class LinearTestBackend implements MlBackend {
  readonly id = "test-p5";
  readonly capabilities = { artifactFormats: ["test-p5-v1"], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: ["standard_scale"], iterative: false, tasks: ["regression"], inputs: ["number"] } as const;
  validate(): void {}
  async prepare(_s: Snapshot, rows: Row[]): Promise<PreparedData> { const all = rows.map((_, i) => i); return { indices: { train: all.slice(0, 36), validation: all.slice(36, 48), test: all.slice(48) }, environment: { protocol: "test-p5", library: "1.0" } }; }
  async train(s: Snapshot): Promise<TrainedModel> { return { bytes: Buffer.from(`linear:${s.hyperparameters.alpha ?? 1}`), format: "test-p5-v1", environment: { protocol: "test-p5", library: "1.0" }, resolvedHyperparameters: { alpha: s.hyperparameters.alpha ?? 1 } }; }
  async evaluate(s: Snapshot): Promise<MetricValue[]> {
    const alpha = Number(s.hyperparameters.alpha ?? 1);
    return ["train", "validation", "test"].map(partition => ({ name: "mae", partition: partition as MetricValue["partition"], value: Math.abs(alpha - 3) / 10 + .01, direction: "lower" }));
  }
  async predict(_s: Snapshot, _a: Buffer, rows: Row[]): Promise<unknown[]> { return rows.map(r => 2 * Number(r.x) + 1); }
}

export async function withStore<T>(work: (store: PostgresArtifactStore) => Promise<T>): Promise<T> { return work(new PostgresArtifactStore(pool)); }
/** Removes stored bytes beneath the append-only blob table (as its owner), to simulate a lost artifact. */
export async function dropBlob(tenantId: string, sha256: string): Promise<void> {
  const c = await pool.connect();
  try { await c.query("ALTER TABLE ml_blobs DISABLE TRIGGER ml_blob_append_only"); await c.query("DELETE FROM ml_blobs WHERE tenant_id=$1 AND content_hash=$2", [tenantId, sha256]); }
  finally { await c.query("ALTER TABLE ml_blobs ENABLE TRIGGER ml_blob_append_only"); c.release(); }
}
// ml_reports references records only by id (in its sources), so no foreign key cascades to it; it is listed explicitly.
export const truncate = () => pool.query("TRUNCATE ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,ml_reports,runs,agents CASCADE");

export async function trainedModel(store: PostgresArtifactStore, backend: MlBackend = new LinearTestBackend(), options: { alpha?: number; rows?: number; algorithm?: string } = {}) {
  const backends = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, backends);
  const dataset = await training.ingest(manager, { name: "linear", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "y", type: "number" }], target: "y" }, rows: Array.from({ length: options.rows ?? 60 }, (_, i) => ({ x: i, y: 2 * i + 1 })) });
  const pipeline = await training.pipeline(manager, { name: "numeric", definition: { features: ["x"], steps: [] } });
  const experiment = await training.experiment(manager, { name: "phase5", project: "mlops", tags: { team: "core" } });
  const spec = { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id, algorithm: options.algorithm ?? "linear", hyperparameters: { alpha: options.alpha ?? 3 }, seed: 7, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2 } };
  const job = await training.queue(manager, { experimentId: experiment.id, spec });
  const worker = new LifecycleWorker(pool, { workerId: "mlops-test", leaseSeconds: 30, kind: "ml" }, new TrainingProcessor(pool, store, backends));
  await worker.tick();
  const attempt = (await training.getJob(manager, job.id)).attempts[0];
  return { backends, training, models, dataset, pipeline, experiment, spec, job, attempt, worker };
}

export function testApp(store: PostgresArtifactStore, backends: ReadonlyMap<string, MlBackend>, principal: Principal = manager, extra: Record<string, unknown> = {}) {
  const app = new Hono<{ Variables: { principal: Principal; database: pg.Pool; correlationId: string } }>();
  app.use("*", async (c, next) => { c.set("principal", principal); c.set("database", pool); await next(); });
  // Mirrors the status mapping of src/api/app.ts for the error classes ML routes throw.
  app.onError((error, c) => c.json({ error: error.message, reasons: (error as { reasons?: unknown }).reasons }, ((error as { status?: number }).status ?? (error.name === "ZodError" ? 400 : 500)) as 400));
  registerMlRoutes(app, { enabled: true, backends, ...extra } as never);
  return app;
}
export const json = (body: unknown, method = "POST") => ({ method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
