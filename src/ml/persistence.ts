import type pg from "pg";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJson, canonicalJsonHash } from "../domain/canonical-json.js";
import { RunRepository } from "../db/repositories.js";
import { scopedPool } from "../db/tenant-session.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, Performance, PreparedData, TrainedModel } from "./backend.js";
import { DatasetSchema, DatasetVersion, FeaturePipeline, Metric, PipelineDefinition, TrainingSnapshot, TrainingSpec, assertMetricShape, assertTrainingTransition, validateRows, validateSnapshot, type ArtifactReference, type MetricValue, type Snapshot, type TrainingState } from "./domain.js";

export function authorizeMl(p: Principal, write = false): void {
  if (write && !p.roles.includes("tenant_admin")) throw new AuthorizationError("Required role: tenant_admin");
}
export async function transaction<T>(pool: pg.Pool, work: (client: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try { await client.query("BEGIN"); const result = await work(client); await client.query("COMMIT"); return result; }
  catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
  finally { client.release(); }
}
export async function audit(c: pg.PoolClient, tenant: string, actor: string, event: string, details: unknown, run: string | null = null, actorType: "user" | "worker" = "user"): Promise<void> {
  await c.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,$3,$4,$5,$6)", [tenant, run, actorType, actor, event, JSON.stringify(details)]);
}
export async function ownedRun(c: pg.PoolClient, runId: string, workerId: string) {
  const r = await c.query("SELECT *,lease_expires_at>now() AS lease_valid FROM runs WHERE id=$1 FOR UPDATE", [runId]);
  const run = r.rows[0];
  if (!run || run.kind !== "ml" || run.status !== "running" || run.cancellation_requested_at || run.lease_owner !== workerId || !run.lease_valid) throw new ConflictError("ML worker does not own an executable lease");
  return run;
}
/** Fold k is stored as `fold_k` (its validation indices) beside train/validation/test, so cohort comparison sees exact CV folds. */
export function splitIndices(prepared: PreparedData): Record<string, number[]> {
  return { ...prepared.indices, ...Object.fromEntries((prepared.folds ?? []).map((f, i) => [`fold_${i}`, f])) };
}
const Name = z.string().trim().min(1).max(120);
export const IngestDataset = z.object({ name: Name, schema: DatasetSchema, rows: z.unknown() });
export const CreatePipeline = z.object({ name: Name, definition: PipelineDefinition });
export const QueueTraining = z.object({ experimentId: z.uuid(), spec: TrainingSpec });

export class MlTrainingService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore) {}
  async ingest(principal: Principal, raw: unknown, datasetId?: string) {
    authorizeMl(principal, true); const input = IngestDataset.parse(raw);
    const rows = validateRows(input.schema, input.rows);
    const content = await this.artifacts.put(principal.tenantId, Buffer.from(canonicalJson(rows)), "application/json");
    return transaction(this.pool, async c => {
      await c.query("INSERT INTO ml_dataset_schemas(id,tenant_id,document) VALUES($1,$2,$3) ON CONFLICT DO NOTHING", [input.schema.id, principal.tenantId, JSON.stringify(input.schema)]);
      const existingSchema = await c.query("SELECT document FROM ml_dataset_schemas WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.schema.id]);
      if (canonicalJsonHash(existingSchema.rows[0].document) !== canonicalJsonHash(input.schema)) throw new ConflictError("Dataset schema ID already has a different definition");
      let id = datasetId;
      if (id) {
        const parent = await c.query("SELECT id FROM ml_datasets WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, id]);
        if (!parent.rowCount) throw new NotFoundError("Dataset not found");
      } else {
        id = randomUUID(); await c.query("INSERT INTO ml_datasets(id,tenant_id,name,created_by) VALUES($1,$2,$3,$4)", [id, principal.tenantId, input.name, principal.userId]);
      }
      const versions = await c.query("SELECT COALESCE(max(version),0)+1 AS version FROM ml_dataset_versions WHERE tenant_id=$1 AND dataset_id=$2", [principal.tenantId, id]);
      const doc = DatasetVersion.parse({ id: randomUUID(), tenantId: principal.tenantId, datasetId: id, version: versions.rows[0].version, createdAt: new Date().toISOString(), schema: input.schema, content, rowCount: rows.length, validation: { valid: true, validator: "meticule-tabular-v1" } });
      await c.query("INSERT INTO ml_dataset_versions(id,tenant_id,dataset_id,version,document,schema_id) VALUES($1,$2,$3,$4,$5,$6)", [doc.id, principal.tenantId, id, doc.version, JSON.stringify(doc), input.schema.id]);
      await audit(c, principal.tenantId, principal.userId, "ml.dataset_version_created", { datasetId: id, versionId: doc.id }); return doc;
    });
  }
  async pipeline(principal: Principal, raw: unknown, logicalId?: string) {
    authorizeMl(principal, true); const input = CreatePipeline.parse(raw);
    return transaction(this.pool, c => this.pipelineIn(c, principal, input, logicalId));
  }
  /** Transaction-scoped form so orchestrations can create pipelines atomically with their jobs. */
  async pipelineIn(c: pg.PoolClient, principal: Principal, input: z.infer<typeof CreatePipeline>, logicalId?: string) {
    let version = 1; const id = randomUUID();
    if (logicalId) {
      const root = await c.query("SELECT id FROM ml_feature_pipelines WHERE tenant_id=$1 AND logical_id=$2 AND version=1 FOR UPDATE", [principal.tenantId, logicalId]);
      if (!root.rowCount) throw new NotFoundError("Pipeline not found");
      const r = await c.query("SELECT max(version)+1 AS version FROM ml_feature_pipelines WHERE tenant_id=$1 AND logical_id=$2", [principal.tenantId, logicalId]); version = r.rows[0].version;
    }
    const doc = FeaturePipeline.parse({ id, tenantId: principal.tenantId, logicalId: logicalId ?? id, version, name: input.name, definition: input.definition, createdAt: new Date().toISOString() });
    await c.query("INSERT INTO ml_feature_pipelines(id,tenant_id,logical_id,version,document) VALUES($1,$2,$3,$4,$5)", [id, principal.tenantId, doc.logicalId, version, JSON.stringify(doc)]);
    await audit(c, principal.tenantId, principal.userId, "ml.pipeline_version_created", { pipelineId: id }); return doc;
  }
  async experiment(principal: Principal, name: string) {
    authorizeMl(principal, true); Name.parse(name);
    return transaction(this.pool, async c => {
      const r = await c.query("INSERT INTO ml_experiments(tenant_id,name,created_by) VALUES($1,$2,$3) RETURNING *", [principal.tenantId, name, principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.experiment_created", { id: r.rows[0].id }); return r.rows[0];
    });
  }
  async queue(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = QueueTraining.parse(raw);
    return transaction(this.pool, c => this.queueIn(c, principal, input));
  }
  /** Transaction-scoped form used by benchmark/search/AutoML to create many ordinary jobs atomically. */
  async queueIn(c: pg.PoolClient, principal: Principal, input: z.infer<typeof QueueTraining>) {
    const e = await c.query("SELECT id FROM ml_experiments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.experimentId]);
    const d = await c.query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.spec.datasetVersionId]);
    const p = await c.query("SELECT document FROM ml_feature_pipelines WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.spec.featurePipelineId]);
    if (!e.rowCount || !d.rowCount || !p.rowCount) throw new NotFoundError("Experiment, dataset version or pipeline not found");
    const snapshot = TrainingSnapshot.parse({ ...input.spec, dataset: d.rows[0].document, pipeline: p.rows[0].document }); validateSnapshot(snapshot);
    const runId = randomUUID();
    await c.query("INSERT INTO runs(id,tenant_id,kind,created_by,goal,root_run_id,token_budget_limit,cost_budget_limit_microusd) VALUES($1,$2,'ml',$3,'ML training',$1,0,0)", [runId, principal.tenantId, principal.userId]);
    const job = await c.query("INSERT INTO ml_training_jobs(tenant_id,run_id,experiment_id,dataset_version_id,pipeline_id,snapshot) VALUES($1,$2,$3,$4,$5,$6) RETURNING *", [principal.tenantId, runId, input.experimentId, snapshot.datasetVersionId, snapshot.featurePipelineId, JSON.stringify(snapshot)]);
    await audit(c, principal.tenantId, principal.userId, "ml.training_queued", { jobId: job.rows[0].id }, runId); return job.rows[0];
  }
  async getJob(principal: Principal, id: string) {
    authorizeMl(principal);
    const j = await this.pool.query("SELECT j.*,r.status AS runtime_status FROM ml_training_jobs j JOIN runs r ON r.tenant_id=j.tenant_id AND r.id=j.run_id WHERE j.tenant_id=$1 AND j.id=$2", [principal.tenantId, id]);
    if (!j.rowCount) throw new NotFoundError("Training job not found");
    const attempts = await this.pool.query("SELECT * FROM ml_training_runs WHERE tenant_id=$1 AND job_id=$2 ORDER BY attempt", [principal.tenantId, id]);
    const metrics = await this.pool.query("SELECT m.* FROM ml_metrics m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE t.tenant_id=$1 AND t.job_id=$2", [principal.tenantId, id]);
    const artifacts = await this.pool.query("SELECT a.* FROM ml_artifacts a JOIN ml_training_runs t ON t.tenant_id=a.tenant_id AND t.id=a.training_run_id WHERE t.tenant_id=$1 AND t.job_id=$2", [principal.tenantId, id]);
    return { job: j.rows[0], attempts: attempts.rows, metrics: metrics.rows, artifacts: artifacts.rows };
  }
  async begin(runId: string, workerId: string) {
    return transaction(this.pool, async c => {
      const run = await ownedRun(c, runId, workerId);
      const job = (await c.query("SELECT * FROM ml_training_jobs WHERE run_id=$1 FOR UPDATE", [runId])).rows[0];
      if (!job || job.status !== "QUEUED") throw new ConflictError("Training job is not queued");
      const snapshot = TrainingSnapshot.parse(job.snapshot); validateSnapshot(snapshot);
      const next = await c.query("SELECT COALESCE(max(attempt),0)+1 AS attempt FROM ml_training_runs WHERE tenant_id=$1 AND job_id=$2", [run.tenant_id, job.id]);
      const t = await c.query("INSERT INTO ml_training_runs(tenant_id,job_id,attempt,snapshot) VALUES($1,$2,$3,$4) RETURNING *", [run.tenant_id, job.id, next.rows[0].attempt, JSON.stringify(snapshot)]);
      await c.query("UPDATE ml_training_jobs SET status='PREPARING' WHERE id=$1", [job.id]);
      await c.query("UPDATE ml_training_runs SET status='PREPARING' WHERE id=$1", [t.rows[0].id]);
      await audit(c, run.tenant_id, workerId, "ml.attempt_started", { trainingRunId: t.rows[0].id, attempt: next.rows[0].attempt }, runId, "worker");
      return { id: t.rows[0].id as string, tenantId: run.tenant_id as string, snapshot };
    });
  }
  async phase(runId: string, workerId: string, attemptId: string, to: TrainingState, prepared?: PreparedData, model?: TrainedModel, performance?: Performance) {
    return transaction(this.pool, async c => {
      const run = await ownedRun(c, runId, workerId);
      const t = (await c.query("SELECT t.* FROM ml_training_runs t JOIN ml_training_jobs j ON j.id=t.job_id AND j.tenant_id=t.tenant_id WHERE t.id=$1 AND j.run_id=$2 FOR UPDATE OF t", [attemptId, runId])).rows[0];
      if (!t) throw new ConflictError("Attempt is not current"); assertTrainingTransition(t.status, to);
      await c.query("UPDATE ml_training_runs SET status=$1,environment=COALESCE($2::jsonb,environment),split_indices=COALESCE($3::jsonb,split_indices),resolved_hyperparameters=COALESCE($4::jsonb,resolved_hyperparameters),performance=performance||$6::jsonb WHERE id=$5", [to, model ? JSON.stringify(model.environment) : prepared ? JSON.stringify(prepared.environment) : null, prepared ? JSON.stringify(splitIndices(prepared)) : null, model ? JSON.stringify(model.resolvedHyperparameters) : null, attemptId, JSON.stringify({ ...model?.performance, ...performance })]);
      await c.query("UPDATE ml_training_jobs SET status=$1 WHERE id=$2", [to, t.job_id]);
      await audit(c, run.tenant_id, workerId, "ml.phase_changed", { trainingRunId: attemptId, from: t.status, to }, runId, "worker");
    });
  }
  async finish(runId: string, workerId: string, attemptId: string, snapshot: Snapshot, metrics: MetricValue[], content: ArtifactReference, format: string, durationMs: number) {
    return transaction(this.pool, async c => {
      const run = await ownedRun(c, runId, workerId);
      const t = (await c.query("SELECT t.id FROM ml_training_runs t JOIN ml_training_jobs j ON j.id=t.job_id AND j.tenant_id=t.tenant_id WHERE t.id=$1 AND j.run_id=$2 AND t.status='SAVING' FOR UPDATE OF t", [attemptId, runId])).rows[0];
      if (!t) throw new ConflictError("No current saving attempt");
      for (const raw of metrics) {
        assertMetricShape(raw); const m = Metric.parse({ ...raw, id: randomUUID() });
        await c.query("INSERT INTO ml_metrics(id,tenant_id,training_run_id,name,partition,value,direction,fold,std) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)", [m.id, run.tenant_id, attemptId, m.name, m.partition, m.value, m.direction, m.fold ?? null, m.std ?? null]);
      }
      const a = await c.query("INSERT INTO ml_artifacts(tenant_id,training_run_id,kind,format,content) VALUES($1,$2,'model',$3,$4) RETURNING id", [run.tenant_id, attemptId, format, JSON.stringify(content)]);
      await c.query("INSERT INTO usage_records(tenant_id,run_id,provider,model,input_tokens,output_tokens,cost_microusd,usage_kind,duration_ms) VALUES($1,$2,$3,$4,0,0,0,'ml_training',$5)", [run.tenant_id, runId, snapshot.backend, snapshot.algorithm, durationMs]);
      await new RunRepository(scopedPool(c)).complete(runId, workerId, { trainingRunId: attemptId, artifactId: a.rows[0].id });
      await audit(c, run.tenant_id, workerId, "ml.training_completed", { trainingRunId: attemptId, artifactId: a.rows[0].id }, runId, "worker");
    });
  }
  async fail(runId: string, workerId: string, attemptId: string | undefined, message: string): Promise<void> {
    await transaction(this.pool, async c => {
      const r = (await c.query("SELECT *,lease_expires_at>now() AS valid FROM runs WHERE id=$1 FOR UPDATE", [runId])).rows[0];
      if (!r || r.kind !== "ml" || r.lease_owner !== workerId || !r.valid || !["running", "cancelling"].includes(r.status)) return;
      const active = await c.query("SELECT t.id FROM ml_training_runs t JOIN ml_training_jobs j ON j.tenant_id=t.tenant_id AND j.id=t.job_id WHERE j.run_id=$1 AND t.ended_at IS NULL", [runId]);
      // An old processor must never fail a newly reclaimed attempt with the same worker ID.
      if (attemptId ? active.rows[0]?.id !== attemptId : active.rowCount) return;
      await new RunRepository(scopedPool(c)).workerTransition(runId, workerId, r.status === "cancelling" ? "cancelled" : "failed", { code: "ml_training_failed", message });
    });
  }
}
