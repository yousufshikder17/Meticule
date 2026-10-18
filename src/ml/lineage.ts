import type pg from "pg";
import { NotFoundError } from "../domain/errors.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore } from "./backend.js";
import { TrainingSnapshot, type ArtifactReference } from "./domain.js";
import { authorizeMl } from "./persistence.js";

/** A lineage defect: a reference that is missing, points elsewhere, or whose bytes fail verification. */
export interface LineageIssue { code: string; subject: string; id: string }
type Db = pg.Pool | pg.PoolClient;
export async function sequence<T extends (() => Promise<unknown>)[]>(queries: [...T]): Promise<{ [K in keyof T]: Awaited<ReturnType<T[K]>> }> {
  const out: unknown[] = []; for (const q of queries) out.push(await q()); return out as { [K in keyof T]: Awaited<ReturnType<T[K]>> };
}

/**
 * Provenance is the existing relational record (foreign keys between datasets, pipelines, jobs, attempts, artifacts,
 * model versions and endpoints), read here as one document. Every query filters the caller's tenant, so another
 * tenant's identifiers are simply not found. Nothing is copied or cached: the document is derived on each read.
 */
export class MlLineageService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore) {}

  async modelVersion(principal: Principal, id: string, verify = false) {
    authorizeMl(principal);
    const m = (await this.pool.query("SELECT m.*,r.name AS registry_name,a.format,a.content,a.training_run_id AS artifact_training_run_id FROM ml_model_versions m JOIN ml_registry_entries r ON r.tenant_id=m.tenant_id AND r.id=m.registry_entry_id JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id WHERE m.tenant_id=$1 AND m.id=$2", [principal.tenantId, id])).rows[0];
    if (!m) throw new NotFoundError("Model version not found");
    const issues: LineageIssue[] = [];
    const upstream = await this.upstream(this.pool, principal.tenantId, m.training_run_id, issues);
    if (upstream.trainingRun.status !== "COMPLETED") issues.push({ code: "model_attempt_not_completed", subject: "training_run", id: m.training_run_id });
    if (m.artifact_training_run_id !== m.training_run_id) issues.push({ code: "artifact_attempt_mismatch", subject: "artifact", id: m.artifact_id });
    if (verify) await this.verify(principal.tenantId, [["dataset_version", upstream.datasetVersion.id, upstream.datasetVersion.content], ["model_artifact", m.artifact_id, m.content]], issues);
    return {
      modelVersion: { id: m.id, registryEntryId: m.registry_entry_id, registryName: m.registry_name, version: m.version, status: m.status, createdAt: m.created_at, artifact: { id: m.artifact_id, format: m.format, sha256: m.content.sha256, size: m.content.size } },
      upstream, downstream: await this.downstream(this.pool, principal.tenantId, [m.id]), issues, verified: verify,
      lifecycle: (await this.pool.query("SELECT event,from_status,to_status,actor,reason,details,created_at FROM ml_model_events WHERE tenant_id=$1 AND model_version_id=$2 ORDER BY created_at,id LIMIT 500", [principal.tenantId, m.id])).rows,
    };
  }

  async trainingRun(principal: Principal, id: string, verify = false) {
    authorizeMl(principal);
    const issues: LineageIssue[] = [];
    const upstream = await this.upstream(this.pool, principal.tenantId, id, issues);
    const models = (await this.pool.query("SELECT id FROM ml_model_versions WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY created_at", [principal.tenantId, id])).rows.map(r => r.id as string);
    if (verify) await this.verify(principal.tenantId, [["dataset_version", upstream.datasetVersion.id, upstream.datasetVersion.content]], issues);
    return { upstream, downstream: await this.downstream(this.pool, principal.tenantId, models), issues, verified: verify };
  }

  /** Everything built from one dataset version: jobs, their attempts, registered versions and what serves them. */
  async datasetVersion(principal: Principal, id: string) {
    authorizeMl(principal);
    const d = (await this.pool.query("SELECT v.id,v.dataset_id,v.version,v.schema_id,v.document,d.name FROM ml_dataset_versions v JOIN ml_datasets d ON d.tenant_id=v.tenant_id AND d.id=v.dataset_id WHERE v.tenant_id=$1 AND v.id=$2", [principal.tenantId, id])).rows[0];
    if (!d) throw new NotFoundError("Dataset version not found");
    const jobs = (await this.pool.query(`SELECT j.id,j.experiment_id,j.status,j.pipeline_id,j.reproduces_job_id,j.snapshot->>'backend' AS backend,j.snapshot->>'algorithm' AS algorithm,j.created_at,
        COALESCE((SELECT jsonb_agg(jsonb_build_object('trainingRunId',t.id,'attempt',t.attempt,'status',t.status) ORDER BY t.attempt) FROM ml_training_runs t WHERE t.tenant_id=j.tenant_id AND t.job_id=j.id),'[]') AS attempts
      FROM ml_training_jobs j WHERE j.tenant_id=$1 AND j.dataset_version_id=$2 ORDER BY j.created_at DESC LIMIT 100`, [principal.tenantId, id])).rows;
    const models = (await this.pool.query("SELECT m.id FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id JOIN ml_training_jobs j ON j.tenant_id=t.tenant_id AND j.id=t.job_id WHERE m.tenant_id=$1 AND j.dataset_version_id=$2 ORDER BY m.created_at LIMIT 100", [principal.tenantId, id])).rows.map(r => r.id as string);
    return { datasetVersion: { id: d.id, datasetId: d.dataset_id, datasetName: d.name, version: d.version, schemaId: d.schema_id, rowCount: d.document.rowCount, sha256: d.document.content.sha256 }, jobs, downstream: await this.downstream(this.pool, principal.tenantId, models) };
  }

  /** Experiment (with ancestors), orchestration, job, attempts, checkpoints, dataset and pipeline of one attempt. */
  async upstream(db: Db, tenantId: string, trainingRunId: string, issues: LineageIssue[]) {
    const t = (await db.query("SELECT * FROM ml_training_runs WHERE tenant_id=$1 AND id=$2", [tenantId, trainingRunId])).rows[0];
    if (!t) throw new NotFoundError("Training run not found");
    const j = (await db.query("SELECT * FROM ml_training_jobs WHERE tenant_id=$1 AND id=$2", [tenantId, t.job_id])).rows[0];
    const snapshot = TrainingSnapshot.parse(j.snapshot);
    // Sequential on purpose: `db` may be one client (a transaction or the API's tenant session), which runs one query at a time.
    const [experiments, orchestrations, attempts, checkpoints, dataset, pipeline] = await sequence([
      () => db.query(`WITH RECURSIVE chain AS (SELECT e.*,0 AS depth FROM ml_experiments e WHERE e.tenant_id=$1 AND e.id=$2
        UNION ALL SELECT p.*,c.depth+1 FROM ml_experiments p JOIN chain c ON p.tenant_id=c.tenant_id AND p.id=c.parent_experiment_id WHERE c.depth<20)
        SELECT id,name,project,tags,parent_experiment_id FROM chain ORDER BY depth`, [tenantId, j.experiment_id]),
      () => db.query("SELECT o.id,o.kind,o.name,o.parent_id,m.label FROM ml_orchestration_members m JOIN ml_orchestrations o ON o.tenant_id=m.tenant_id AND o.id=m.orchestration_id WHERE m.tenant_id=$1 AND m.job_id=$2", [tenantId, j.id]),
      () => db.query("SELECT id,attempt,status,started_at,ended_at,resumed_from_epoch,failure->>'category' AS failure_category FROM ml_training_runs WHERE tenant_id=$1 AND job_id=$2 ORDER BY attempt", [tenantId, j.id]),
      () => db.query("SELECT c.id,c.training_run_id,c.epoch,c.format,c.content->>'sha256' AS sha256,t.attempt FROM ml_training_checkpoints c JOIN ml_training_runs t ON t.tenant_id=c.tenant_id AND t.id=c.training_run_id WHERE c.tenant_id=$1 AND t.job_id=$2 ORDER BY c.epoch", [tenantId, j.id]),
      () => db.query("SELECT v.id,v.dataset_id,v.version,v.schema_id,v.document,d.name FROM ml_dataset_versions v JOIN ml_datasets d ON d.tenant_id=v.tenant_id AND d.id=v.dataset_id WHERE v.tenant_id=$1 AND v.id=$2", [tenantId, j.dataset_version_id]),
      () => db.query("SELECT id,logical_id,version,document FROM ml_feature_pipelines WHERE tenant_id=$1 AND id=$2", [tenantId, j.pipeline_id]),
    ]);
    const d = dataset.rows[0], p = pipeline.rows[0];
    if (snapshot.datasetVersionId !== j.dataset_version_id || snapshot.dataset.id !== j.dataset_version_id) issues.push({ code: "snapshot_dataset_mismatch", subject: "training_job", id: j.id });
    if (snapshot.featurePipelineId !== j.pipeline_id || snapshot.pipeline.id !== j.pipeline_id) issues.push({ code: "snapshot_pipeline_mismatch", subject: "training_job", id: j.id });
    if (d.document.content.sha256 !== snapshot.dataset.content.sha256) issues.push({ code: "dataset_content_mismatch", subject: "dataset_version", id: d.id });
    if (experiments.rows.at(-1)?.parent_experiment_id) issues.push({ code: "experiment_chain_truncated", subject: "experiment", id: experiments.rows.at(-1)!.id });
    // A resumed attempt must point at a checkpoint written by an earlier attempt of the same job.
    for (const a of attempts.rows) if (a.resumed_from_epoch && !checkpoints.rows.some(c => c.epoch === a.resumed_from_epoch && c.attempt < a.attempt)) issues.push({ code: "checkpoint_reference_missing", subject: "training_run", id: a.id });
    return {
      experiment: experiments.rows[0], experimentAncestors: experiments.rows.slice(1), orchestrations: orchestrations.rows,
      job: { id: j.id, runId: j.run_id, status: j.status, reproducesJobId: j.reproduces_job_id, createdAt: j.created_at },
      trainingRun: { id: t.id, attempt: t.attempt, status: t.status, startedAt: t.started_at, endedAt: t.ended_at, resumedFromEpoch: t.resumed_from_epoch, bestEpoch: t.best_epoch, stopReason: t.stop_reason, environment: t.environment, resolvedHyperparameters: t.resolved_hyperparameters },
      attempts: attempts.rows, checkpoints: checkpoints.rows.map(({ attempt: _attempt, ...c }) => c),
      training: { backend: snapshot.backend, algorithm: snapshot.algorithm, hyperparameters: snapshot.hyperparameters, seed: snapshot.seed, split: snapshot.split, evaluation: snapshot.evaluation, policy: snapshot.training, source: snapshot.source },
      datasetVersion: { id: d.id, datasetId: d.dataset_id, datasetName: d.name, version: d.version, schemaId: d.schema_id, schema: d.document.schema, rowCount: d.document.rowCount, sha256: d.document.content.sha256, content: d.document.content as ArtifactReference },
      featurePipeline: { id: p.id, logicalId: p.logical_id, version: p.version, definition: p.document.definition },
    };
  }

  /** What serves the given model versions. Later Phase 5 milestones add deployments, exports and reports here. */
  async downstream(db: Db, tenantId: string, modelVersionIds: string[]) {
    if (!modelVersionIds.length) return { modelVersions: [], endpoints: [] };
    const [models, endpoints] = await sequence([
      () => db.query("SELECT id,registry_entry_id,version,status,training_run_id,created_at FROM ml_model_versions WHERE tenant_id=$1 AND id=ANY($2::uuid[]) ORDER BY created_at", [tenantId, modelVersionIds]),
      () => db.query("SELECT id,name,model_version_id,status,created_at FROM ml_endpoints WHERE tenant_id=$1 AND model_version_id=ANY($2::uuid[]) ORDER BY created_at", [tenantId, modelVersionIds]),
    ]);
    return { modelVersions: models.rows, endpoints: endpoints.rows };
  }

  /** Re-reads the bytes behind each reference through the store's checksum check. Reports which reference failed, never why. */
  private async verify(tenantId: string, references: [string, string, ArtifactReference][], issues: LineageIssue[]) {
    for (const [subject, id, reference] of references) {
      try { await this.artifacts.get(tenantId, reference); } catch { issues.push({ code: "artifact_unavailable", subject, id }); }
    }
  }
}
