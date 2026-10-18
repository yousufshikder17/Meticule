import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJson, canonicalJsonHash } from "../domain/canonical-json.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { TrainingSnapshot, cohortKey, type MetricValue } from "./domain.js";
import { sequence } from "./lineage.js";
import { audit, ExperimentTags, MlTrainingService, authorizeMl, transaction } from "./persistence.js";

const Name = z.string().trim().min(1).max(120);
export const EXPERIMENT_STATUSES = ["EMPTY", "RUNNING", "COMPLETED", "PARTIALLY_COMPLETED", "FAILED", "CANCELLED"] as const;
export const ExperimentQuery = z.strictObject({
  project: Name.optional(), nameContains: z.string().min(1).max(120).optional(), status: z.enum(EXPERIMENT_STATUSES).optional(),
  // `tag=key:value`; repeated keys narrow the result (every pair must match).
  tag: z.array(z.string().regex(/^[^:]{1,60}:.{0,200}$/)).max(10).default([]),
  parentExperimentId: z.uuid().optional(), createdAfter: z.iso.datetime().optional(), createdBefore: z.iso.datetime().optional(),
  limit: z.coerce.number().int().min(1).max(100).default(100),
});
export const ExperimentPatch = z.strictObject({ description: z.string().max(2000).optional(), tags: ExperimentTags.optional() }).refine(p => p.description !== undefined || p.tags !== undefined, "Nothing to update");
export const CompareRuns = z.strictObject({ trainingRunIds: z.array(z.uuid()).min(2).max(10).refine(x => new Set(x).size === x.length, "Duplicate runs") });

// Status is derived from the experiment's jobs, never stored, so it cannot drift from the durable job states.
const STATUS_SQL = `CASE WHEN s.total=0 THEN 'EMPTY' WHEN s.open>0 THEN 'RUNNING' WHEN s.completed=s.total THEN 'COMPLETED'
  WHEN s.completed>0 THEN 'PARTIALLY_COMPLETED' WHEN s.failed>0 THEN 'FAILED' ELSE 'CANCELLED' END`;
const SUMMARY_SQL = `LEFT JOIN LATERAL (SELECT count(*)::int AS total, count(*) FILTER(WHERE j.status='COMPLETED')::int AS completed,
  count(*) FILTER(WHERE j.status='FAILED')::int AS failed, count(*) FILTER(WHERE j.status='CANCELLED')::int AS cancelled,
  count(*) FILTER(WHERE j.status NOT IN ('COMPLETED','FAILED','CANCELLED'))::int AS open
  FROM ml_training_jobs j WHERE j.tenant_id=e.tenant_id AND j.experiment_id=e.id) s ON true`;
const like = (text: string) => `%${text.replace(/[\\%_]/g, m => `\\${m}`)}%`;
// Exact equality is "identical"; this relative tolerance only separates "numerically equivalent" from "different".
const EQUIVALENCE_TOLERANCE = 1e-9;

/** Experiment queries, run history, comparison and reproduction over the existing durable training records. */
export class MlTrackingService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>) {}

  async list(principal: Principal, raw: unknown) {
    authorizeMl(principal); const q = ExperimentQuery.parse(raw);
    const tags = Object.fromEntries(q.tag.map(t => { const i = t.indexOf(":"); return [t.slice(0, i), t.slice(i + 1)]; }));
    return (await this.pool.query(`SELECT * FROM (SELECT e.*,${STATUS_SQL} AS status,to_jsonb(s) AS summary FROM ml_experiments e ${SUMMARY_SQL}
      WHERE e.tenant_id=$1 AND ($2::text IS NULL OR e.project=$2) AND ($3::text IS NULL OR e.name ILIKE $3) AND e.tags @> $4::jsonb
        AND ($5::uuid IS NULL OR e.parent_experiment_id=$5) AND ($6::timestamptz IS NULL OR e.created_at>=$6) AND ($7::timestamptz IS NULL OR e.created_at<$7)) x
      WHERE ($8::text IS NULL OR x.status=$8) ORDER BY created_at DESC,id LIMIT $9`,
      [principal.tenantId, q.project ?? null, q.nameContains ? like(q.nameContains) : null, JSON.stringify(tags), q.parentExperimentId ?? null, q.createdAfter ?? null, q.createdBefore ?? null, q.status ?? null, q.limit])).rows;
  }

  async get(principal: Principal, id: string) {
    authorizeMl(principal);
    const e = await this.pool.query(`SELECT e.*,${STATUS_SQL} AS status,to_jsonb(s) AS summary FROM ml_experiments e ${SUMMARY_SQL} WHERE e.tenant_id=$1 AND e.id=$2`, [principal.tenantId, id]);
    if (!e.rowCount) throw new NotFoundError("Experiment not found");
    const [jobs, children] = await sequence([
      () => this.pool.query("SELECT * FROM ml_training_jobs WHERE tenant_id=$1 AND experiment_id=$2 ORDER BY created_at DESC LIMIT 100", [principal.tenantId, id]),
      () => this.pool.query("SELECT id,name,project,created_at FROM ml_experiments WHERE tenant_id=$1 AND parent_experiment_id=$2 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, id]),
    ]);
    return { experiment: e.rows[0], jobs: jobs.rows, children: children.rows };
  }

  async update(principal: Principal, id: string, raw: unknown) {
    authorizeMl(principal, true); const patch = ExperimentPatch.parse(raw);
    return transaction(this.pool, async c => {
      const r = await c.query("UPDATE ml_experiments SET description=COALESCE($3,description),tags=COALESCE($4::jsonb,tags) WHERE tenant_id=$1 AND id=$2 RETURNING *", [principal.tenantId, id, patch.description ?? null, patch.tags ? JSON.stringify(patch.tags) : null]);
      if (!r.rowCount) throw new NotFoundError("Experiment not found");
      await audit(c, principal.tenantId, principal.userId, "ml.experiment_updated", { id, fields: Object.keys(patch) }); return r.rows[0];
    });
  }

  /** Every attempt of every job in the experiment, newest first, with its parameters, outcome and metrics. */
  async runs(principal: Principal, experimentId: string) {
    authorizeMl(principal);
    if (!(await this.pool.query("SELECT 1 FROM ml_experiments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, experimentId])).rowCount) throw new NotFoundError("Experiment not found");
    return (await this.pool.query(`SELECT t.id AS training_run_id,j.id AS job_id,j.reproduces_job_id,t.attempt,t.status,t.started_at,t.ended_at,t.failure,
        j.snapshot->>'backend' AS backend,j.snapshot->>'algorithm' AS algorithm,j.snapshot->'hyperparameters' AS hyperparameters,(j.snapshot->>'seed')::bigint AS seed,
        j.dataset_version_id,j.pipeline_id,t.resolved_hyperparameters,t.resumed_from_epoch,t.best_epoch,t.stop_reason,t.performance,
        COALESCE((SELECT jsonb_agg(jsonb_build_object('name',m.name,'partition',m.partition,'value',m.value,'direction',m.direction,'fold',m.fold,'std',m.std) ORDER BY m.partition,m.name,m.fold)
          FROM ml_metrics m WHERE m.tenant_id=t.tenant_id AND m.training_run_id=t.id),'[]') AS metrics
      FROM ml_training_runs t JOIN ml_training_jobs j ON j.tenant_id=t.tenant_id AND j.id=t.job_id
      WHERE t.tenant_id=$1 AND j.experiment_id=$2 ORDER BY t.started_at DESC,t.id LIMIT 500`, [principal.tenantId, experimentId])).rows;
  }

  /** Side-by-side view of 2..10 attempts. Metrics are only declared comparable when every attempt shares one cohort. */
  async compare(principal: Principal, raw: unknown) {
    authorizeMl(principal); const { trainingRunIds } = CompareRuns.parse(raw);
    const rows = (await this.pool.query(`SELECT t.id,t.job_id,t.status,t.snapshot,t.split_indices,t.environment,t.resolved_hyperparameters,j.experiment_id FROM ml_training_runs t
      JOIN ml_training_jobs j ON j.tenant_id=t.tenant_id AND j.id=t.job_id WHERE t.tenant_id=$1 AND t.id=ANY($2::uuid[])`, [principal.tenantId, trainingRunIds])).rows;
    if (rows.length !== trainingRunIds.length) throw new NotFoundError("Training run not found");
    const metrics = (await this.pool.query("SELECT training_run_id,name,partition,value,direction,fold FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=ANY($2::uuid[])", [principal.tenantId, trainingRunIds])).rows;
    const ordered = trainingRunIds.map(id => rows.find(r => r.id === id)!);
    const completed = ordered.filter(r => r.status === "COMPLETED");
    const cohorts = new Set(completed.map(r => cohortKey(r.snapshot, r.split_indices)));
    const keys = new Map<string, { name: string; partition: string; fold: number | null; direction: string }>();
    for (const m of metrics) keys.set(`${m.partition}/${m.name}/${m.fold ?? ""}`, { name: m.name, partition: m.partition, fold: m.fold, direction: m.direction });
    const table = [...keys.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([key, k]) => {
      const values = Object.fromEntries(ordered.map(r => [r.id, metrics.find(m => m.training_run_id === r.id && `${m.partition}/${m.name}/${m.fold ?? ""}` === key)?.value ?? null]));
      const present = Object.entries(values).filter(([, v]) => v !== null) as [string, number][];
      const best = present.length ? present.reduce((a, b) => (k.direction === "higher" ? b[1] > a[1] : b[1] < a[1]) ? b : a)[0] : null;
      return { ...k, values, best };
    });
    const envKeys = [...new Set(ordered.flatMap(r => Object.keys(r.environment ?? {})))].sort();
    const environmentDifferences = envKeys.filter(k => new Set(ordered.map(r => canonicalJson(r.environment?.[k] ?? null))).size > 1)
      .map(k => ({ key: k, values: Object.fromEntries(ordered.map(r => [r.id, r.environment?.[k] ?? null])) }));
    return {
      runs: ordered.map(r => { const s = TrainingSnapshot.parse(r.snapshot); return { trainingRunId: r.id, jobId: r.job_id, experimentId: r.experiment_id, status: r.status, backend: s.backend, algorithm: s.algorithm, hyperparameters: s.hyperparameters, resolvedHyperparameters: r.resolved_hyperparameters, seed: s.seed, datasetVersionId: s.datasetVersionId, datasetSha256: s.dataset.content.sha256, featurePipelineId: s.featurePipelineId }; }),
      comparable: completed.length === ordered.length && cohorts.size === 1,
      comparabilityNote: completed.length !== ordered.length ? "Some runs are not completed" : cohorts.size > 1 ? "Runs differ in data, schema, features, split, seed or folds; metric differences are not attributable to the model alone" : "All runs share one cohort",
      metrics: table, environmentDifferences,
    };
  }

  /**
   * Queues an ordinary job with the original job's frozen specification (same dataset version, pipeline version, split
   * definition, seed and hyperparameters) in the same experiment. Refused when the frozen inputs are no longer available.
   */
  async reproduce(principal: Principal, jobId: string) {
    authorizeMl(principal, true);
    const original = (await this.pool.query("SELECT * FROM ml_training_jobs WHERE tenant_id=$1 AND id=$2", [principal.tenantId, jobId])).rows[0];
    if (!original) throw new NotFoundError("Training job not found");
    const snapshot = TrainingSnapshot.parse(original.snapshot);
    const backend = this.backends.get(snapshot.backend); if (!backend) throw new ConflictError("The original backend is not configured; the run cannot be reproduced here");
    try { await this.artifacts.get(principal.tenantId, snapshot.dataset.content); } catch { throw new ConflictError("The original dataset content is unavailable or failed integrity verification"); }
    backend.validate(snapshot);
    const { dataset: _dataset, pipeline: _pipeline, ...spec } = snapshot;
    const job = await transaction(this.pool, async c => {
      const queued = await new MlTrainingService(this.pool, this.artifacts).queueIn(c, principal, { experimentId: original.experiment_id, spec }, jobId);
      await audit(c, principal.tenantId, principal.userId, "ml.training_reproduction_queued", { jobId: queued.id, reproducesJobId: jobId }, queued.run_id); return queued;
    });
    return { job, reproducesJobId: jobId, caveats: [
      "Same frozen dataset content, pipeline definition, split definition, seed and hyperparameters as the original job.",
      "Results can still differ when library versions, platform, hardware or thread scheduling differ; compare the two attempts with GET /ml/training-runs/:id/reproducibility/:candidate.",
    ] };
  }

  /** Evidence-based verdict on whether `candidate` reproduced `original`. Never claims more than the stored records show. */
  async reproducibility(principal: Principal, originalId: string, candidateId: string) {
    authorizeMl(principal);
    const rows = (await this.pool.query("SELECT id,snapshot,split_indices,environment FROM ml_training_runs WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='COMPLETED'", [principal.tenantId, [originalId, candidateId]])).rows;
    const a = rows.find(r => r.id === originalId), b = rows.find(r => r.id === candidateId);
    if (!a || !b) throw new NotFoundError("Completed training runs not found");
    const spec = (raw: unknown) => { const { dataset, pipeline, split, ...s } = TrainingSnapshot.parse(raw); return canonicalJsonHash({ ...s, split: { ...split, id: null }, dataset: dataset.content.sha256, schema: dataset.schema, pipeline: pipeline.definition }); };
    const sameSpecification = spec(a.snapshot) === spec(b.snapshot);
    const sameSplit = canonicalJson(a.split_indices) === canonicalJson(b.split_indices);
    const envKeys = [...new Set([...Object.keys(a.environment), ...Object.keys(b.environment)])].sort();
    const environmentDifferences = envKeys.filter(k => a.environment[k] !== b.environment[k]).map(k => ({ key: k, original: a.environment[k] ?? null, candidate: b.environment[k] ?? null }));
    const load = async (id: string) => (await this.pool.query("SELECT name,partition,value,direction,fold FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=$2", [principal.tenantId, id])).rows as MetricValue[];
    const ma = await load(originalId), mb = await load(candidateId);
    const metrics = ma.map(m => { const o = mb.find(x => x.name === m.name && x.partition === m.partition && (x.fold ?? null) === (m.fold ?? null)); return { name: m.name, partition: m.partition, fold: m.fold ?? null, original: m.value, candidate: o?.value ?? null, absoluteDelta: o ? Math.abs(o.value - m.value) : null }; });
    const missing = metrics.some(m => m.candidate === null) || mb.length !== ma.length;
    const exact = !missing && metrics.every(m => m.absoluteDelta === 0);
    const close = !missing && metrics.every(m => m.absoluteDelta! <= EQUIVALENCE_TOLERANCE * Math.max(1, Math.abs(m.original)));
    const verdict = !sameSpecification ? "not_comparable" : sameSplit && exact ? "identical" : sameSplit && close ? "numerically_equivalent" : "different";
    return { verdict, sameSpecification, sameSplit, environmentDifferences, metrics, tolerance: EQUIVALENCE_TOLERANCE,
      note: "The verdict covers the stored split, metrics and recorded environment only. Matching values do not prove bit-identical artifacts." };
  }
}
