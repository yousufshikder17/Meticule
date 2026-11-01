import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore } from "./backend.js";
import { isTabularType, TrainingSnapshot, type Row, type Snapshot } from "./domain.js";
import { getModel } from "./model-catalog.js";
import { audit, authorizeMl, transaction } from "./persistence.js";

/**
 * Drift policy. Numeric features use the training partition's deciles as bin edges; string and boolean features use their
 * 20 most frequent training values plus "other"; every feature has a null bucket. Proportions below EPSILON are raised to
 * it for PSI only (empty bins would make PSI infinite). Thresholds are the conventional PSI bands; they describe how far a
 * distribution moved, not statistical significance.
 */
export const DRIFT = { quantiles: [.1, .2, .3, .4, .5, .6, .7, .8, .9], topCategories: 20, epsilon: 1e-4, maxWindowDays: 31 } as const;
export const DriftInput = z.strictObject({
  modelVersionId: z.uuid().optional(), from: z.iso.datetime({ offset: true }).optional(), to: z.iso.datetime({ offset: true }).optional(),
  minSamples: z.int().min(1).max(1_000_000).default(100),
  thresholds: z.strictObject({ psiModerate: z.number().positive().default(.1), psiSignificant: z.number().positive().default(.25) }).refine(t => t.psiModerate < t.psiSignificant, "psiModerate must be below psiSignificant").default({ psiModerate: .1, psiSignificant: .25 }),
});
export const WindowInput = z.strictObject({ from: z.iso.datetime({ offset: true }).optional(), to: z.iso.datetime({ offset: true }).optional() });
export const FeedbackInput = z.strictObject({ labels: z.array(z.union([z.number().finite(), z.string().max(10_000), z.boolean()])).min(1).max(1000) });

type Cell = number | string | boolean | null;
export type Bins = { name: string; kind: "numeric"; edges: number[]; counts: number[] } | { name: string; kind: "categorical"; categories: string[]; counts: number[] };
export interface ReferenceProfile { version: 1; rows: number; features: Bins[]; output: Bins | null }
export type DriftStatus = "stable" | "moderate" | "significant" | "insufficient_data";

const quantile = (sorted: number[], p: number) => { const x = (sorted.length - 1) * p, lo = Math.floor(x); return sorted[lo]! + (sorted[Math.min(lo + 1, sorted.length - 1)]! - sorted[lo]!) * (x - lo); };
const category = (v: Cell) => String(v);
/** Bucket of one value: numeric bins (edges.length + 1, a value equal to an edge falls below it), categories, other, null. */
export function binOf(b: Bins, v: Cell): number {
  if (v === null || v === undefined) return b.counts.length - 1;
  if (b.kind === "numeric") return b.edges.filter(e => Number(v) > e).length;
  const i = b.categories.indexOf(category(v)); return i >= 0 ? i : b.categories.length;
}
export function countInto(b: Bins, values: Cell[]): number[] { const counts = new Array<number>(b.counts.length).fill(0); for (const v of values) counts[binOf(b, v)]!++; return counts; }
/** Reference bins from training values. Deciles are interpolated and de-duplicated, so a constant feature has one bin. */
export function referenceBins(name: string, kind: "numeric" | "categorical", values: Cell[]): Bins {
  if (kind === "numeric") {
    const sorted = values.filter((v): v is number => typeof v === "number").sort((a, b) => a - b);
    const edges = sorted.length ? [...new Set(DRIFT.quantiles.map(p => quantile(sorted, p)))] : [];
    const bins: Bins = { name, kind, edges, counts: new Array(edges.length + 2).fill(0) };
    return { ...bins, counts: countInto(bins, values) };
  }
  const frequency = new Map<string, number>(); for (const v of values) if (v !== null) frequency.set(category(v), (frequency.get(category(v)) ?? 0) + 1);
  const categories = [...frequency.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, DRIFT.topCategories).map(([c]) => c);
  const bins: Bins = { name, kind, categories, counts: new Array(categories.length + 2).fill(0) };
  return { ...bins, counts: countInto(bins, values) };
}
const proportions = (counts: number[]) => { const total = counts.reduce((a, b) => a + b, 0); return counts.map(c => total ? c / total : 0); };
/** Population Stability Index, sum((o - r) * ln(o / r)) over buckets, with proportions floored at DRIFT.epsilon. */
export function psi(reference: number[], observed: number[]): number {
  const r = proportions(reference).map(p => Math.max(p, DRIFT.epsilon)), o = proportions(observed).map(p => Math.max(p, DRIFT.epsilon));
  return r.reduce((sum, ri, i) => sum + (o[i]! - ri) * Math.log(o[i]! / ri), 0);
}
/** Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint), on unsmoothed proportions. */
export function jensenShannon(reference: number[], observed: number[]): number {
  const r = proportions(reference), o = proportions(observed), m = r.map((ri, i) => (ri + o[i]!) / 2);
  const kl = (p: number[]) => p.reduce((sum, pi, i) => pi > 0 ? sum + pi * Math.log2(pi / m[i]!) : sum, 0);
  return Math.max(0, Math.min(1, (kl(r) + kl(o)) / 2));
}
export function driftStatus(value: number, t: { psiModerate: number; psiSignificant: number }): DriftStatus { return value >= t.psiSignificant ? "significant" : value >= t.psiModerate ? "moderate" : "stable"; }
const RANK: Record<DriftStatus, number> = { stable: 0, moderate: 1, significant: 2, insufficient_data: -1 };

/** Tabular features of a model, with their bin kind. Text, timestamp and media features are not profiled. */
function profiled(snapshot: Snapshot) {
  const types = new Map(snapshot.dataset.schema.columns.map(c => [c.name, c.type]));
  return snapshot.pipeline.definition.features.filter(f => isTabularType(types.get(f)!)).map(f => ({ name: f, kind: types.get(f) === "number" ? "numeric" as const : "categorical" as const }));
}
const outputKind = (snapshot: Snapshot) => { const task = getModel(snapshot.backend, snapshot.algorithm).taskType; return task === "regression" ? "numeric" as const : task === "classification" ? "categorical" as const : null; };

export function buildReference(snapshot: Snapshot, rows: Row[], train: number[]): ReferenceProfile {
  const sample = train.map(i => rows[i]!), target = snapshot.dataset.schema.target, kind = outputKind(snapshot);
  return { version: 1, rows: sample.length, features: profiled(snapshot).map(f => referenceBins(f.name, f.kind, sample.map(r => r[f.name] ?? null))),
    output: kind && target ? referenceBins("output", kind, sample.map(r => r[target] ?? null)) : null };
}
export function observedProfile(reference: ReferenceProfile, rows: Row[], outputs: unknown[] | null) {
  return { features: Object.fromEntries(reference.features.map(b => [b.name, countInto(b, rows.map(r => r[b.name] ?? null))])),
    output: reference.output && outputs ? countInto(reference.output, outputs as Cell[]) : null };
}

/**
 * Durable, bounded model monitoring. References and per-prediction bin counts live in PostgreSQL; aggregation over a
 * window happens in SQL, so memory does not grow with traffic. Nothing here retrains, promotes or deactivates anything.
 */
export class MlMonitoringService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore) {}

  /** The version's reference profile, built from its training partition on first use. Null when it cannot be built. */
  async reference(db: pg.Pool | pg.PoolClient, tenantId: string, modelVersionId: string): Promise<ReferenceProfile | null> {
    const existing = (await db.query("SELECT profile FROM ml_drift_references WHERE tenant_id=$1 AND model_version_id=$2", [tenantId, modelVersionId])).rows[0];
    if (existing) return existing.profile;
    const t = (await db.query("SELECT t.snapshot,t.split_indices FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE m.tenant_id=$1 AND m.id=$2", [tenantId, modelVersionId])).rows[0];
    if (!t || !t.split_indices.train?.length) return null;
    const snapshot = TrainingSnapshot.parse(t.snapshot);
    // Nothing to profile (text, timestamp or media features and no label output): decided without reading the dataset.
    if (!profiled(snapshot).length && !outputKind(snapshot)) return null;
    // Only computation is guarded here: a failed SQL statement would abort the caller's transaction, so none is caught.
    let profile: ReferenceProfile;
    try { profile = buildReference(snapshot, JSON.parse((await this.artifacts.get(tenantId, snapshot.dataset.content)).toString("utf8")), t.split_indices.train); } catch { return null; }
    if (!profile.features.length && !profile.output) return null;
    await db.query("INSERT INTO ml_drift_references(tenant_id,model_version_id,profile,rows) VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING", [tenantId, modelVersionId, JSON.stringify(profile), profile.rows]);
    return (await db.query("SELECT profile FROM ml_drift_references WHERE tenant_id=$1 AND model_version_id=$2", [tenantId, modelVersionId])).rows[0].profile;
  }

  /** Records the bin counts of one completed prediction (called inside the prediction's final transaction). */
  async record(c: pg.PoolClient, tenantId: string, prediction: { id: string; endpoint_id: string; model_version_id: string; count: number }, reference: ReferenceProfile, rows: Row[], outputs: unknown[] | null) {
    await c.query("INSERT INTO ml_prediction_profiles(tenant_id,prediction_id,endpoint_id,model_version_id,rows,profile) VALUES($1,$2,$3,$4,$5,$6)", [tenantId, prediction.id, prediction.endpoint_id, prediction.model_version_id, prediction.count, JSON.stringify(observedProfile(reference, rows, outputs))]);
  }

  /** Ground-truth labels for a completed classification or regression prediction, one per row, typed like the target. */
  async feedback(principal: Principal, predictionId: string, raw: unknown) {
    authorizeMl(principal, true); const { labels } = FeedbackInput.parse(raw);
    return transaction(this.pool, async c => {
      const p = (await c.query("SELECT p.status,p.count,t.snapshot FROM ml_predictions p JOIN ml_model_versions m ON m.tenant_id=p.tenant_id AND m.id=p.model_version_id JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE p.tenant_id=$1 AND p.id=$2", [principal.tenantId, predictionId])).rows[0];
      if (!p) throw new NotFoundError("Prediction not found");
      const snapshot = TrainingSnapshot.parse(p.snapshot), kind = outputKind(snapshot);
      if (p.status !== "COMPLETED" || !kind) throw new ConflictError("Feedback needs a completed classification or regression prediction");
      if (labels.length !== p.count) throw new ConflictError(`Feedback needs exactly ${p.count} labels`);
      const type = snapshot.dataset.schema.columns.find(col => col.name === snapshot.dataset.schema.target)!.type;
      if (labels.some(l => typeof l !== (type === "number" ? "number" : type === "boolean" ? "boolean" : "string"))) throw new ConflictError("Labels must have the target column's type");
      const r = await c.query("INSERT INTO ml_prediction_feedback(tenant_id,prediction_id,labels,created_by) VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING RETURNING *", [principal.tenantId, predictionId, JSON.stringify(labels), principal.userId]);
      if (!r.rowCount) throw new ConflictError("Feedback was already recorded for this prediction");
      await audit(c, principal.tenantId, principal.userId, "ml.prediction_feedback_recorded", { predictionId }); return r.rows[0];
    });
  }

  /** Request, latency, failure, output and ground-truth summary of a deployment's endpoints, per model version. */
  async monitor(principal: Principal, deploymentId: string, raw: unknown) {
    authorizeMl(principal); const window = await this.window(WindowInput.parse(raw));
    const endpoints = await this.endpoints(principal.tenantId, deploymentId);
    const args = [principal.tenantId, endpoints, window.from, window.to];
    const base = (await this.pool.query(`SELECT model_version_id,count(*)::int AS requests,COALESCE(sum(count),0)::bigint AS rows,count(*) FILTER(WHERE status='COMPLETED')::int AS completed,
        count(*) FILTER(WHERE status='FAILED')::int AS failures,count(*) FILTER(WHERE status='PENDING')::int AS pending,avg(latency_ms)::float8 AS average,
        percentile_cont(.5) WITHIN GROUP (ORDER BY latency_ms) AS p50,percentile_cont(.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,max(latency_ms)::float8 AS max
      FROM ml_predictions WHERE tenant_id=$1 AND endpoint_id=ANY($2::uuid[]) AND created_at>=$3 AND created_at<$4 GROUP BY model_version_id ORDER BY model_version_id`, args)).rows;
    const categories = (await this.pool.query("SELECT model_version_id,failure->>'category' AS category,count(*)::int AS n FROM ml_predictions WHERE tenant_id=$1 AND endpoint_id=ANY($2::uuid[]) AND created_at>=$3 AND created_at<$4 AND status='FAILED' GROUP BY 1,2", args)).rows;
    const numeric = (await this.pool.query(`SELECT p.model_version_id,count(*)::int AS n,avg(v::text::float8) AS mean,stddev_pop(v::text::float8) AS std,min(v::text::float8) AS min,max(v::text::float8) AS max
      FROM ml_predictions p,jsonb_array_elements(p.output) v WHERE p.tenant_id=$1 AND p.endpoint_id=ANY($2::uuid[]) AND p.created_at>=$3 AND p.created_at<$4 AND p.status='COMPLETED' AND jsonb_typeof(p.output)='array' AND jsonb_typeof(v)='number' GROUP BY 1`, args)).rows;
    const labels = (await this.pool.query(`SELECT model_version_id,label,n FROM (SELECT p.model_version_id,v #>> '{}' AS label,count(*)::int AS n,row_number() OVER (PARTITION BY p.model_version_id ORDER BY count(*) DESC,v #>> '{}') AS rank
      FROM ml_predictions p,jsonb_array_elements(p.output) v WHERE p.tenant_id=$1 AND p.endpoint_id=ANY($2::uuid[]) AND p.created_at>=$3 AND p.created_at<$4 AND p.status='COMPLETED' AND jsonb_typeof(p.output)='array' AND jsonb_typeof(v) IN ('string','boolean','number') GROUP BY 1,2) x WHERE rank<=20`, args)).rows;
    const truth = (await this.pool.query(`SELECT p.model_version_id,count(*)::int AS n,avg((o.value=l.value)::int)::float8 AS accuracy,
        avg(abs(o.value::text::float8-l.value::text::float8)) FILTER(WHERE jsonb_typeof(o.value)='number' AND jsonb_typeof(l.value)='number') AS mae,
        sqrt(avg((o.value::text::float8-l.value::text::float8)^2) FILTER(WHERE jsonb_typeof(o.value)='number' AND jsonb_typeof(l.value)='number')) AS rmse
      FROM ml_prediction_feedback f JOIN ml_predictions p ON p.tenant_id=f.tenant_id AND p.id=f.prediction_id,
        jsonb_array_elements(p.output) WITH ORDINALITY o(value,i) JOIN jsonb_array_elements(f.labels) WITH ORDINALITY l(value,j) ON i=j
      WHERE p.tenant_id=$1 AND p.endpoint_id=ANY($2::uuid[]) AND p.created_at>=$3 AND p.created_at<$4 GROUP BY 1`, args)).rows;
    const tasks = new Map((await this.pool.query("SELECT m.id,t.snapshot FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE m.tenant_id=$1 AND m.id=ANY($2::uuid[])", [principal.tenantId, base.map(b => b.model_version_id)])).rows.map(r => [r.id, getModel(r.snapshot.backend, r.snapshot.algorithm).taskType]));
    const latest = (await this.pool.query("SELECT id,model_version_id,status,window_start,window_end,observed_rows,created_at FROM ml_drift_evaluations WHERE tenant_id=$1 AND deployment_id=$2 ORDER BY created_at DESC LIMIT 1", [principal.tenantId, deploymentId])).rows[0] ?? null;
    return {
      window, latestDrift: latest,
      versions: base.map(b => {
        const task = tasks.get(b.model_version_id), n = numeric.find(x => x.model_version_id === b.model_version_id), g = truth.find(x => x.model_version_id === b.model_version_id);
        return { modelVersionId: b.model_version_id, task, requests: b.requests, rows: Number(b.rows), completed: b.completed, failures: b.failures, pending: b.pending, errorRate: b.requests ? b.failures / b.requests : 0,
          latencyMs: { average: b.average, p50: b.p50, p95: b.p95, max: b.max },
          failuresByCategory: Object.fromEntries(categories.filter(x => x.model_version_id === b.model_version_id).map(x => [x.category ?? "unknown", x.n])),
          outputs: task === "regression" ? (n ? { count: n.n, mean: n.mean, std: n.std, min: n.min, max: n.max } : null) : task === "classification" ? Object.fromEntries(labels.filter(x => x.model_version_id === b.model_version_id).map(x => [x.label, x.n])) : null,
          groundTruth: g ? (task === "regression" ? { labelledRows: g.n, mae: g.mae, rmse: g.rmse } : { labelledRows: g.n, accuracy: g.accuracy }) : null };
      }),
    };
  }

  /**
   * Compares the observed bin counts of one model version, served by this deployment inside the window, with the version's
   * reference. Persisted with its inputs; below `minSamples` observed rows the status is insufficient_data.
   */
  async evaluateDrift(principal: Principal, deploymentId: string, raw: unknown) {
    authorizeMl(principal, true); const input = DriftInput.parse(raw), window = await this.window(input);
    const endpoints = await this.endpoints(principal.tenantId, deploymentId);
    // Only a version one of the deployment's revisions pins; by default the serving one, else the one that served last.
    const version = (await this.pool.query(`SELECT r.model_version_id FROM ml_deployments d JOIN ml_deployment_revisions r ON r.tenant_id=d.tenant_id AND r.deployment_id=d.id
      LEFT JOIN LATERAL (SELECT max(e.created_at) AS served FROM ml_deployment_events e WHERE e.tenant_id=d.tenant_id AND e.deployment_id=d.id AND e.revision=r.revision AND e.event IN ('activated','rolled_back')) s ON true
      WHERE d.tenant_id=$1 AND d.id=$2 AND ($3::uuid IS NULL OR r.model_version_id=$3) AND ($3::uuid IS NOT NULL OR s.served IS NOT NULL)
      ORDER BY (r.revision=d.active_revision) DESC NULLS LAST,s.served DESC NULLS LAST LIMIT 1`, [principal.tenantId, deploymentId, input.modelVersionId ?? null])).rows[0]?.model_version_id;
    if (!version) throw new ConflictError(input.modelVersionId ? "The model version is not part of this deployment" : "The deployment has not served a model version yet");
    const reference = await this.reference(this.pool, principal.tenantId, version);
    if (!reference) throw new ConflictError("No drift reference can be built for this model version");
    const args = [principal.tenantId, version, endpoints, window.from, window.to];
    const observedRows = Number((await this.pool.query("SELECT COALESCE(sum(rows),0)::bigint AS n FROM ml_prediction_profiles WHERE tenant_id=$1 AND model_version_id=$2 AND endpoint_id=ANY($3::uuid[]) AND created_at>=$4 AND created_at<$5", args)).rows[0].n);
    const sums = (path: string) => this.pool.query(`SELECT f.key AS name,(b.i-1)::int AS bin,sum(b.v::bigint)::bigint AS n FROM ml_prediction_profiles p,jsonb_each(p.profile->'${path}') f,jsonb_array_elements_text(f.value) WITH ORDINALITY b(v,i)
      WHERE p.tenant_id=$1 AND p.model_version_id=$2 AND p.endpoint_id=ANY($3::uuid[]) AND p.created_at>=$4 AND p.created_at<$5 AND jsonb_typeof(f.value)='array' GROUP BY 1,2`, args);
    const features = (await sums("features")).rows;
    const output = (await this.pool.query(`SELECT (b.i-1)::int AS bin,sum(b.v::bigint)::bigint AS n FROM ml_prediction_profiles p,jsonb_array_elements_text(p.profile->'output') WITH ORDINALITY b(v,i)
      WHERE p.tenant_id=$1 AND p.model_version_id=$2 AND p.endpoint_id=ANY($3::uuid[]) AND p.created_at>=$4 AND p.created_at<$5 AND jsonb_typeof(p.profile->'output')='array' GROUP BY 1`, args)).rows;
    const compare = (b: Bins, rows: { bin: number; n: string }[]) => {
      const observed = new Array<number>(b.counts.length).fill(0); for (const r of rows) if (r.bin < observed.length) observed[r.bin] = Number(r.n);
      const value = psi(b.counts, observed), total = observed.reduce((a, c) => a + c, 0);
      return { name: b.name, kind: b.kind, ...(b.kind === "numeric" ? { edges: b.edges } : { categories: b.categories }), reference: b.counts, observed, psi: value, jensenShannon: jensenShannon(b.counts, observed),
        nullRate: { reference: b.counts.at(-1)! / Math.max(1, b.counts.reduce((a, c) => a + c, 0)), observed: total ? observed.at(-1)! / total : null }, status: driftStatus(value, input.thresholds) };
    };
    const featureResults = reference.features.map(b => compare(b, features.filter(f => f.name === b.name)));
    const outputResult = reference.output ? compare(reference.output, output) : null;
    const worst = [...featureResults, ...(outputResult ? [outputResult] : [])].reduce<DriftStatus>((s, r) => RANK[r.status] > RANK[s] ? r.status : s, "stable");
    const status: DriftStatus = observedRows < input.minSamples ? "insufficient_data" : worst;
    return transaction(this.pool, async c => {
      const r = await c.query("INSERT INTO ml_drift_evaluations(tenant_id,deployment_id,model_version_id,window_start,window_end,thresholds,observed_rows,reference_rows,status,features,output,created_by) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING *",
        [principal.tenantId, deploymentId, version, window.from, window.to, JSON.stringify({ ...input.thresholds, minSamples: input.minSamples, epsilon: DRIFT.epsilon }), observedRows, reference.rows, status, JSON.stringify(featureResults), outputResult ? JSON.stringify(outputResult) : null, principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.drift_evaluated", { deploymentId, evaluationId: r.rows[0].id, status }); return r.rows[0];
    });
  }

  async evaluations(principal: Principal, deploymentId: string) {
    authorizeMl(principal); await this.endpoints(principal.tenantId, deploymentId);
    return (await this.pool.query("SELECT * FROM ml_drift_evaluations WHERE tenant_id=$1 AND deployment_id=$2 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, deploymentId])).rows;
  }

  /** Every endpoint the deployment has served through (from its activation history). */
  private async endpoints(tenantId: string, deploymentId: string): Promise<string[]> {
    if (!(await this.pool.query("SELECT 1 FROM ml_deployments WHERE tenant_id=$1 AND id=$2", [tenantId, deploymentId])).rowCount) throw new NotFoundError("Deployment not found");
    return (await this.pool.query("SELECT DISTINCT endpoint_id FROM ml_deployment_events WHERE tenant_id=$1 AND deployment_id=$2 AND endpoint_id IS NOT NULL", [tenantId, deploymentId])).rows.map(r => r.endpoint_id);
  }
  /** Explicit, bounded windows: default the last 24 hours, at most DRIFT.maxWindowDays. */
  private async window(input: { from?: string | undefined; to?: string | undefined }) {
    // DB timestamps have microseconds; round its clock up to a replayable millisecond exclusive bound.
    const to = input.to ? new Date(input.to) : (await this.pool.query<{ bound: Date }>("SELECT date_trunc('milliseconds',clock_timestamp()) + interval '1 millisecond' AS bound")).rows[0]!.bound;
    const from = input.from ? new Date(input.from) : new Date(to.getTime() - 86_400_000);
    if (!(from < to)) throw new ConflictError("Window start must precede its end");
    if (to.getTime() - from.getTime() > DRIFT.maxWindowDays * 86_400_000) throw new ConflictError(`Windows are limited to ${DRIFT.maxWindowDays} days`);
    return { from: from.toISOString(), to: to.toISOString() };
  }
}
