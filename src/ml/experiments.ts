import type pg from "pg";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { RunRepository } from "../db/repositories.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { CvSpec, JsonObject, TrainingSnapshot, cohortKey, type MetricValue } from "./domain.js";
import { assertMetricsSupported } from "./metrics.js";
import { getModel, validateHyperparameters, type TaskType } from "./model-catalog.js";
import { audit, MlTrainingService, authorizeMl, transaction } from "./persistence.js";
import { AutoMlInput, planAutoMl } from "./automl.js";
import { generateCandidates, SearchSpace } from "./search.js";
import { rankCandidates, type RankableCandidate, type Ranking, type RankingConfigType } from "./ranking.js";

const Name = z.string().trim().min(1).max(120);
const TERMINAL = ["COMPLETED", "FAILED", "CANCELLED"];
export type OrchestrationKind = "benchmark" | "search" | "automl";
export type OrchestrationStatus = "RUNNING" | "COMPLETED" | "PARTIALLY_COMPLETED" | "FAILED" | "CANCELLED";

const SplitInput = z.object({ train: z.number().gt(0).lt(1), validation: z.number().gt(0).lt(1), test: z.number().gt(0).lt(1), stratify: z.boolean().optional() })
  .refine(x => Math.abs(x.train + x.validation + x.test - 1) < 1e-9, "Split fractions must sum to one");
export const EvaluationInput = z.object({ primaryMetric: Name, secondaryMetrics: z.array(Name).max(9).default([]), cv: CvSpec.optional() });
const Source = z.object({ revision: z.string().min(1).max(200), repository: z.string().max(500).optional(), dirty: z.boolean() }).nullable().default(null);
export const BaseInput = z.object({
  name: Name, experimentId: z.uuid().optional(), idempotencyKey: z.string().min(1).max(200).optional(),
  datasetVersionId: z.uuid(), featurePipelineId: z.uuid(), backend: Name.default("sklearn"), taskType: z.enum(["classification", "regression"]),
  seed: z.int().min(0).max(4_294_967_295), split: SplitInput.default({ train: .6, validation: .2, test: .2 }), evaluation: EvaluationInput, source: Source,
});
export const BenchmarkInput = BaseInput.extend({
  candidates: z.array(z.object({ algorithm: Name, hyperparameters: JsonObject.default({}), label: Name.optional() })).min(2).max(20),
});
export const SearchInput = BaseInput.extend({
  algorithm: Name, fixedHyperparameters: JsonObject.default({}), space: SearchSpace, strategy: z.enum(["grid", "random"]), maxCandidates: z.int().min(1).max(50),
});

/** What every candidate of one experiment shares; only algorithm and hyperparameters vary. */
export interface SharedSettings { datasetVersionId: string; featurePipelineId: string; backend: string; taskType: TaskType; seed: number; split: z.infer<typeof SplitInput>; evaluation: z.infer<typeof EvaluationInput>; source: z.infer<typeof Source> }
export interface CandidatePlan { label: string; algorithm: string; hyperparameters: Record<string, unknown>; origin: "listed" | "search" | "default" }
export interface Plan { kind: OrchestrationKind; parentId: string | null; experimentId: string; name: string; config: unknown; ranking: RankingConfigType; idempotencyKey: string | null; shared: SharedSettings; candidates: CandidatePlan[] }
interface Row { [column: string]: any }

export const rankingFor = (e: z.infer<typeof EvaluationInput>): RankingConfigType => ({ partition: e.cv ? "cv" : "validation", primaryMetric: e.primaryMetric, secondaryMetrics: e.secondaryMetrics, tolerance: 0 });

export class MlExperimentService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>) {}
  private get training() { return new MlTrainingService(this.pool, this.artifacts); }

  async benchmark(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = BenchmarkInput.parse(raw);
    const labels = input.candidates.map(c => c.label ?? c.algorithm);
    if (new Set(labels).size !== labels.length) throw new ConflictError("Candidate labels must be unique; set `label` to compare one algorithm several times");
    const candidates = input.candidates.map((c, i): CandidatePlan => ({ label: labels[i]!, algorithm: c.algorithm, hyperparameters: c.hyperparameters, origin: "listed" }));
    return this.create(principal, "benchmark", input, { candidates: input.candidates }, candidates);
  }

  /** Hyperparameter search: Meticule generates the candidates; each becomes an ordinary child training job. */
  async search(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = SearchInput.parse(raw);
    const model = getModel(input.backend, input.algorithm);
    const points = generateCandidates(model, input.space, input.fixedHyperparameters, input.strategy, input.maxCandidates, input.seed);
    const candidates = points.map((point, i): CandidatePlan => ({ label: `${input.algorithm}#${i + 1}`, algorithm: input.algorithm, hyperparameters: { ...input.fixedHyperparameters, ...point }, origin: "search" }));
    const { space, strategy, maxCandidates, fixedHyperparameters } = input;
    return this.create(principal, "search", input, { algorithm: input.algorithm, space, strategy, maxCandidates, fixedHyperparameters }, candidates);
  }

  /**
   * Bounded AutoML V1: validate, plan deterministically from the input and budget, then create one benchmark child
   * (default configurations) and up to two search children, all under an AutoML parent. It only recommends.
   */
  async automl(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = AutoMlInput.parse(raw);
    return transaction(this.pool, async c => {
      const dataset = await c.query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.datasetVersionId]);
      if (!dataset.rowCount) throw new NotFoundError("Dataset version not found");
      const doc = dataset.rows[0].document as { schema: { columns: { name: string; type: "number" | "string" | "boolean" }[]; target: string }; rowCount: number };
      const plan = planAutoMl(input, doc.schema, doc.rowCount);
      const { idempotencyKey: _key, name: _name, experimentId: _experiment, ...definition } = input;
      const configHash = canonicalJsonHash({ kind: "automl", definition, plan });
      if (input.idempotencyKey) {
        const existing = await c.query("SELECT id,config_hash FROM ml_orchestrations WHERE tenant_id=$1 AND idempotency_key=$2", [principal.tenantId, input.idempotencyKey]);
        if (existing.rowCount) {
          if (existing.rows[0].config_hash !== configHash) throw new ConflictError("Idempotency key was already used with a different definition");
          return this.read(principal.tenantId, existing.rows[0].id, c);
        }
      }
      const pipeline = await this.training.pipelineIn(c, principal, { name: `automl: ${input.name}`.slice(0, 120), definition: { features: plan.features, steps: plan.steps } });
      const experimentId = input.experimentId ?? (await c.query("INSERT INTO ml_experiments(tenant_id,name,created_by) VALUES($1,$2,$3) RETURNING id", [principal.tenantId, input.name, principal.userId])).rows[0].id as string;
      const classification = input.taskType === "classification";
      const evaluation = { primaryMetric: input.metric, secondaryMetrics: input.secondaryMetrics, cv: { strategy: classification ? "stratified_kfold" as const : "kfold" as const, folds: input.cvFolds } };
      const shared: SharedSettings = { datasetVersionId: input.datasetVersionId, featurePipelineId: pipeline.id, backend: input.backend, taskType: input.taskType, seed: input.seed, split: { train: .6, validation: .2, test: .2, stratify: classification }, evaluation, source: input.source };
      const parent = await c.query("INSERT INTO ml_orchestrations(tenant_id,experiment_id,kind,name,config,config_hash,ranking_config,idempotency_key,created_by) VALUES($1,$2,'automl',$3,$4,$5,$6,$7,$8) RETURNING id",
        [principal.tenantId, experimentId, input.name, JSON.stringify({ definition, plan, shared }), configHash, JSON.stringify(rankingFor(evaluation)), input.idempotencyKey ?? null, principal.userId]);
      const parentId = parent.rows[0].id as string;
      const child = (kind: OrchestrationKind, name: string, candidates: CandidatePlan[], config: unknown) =>
        this.insert(c, principal, { kind, parentId, experimentId, name, config: { definition: config, shared, candidates }, ranking: rankingFor(evaluation), idempotencyKey: null, shared, candidates }, canonicalJsonHash({ kind, config, shared, candidates }));
      await child("benchmark", `${input.name}: default configurations`, plan.benchmark.map(algorithm => ({ label: algorithm, algorithm, hyperparameters: {}, origin: "default" as const })), { algorithms: plan.benchmark });
      for (const search of plan.searches) {
        const points = generateCandidates(getModel(input.backend, search.algorithm), search.space, {}, "random", search.maxCandidates, input.seed);
        await child("search", `${input.name}: ${search.algorithm} search`, points.map((p, i) => ({ label: `${search.algorithm}#${i + 1}`, algorithm: search.algorithm, hyperparameters: p, origin: "search" as const })), { algorithm: search.algorithm, space: search.space, strategy: "random", maxCandidates: search.maxCandidates });
      }
      await audit(c, principal.tenantId, principal.userId, "ml.automl_created", { orchestrationId: parentId, candidates: plan.totalCandidates });
      return this.read(principal.tenantId, parentId, c);
    });
  }

  /** Validates, then atomically creates the orchestration row and one ordinary durable training job per candidate. */
  protected async create(principal: Principal, kind: OrchestrationKind, input: z.infer<typeof BaseInput>, definition: unknown, candidates: CandidatePlan[]) {
    const shared: SharedSettings = { datasetVersionId: input.datasetVersionId, featurePipelineId: input.featurePipelineId, backend: input.backend, taskType: input.taskType, seed: input.seed, split: input.split, evaluation: input.evaluation, source: input.source };
    const configHash = canonicalJsonHash({ kind, definition, shared, candidates });
    return transaction(this.pool, async c => {
      const existing = input.idempotencyKey ? await c.query("SELECT id,config_hash FROM ml_orchestrations WHERE tenant_id=$1 AND idempotency_key=$2", [principal.tenantId, input.idempotencyKey]) : null;
      if (existing?.rowCount) {
        if (existing.rows[0].config_hash !== configHash) throw new ConflictError("Idempotency key was already used with a different definition");
        return this.read(principal.tenantId, existing.rows[0].id, c);
      }
      const experimentId = input.experimentId ?? (await c.query("INSERT INTO ml_experiments(tenant_id,name,created_by) VALUES($1,$2,$3) RETURNING id", [principal.tenantId, input.name, principal.userId])).rows[0].id as string;
      const id = await this.insert(c, principal, { kind, parentId: null, experimentId, name: input.name, config: { definition, shared, candidates }, ranking: rankingFor(input.evaluation), idempotencyKey: input.idempotencyKey ?? null, shared, candidates }, configHash);
      return this.read(principal.tenantId, id, c);
    });
  }
  /** Shared by every kind: validate each candidate against the catalog and backend, then queue ordinary jobs. */
  async insert(c: pg.PoolClient, principal: Principal, plan: Plan, configHash: string): Promise<string> {
    const { shared } = plan;
    const backend = this.backends.get(shared.backend); if (!backend) throw new ConflictError("Requested ML backend is not configured");
    const dataset = await c.query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, shared.datasetVersionId]);
    const pipeline = await c.query("SELECT document FROM ml_feature_pipelines WHERE tenant_id=$1 AND id=$2", [principal.tenantId, shared.featurePipelineId]);
    if (!dataset.rowCount || !pipeline.rowCount) throw new NotFoundError("Dataset version or pipeline not found");
    const experiment = await c.query("SELECT id FROM ml_experiments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, plan.experimentId]);
    if (!experiment.rowCount) throw new NotFoundError("Experiment not found");
    const schema = dataset.rows[0].document.schema as { columns: { name: string; type: string }[]; target: string };
    if (shared.taskType === "regression" && schema.columns.find(col => col.name === schema.target)?.type !== "number") throw new ConflictError("Regression requires a numeric target");
    const metrics = [shared.evaluation.primaryMetric, ...shared.evaluation.secondaryMetrics];
    if (new Set(metrics).size !== metrics.length) throw new ConflictError("Primary and secondary metrics must be distinct");
    const split = { id: randomUUID(), strategy: "random" as const, ...shared.split, stratify: shared.split.stratify ?? shared.taskType === "classification" };
    const specs = plan.candidates.map(candidate => {
      const model = getModel(shared.backend, candidate.algorithm);
      if (model.taskType !== shared.taskType) throw new ConflictError(`${candidate.algorithm} is a ${model.taskType} model`);
      assertMetricsSupported(model, metrics);
      const spec = { datasetVersionId: shared.datasetVersionId, featurePipelineId: shared.featurePipelineId, backend: shared.backend, algorithm: candidate.algorithm, hyperparameters: validateHyperparameters(model, candidate.hyperparameters) as Record<string, never>, seed: shared.seed, split, evaluation: { metrics, ...(shared.evaluation.cv ? { cv: shared.evaluation.cv } : {}) }, source: shared.source };
      backend.validate(TrainingSnapshot.parse({ ...spec, dataset: dataset.rows[0].document, pipeline: pipeline.rows[0].document }));
      return spec;
    });
    const o = await c.query("INSERT INTO ml_orchestrations(tenant_id,experiment_id,parent_id,kind,name,config,config_hash,ranking_config,idempotency_key,created_by) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING id",
      [principal.tenantId, plan.experimentId, plan.parentId, plan.kind, plan.name, JSON.stringify(plan.config), configHash, JSON.stringify(plan.ranking), plan.idempotencyKey, principal.userId]);
    const id = o.rows[0].id as string;
    for (const [ordinal, spec] of specs.entries()) {
      const job = await this.training.queueIn(c, principal, { experimentId: plan.experimentId, spec: spec as never });
      const candidate = plan.candidates[ordinal]!;
      await c.query("INSERT INTO ml_orchestration_members(tenant_id,orchestration_id,ordinal,job_id,label,candidate) VALUES($1,$2,$3,$4,$5,$6)", [principal.tenantId, id, ordinal, job.id, candidate.label, JSON.stringify(candidate)]);
    }
    await audit(c, principal.tenantId, principal.userId, `ml.${plan.kind}_created`, { orchestrationId: id, candidates: specs.length });
    return id;
  }

  async list(principal: Principal, kind: OrchestrationKind) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT id,experiment_id,parent_id,kind,name,status,created_at,finalized_at,cancel_requested_at FROM ml_orchestrations WHERE tenant_id=$1 AND kind=$2 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, kind])).rows;
  }

  /** Inspect. Managers also persist the final outcome the first time they observe all candidates terminal. */
  async get(principal: Principal, id: string, kind?: OrchestrationKind) {
    authorizeMl(principal);
    let view = await this.read(principal.tenantId, id, this.pool, kind);
    if (view.orchestration.status === "RUNNING" && view.outcome.terminal && principal.roles.includes("tenant_admin")) { await this.refresh(principal.tenantId, id); view = await this.read(principal.tenantId, id, this.pool, kind); }
    return view;
  }
  async ranking(principal: Principal, id: string) { const v = await this.get(principal, id); return { status: v.orchestration.status, final: v.orchestration.status !== "RUNNING", ranking: v.ranking, recommendation: v.recommendation }; }

  /** Cancels every unfinished candidate through the canonical run cancellation; finished candidates keep their results. */
  async cancel(principal: Principal, id: string) {
    authorizeMl(principal, true);
    const ids = await transaction(this.pool, async c => {
      const o = await c.query("SELECT id,status FROM ml_orchestrations WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, id]);
      if (!o.rowCount) throw new NotFoundError("Experiment not found");
      if (o.rows[0].status !== "RUNNING") throw new ConflictError("Experiment is already finalized");
      const family = await this.family(c, principal.tenantId, id);
      await c.query("UPDATE ml_orchestrations SET cancel_requested_at=now() WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='RUNNING' AND cancel_requested_at IS NULL", [principal.tenantId, family]);
      await audit(c, principal.tenantId, principal.userId, "ml.experiment_cancel_requested", { orchestrationId: id });
      return family;
    });
    const runs = new RunRepository(this.pool);
    const open = await this.pool.query(`SELECT j.run_id FROM ml_orchestration_members m JOIN ml_training_jobs j ON j.tenant_id=m.tenant_id AND j.id=m.job_id
      WHERE m.tenant_id=$1 AND m.orchestration_id=ANY($2::uuid[]) AND j.status NOT IN ('COMPLETED','FAILED','CANCELLED') ORDER BY m.orchestration_id,m.ordinal`, [principal.tenantId, ids]);
    for (const row of open.rows) {
      try { await runs.requestCancellation(principal, row.run_id); } catch (error) { if (!(error instanceof ConflictError)) throw error; /* finished between the query and the request */ }
    }
    return this.get(principal, id);
  }

  /** Persists the outcome once every candidate is terminal. Idempotent and safe to call from workers or readers. */
  async refresh(tenantId: string, id: string): Promise<void> {
    const parent = await this.pool.query("SELECT kind,parent_id FROM ml_orchestrations WHERE tenant_id=$1 AND id=$2", [tenantId, id]);
    if (!parent.rowCount) return;
    if (parent.rows[0].kind === "automl") for (const child of (await this.pool.query("SELECT id FROM ml_orchestrations WHERE tenant_id=$1 AND parent_id=$2 ORDER BY kind,name,id", [tenantId, id])).rows) await this.refresh(tenantId, child.id);
    await transaction(this.pool, async c => {
      const o = (await c.query("SELECT * FROM ml_orchestrations WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [tenantId, id])).rows[0];
      if (!o || o.status !== "RUNNING") return;
      const family = await this.family(c, tenantId, id);
      const { candidates } = await this.load(c, tenantId, family);
      const outcome = this.outcome(o, candidates);
      if (!outcome.terminal) return;
      await c.query("UPDATE ml_orchestrations SET status=$1,ranking=$2,finalized_at=now() WHERE tenant_id=$3 AND id=$4", [outcome.status, JSON.stringify(outcome.ranking), tenantId, id]);
      await audit(c, tenantId, "system", "ml.experiment_finalized", { orchestrationId: id, status: outcome.status, counts: outcome.counts }, null, "worker");
    });
  }
  /** Worker hook: a job reached a terminal state, so any experiment containing it may now be final. Never throws. */
  async onJobTerminal(runId: string): Promise<void> {
    try {
      const owners = await this.pool.query("SELECT m.tenant_id,m.orchestration_id,o.parent_id FROM ml_orchestration_members m JOIN ml_training_jobs j ON j.tenant_id=m.tenant_id AND j.id=m.job_id JOIN ml_orchestrations o ON o.tenant_id=m.tenant_id AND o.id=m.orchestration_id WHERE j.run_id=$1", [runId]);
      for (const owner of owners.rows) await this.refresh(owner.tenant_id, owner.parent_id ?? owner.orchestration_id);
    } catch { /* the next reader or terminal job re-derives the same outcome */ }
  }

  private async family(c: pg.PoolClient | pg.Pool, tenantId: string, id: string): Promise<string[]> {
    return (await c.query("SELECT id FROM ml_orchestrations WHERE tenant_id=$1 AND (id=$2 OR parent_id=$2) ORDER BY (id=$2) DESC,kind,name,id", [tenantId, id])).rows.map(r => r.id as string);
  }

  private async load(db: pg.PoolClient | pg.Pool, tenantId: string, orchestrationIds: string[]) {
    const rows = (await db.query(`SELECT m.orchestration_id,m.ordinal,m.label,m.candidate,j.id AS job_id,j.run_id,j.status AS job_status,r.status AS runtime_status,t.id AS training_run_id,t.attempt,
        t.performance,t.split_indices,t.snapshot,t.resolved_hyperparameters,t.failure,a.id AS artifact_id,(a.content->>'size')::bigint AS artifact_bytes
      FROM ml_orchestration_members m JOIN ml_orchestrations o ON o.tenant_id=m.tenant_id AND o.id=m.orchestration_id JOIN ml_training_jobs j ON j.tenant_id=m.tenant_id AND j.id=m.job_id JOIN runs r ON r.tenant_id=j.tenant_id AND r.id=j.run_id
      LEFT JOIN LATERAL (SELECT * FROM ml_training_runs x WHERE x.tenant_id=j.tenant_id AND x.job_id=j.id ORDER BY (x.status='COMPLETED') DESC,x.attempt DESC LIMIT 1) t ON true
      LEFT JOIN ml_artifacts a ON a.tenant_id=t.tenant_id AND a.training_run_id=t.id AND a.kind='model'
      WHERE m.tenant_id=$1 AND m.orchestration_id=ANY($2::uuid[]) ORDER BY o.kind,o.name,o.id,m.ordinal`, [tenantId, orchestrationIds])).rows as Row[];
    const done = rows.filter(r => r.job_status === "COMPLETED").map(r => r.training_run_id as string);
    const metrics = done.length ? (await db.query("SELECT training_run_id,name,partition,value,direction,fold,std FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=ANY($2::uuid[]) ORDER BY partition,name,fold", [tenantId, done])).rows as Row[] : [];
    const candidates = rows.map(r => ({
      orchestrationId: r.orchestration_id as string, ordinal: r.ordinal as number, label: r.label as string, algorithm: r.candidate.algorithm as string, origin: r.candidate.origin as string,
      requestedHyperparameters: r.candidate.hyperparameters as Record<string, unknown>, resolvedHyperparameters: (r.resolved_hyperparameters ?? null) as Record<string, unknown> | null,
      jobId: r.job_id as string, runId: r.run_id as string, status: r.job_status as string, runtimeStatus: r.runtime_status as string, trainingRunId: (r.training_run_id ?? null) as string | null, attempt: (r.attempt ?? null) as number | null,
      performance: (r.performance ?? {}) as Record<string, number>, artifactId: (r.artifact_id ?? null) as string | null, artifactBytes: r.artifact_bytes == null ? null : Number(r.artifact_bytes), failure: (r.failure ?? null) as unknown,
      snapshot: r.snapshot ?? null, splitIndices: r.split_indices ?? null,
      metrics: metrics.filter(m => m.training_run_id === r.training_run_id).map(m => ({ name: m.name, partition: m.partition, value: m.value, direction: m.direction, fold: m.fold, std: m.std }) as MetricValue),
    }));
    return { candidates };
  }

  private outcome(o: Row, candidates: Awaited<ReturnType<MlExperimentService["load"]>>["candidates"]) {
    const counts = { total: candidates.length, completed: 0, failed: 0, cancelled: 0, open: 0 };
    for (const c of candidates) { if (c.status === "COMPLETED") counts.completed++; else if (c.status === "FAILED") counts.failed++; else if (c.status === "CANCELLED") counts.cancelled++; else counts.open++; }
    const terminal = counts.open === 0;
    const successes = candidates.filter(c => c.status === "COMPLETED" && c.trainingRunId);
    const keys = new Set(successes.map(c => cohortKey(c.snapshot, c.splitIndices)));
    const rankable: RankableCandidate[] = successes.map(c => ({ jobId: c.jobId, trainingRunId: c.trainingRunId!, label: c.label, algorithm: c.algorithm, hyperparameters: (c.resolvedHyperparameters ?? c.requestedHyperparameters), metrics: c.metrics, performance: c.performance, artifactBytes: c.artifactBytes }));
    // Fairness gate: candidates that do not share data, features, split, seed and folds are never ranked against each other.
    const ranking: Ranking = keys.size > 1
      ? { config: o.ranking_config, tieBreakers: ["primary_std", "secondary_metrics", "fit_seconds", "training_run_id"], entries: [], unranked: successes.map(c => ({ jobId: c.jobId, label: c.label, reason: "candidates do not share one comparable cohort" })) }
      : rankCandidates(rankable, o.ranking_config);
    let status: OrchestrationStatus = "RUNNING";
    if (terminal) status = o.cancel_requested_at ? "CANCELLED" : counts.completed === 0 ? (counts.failed === 0 && counts.cancelled > 0 ? "CANCELLED" : "FAILED") : counts.failed + counts.cancelled === 0 ? "COMPLETED" : "PARTIALLY_COMPLETED";
    return { terminal, status, counts, ranking, cohortKeys: keys.size };
  }

  async read(tenantId: string, id: string, db: pg.PoolClient | pg.Pool = this.pool, kind?: OrchestrationKind) {
    const o = (await db.query("SELECT * FROM ml_orchestrations WHERE tenant_id=$1 AND id=$2", [tenantId, id])).rows[0];
    if (!o || (kind && o.kind !== kind)) throw new NotFoundError("Experiment not found");
    const family = await this.family(db, tenantId, id);
    const children = family.filter(x => x !== id);
    const { candidates } = await this.load(db, tenantId, family);
    const live = this.outcome(o, candidates);
    const persisted = o.status !== "RUNNING";
    const ranking: Ranking = persisted ? o.ranking : live.ranking;
    const top = ranking.entries[0] ?? null;
    return {
      orchestration: { id: o.id as string, kind: o.kind as OrchestrationKind, name: o.name as string, experimentId: o.experiment_id as string, parentId: (o.parent_id ?? null) as string | null, status: o.status as OrchestrationStatus, config: o.config, rankingConfig: o.ranking_config, cancelRequestedAt: o.cancel_requested_at as string | null, createdAt: o.created_at as string, finalizedAt: o.finalized_at as string | null },
      children: children.length ? (await db.query("SELECT id,kind,name,status FROM ml_orchestrations WHERE tenant_id=$1 AND id=ANY($2::uuid[]) ORDER BY kind,name,id", [tenantId, children])).rows : [],
      outcome: { terminal: persisted || live.terminal, status: persisted ? o.status : live.status, counts: live.counts },
      candidates: candidates.map(({ snapshot, splitIndices: _indices, ...c }) => ({ ...c, lineage: snapshot ? lineage(o, c, snapshot) : null })),
      ranking, final: persisted,
      recommendation: top ? { jobId: top.jobId, trainingRunId: top.trainingRunId, label: top.label, algorithm: top.algorithm, basis: `${ranking.config.primaryMetric} on ${ranking.config.partition}`, note: "Best candidate under this experiment's objective only. It is not registered, promoted or deployed." } : null,
    };
  }
}

function lineage(o: Row, c: { orchestrationId: string; jobId: string; trainingRunId: string | null; artifactId: string | null }, raw: unknown) {
  const s = TrainingSnapshot.parse(raw);
  return { experimentId: o.experiment_id as string, orchestrationId: c.orchestrationId, rootOrchestrationId: (o.parent_id ?? o.id) as string, jobId: c.jobId, trainingRunId: c.trainingRunId, datasetVersionId: s.datasetVersionId, featurePipelineId: s.featurePipelineId, backend: s.backend, algorithm: s.algorithm, hyperparameters: s.hyperparameters, seed: s.seed, split: s.split, cv: s.evaluation.cv ?? null, metrics: s.evaluation.metrics ?? null, artifactId: c.artifactId };
}
