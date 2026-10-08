import pg from "pg";
import { randomUUID } from "node:crypto";
import { Hono } from "hono";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { RunRepository } from "../../src/db/repositories.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlExperimentService } from "../../src/ml/experiments.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { registerMlRoutes } from "../../src/ml/routes.js";
import { FakeBackend } from "./ml-fake-backend.js";
import type { MlBackend } from "../../src/ml/backend.js";

const python = process.env.ML_TEST_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 12 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const viewer: Principal = { ...manager, roles: [] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

const input = (ds: { id: string }, pl: { id: string }, extra: Record<string, unknown> = {}) => ({
  name: "churn benchmark", datasetVersionId: ds.id, featurePipelineId: pl.id, taskType: "classification", seed: 5, evaluation: { primaryMetric: "f1", secondaryMetrics: ["accuracy"], cv: { strategy: "stratified_kfold", folds: 5 } },
  candidates: [{ algorithm: "logistic_regression" }, { algorithm: "random_forest_classifier" }, { algorithm: "gradient_boosting_classifier" }, { algorithm: "svc" }], ...extra });

async function fixture(backend: MlBackend = new FakeBackend()) {
  const backends = new Map([[backend.id, backend]]); const training = new MlTrainingService(pool, store);
  const dataset = await training.ingest(manager, { name: "labels", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "label", type: "string" }], target: "label" }, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, label: i % 2 ? "odd" : "even" })) });
  const pipeline = await training.pipeline(manager, { name: "numeric", definition: { features: ["x"], steps: [{ operation: "standard_scale" }] } });
  const processor = new TrainingProcessor(pool, store, backends);
  const worker = (id = "bench-worker") => new LifecycleWorker(pool, { workerId: id, leaseSeconds: 30, kind: "ml" }, processor);
  const experiments = new MlExperimentService(pool, store, backends), models = new MlModelService(pool, store, backends);
  const drain = async (w = worker()) => { while (await w.tick()); };
  return { backend, backends, training, dataset, pipeline, experiments, models, worker, drain };
}
const labels = (v: { ranking: { entries: { label: string }[] } }) => v.ranking.entries.map(e => e.label);

describe("benchmark experiments", () => {
  it("runs every candidate as its own durable job and ranks them deterministically", async () => {
    const f = await fixture(); const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline));
    expect(created.orchestration.status).toBe("RUNNING"); expect(created.candidates).toHaveLength(4); expect(new Set(created.candidates.map(c => c.runId)).size).toBe(4);
    expect((await pool.query("SELECT DISTINCT kind FROM runs WHERE id=ANY($1)", [created.candidates.map(c => c.runId)])).rows).toEqual([{ kind: "ml" }]);
    expect(created.recommendation).toBeNull();
    await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id, "benchmark");
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.final).toBe(true);
    expect(labels(done)).toEqual(["random_forest_classifier", "gradient_boosting_classifier", "logistic_regression", "svc"]);
    expect(done.ranking.config).toMatchObject({ partition: "cv", primaryMetric: "f1", secondaryMetrics: ["accuracy"] });
    expect(done.recommendation).toMatchObject({ label: "random_forest_classifier" }); expect(done.recommendation!.note).toMatch(/not registered, promoted or deployed/);
    const top = done.ranking.entries[0]!; expect(top.primary).toMatchObject({ name: "f1", value: .9, std: .01, direction: "higher" }); expect(top.fitSeconds).toBe(.5); expect(top.artifactBytes).toBeGreaterThan(0);
    for (const c of done.candidates) { expect(c.lineage).toMatchObject({ datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, seed: 5, cv: { strategy: "stratified_kfold", folds: 5 } }); expect(c.artifactId).not.toBeNull(); expect(c.metrics.filter(m => m.partition === "cv_fold")).toHaveLength(10); }
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_registry_entries")).rows[0].n).toBe(0);
    expect((await f.experiments.get(manager, created.orchestration.id)).ranking).toEqual(done.ranking);
    expect((await f.experiments.ranking(viewer, created.orchestration.id)).final).toBe(true);
  });
  it("keeps valid results through candidate failure and individual cancellation", async () => {
    const backend = new FakeBackend(); backend.fail.add("svc"); const f = await fixture(backend);
    const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline, { candidates: ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier", "svc", "k_neighbors_classifier"].map(algorithm => ({ algorithm })) }));
    await new RunRepository(pool).requestCancellation(manager, created.candidates[4]!.runId); await new RunRepository(pool).finalizeUnleasedCancellations();
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("PARTIALLY_COMPLETED"); expect(done.outcome.counts).toMatchObject({ completed: 3, failed: 1, cancelled: 1 });
    expect(labels(done)).toEqual(["random_forest_classifier", "gradient_boosting_classifier", "logistic_regression"]);
    expect(done.candidates.find(c => c.label === "svc")).toMatchObject({ status: "FAILED", metrics: [] }); expect(done.recommendation).not.toBeNull();
  });
  it("reports FAILED without a recommendation when no candidate succeeds", async () => {
    const backend = new FakeBackend(); ["logistic_regression", "svc"].forEach(a => backend.fail.add(a)); const f = await fixture(backend);
    const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline, { candidates: [{ algorithm: "logistic_regression" }, { algorithm: "svc" }] })); await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("FAILED"); expect(done.ranking.entries).toEqual([]); expect(done.recommendation).toBeNull();
  });
  it("propagates parent cancellation to queued candidates", async () => {
    const f = await fixture(); const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline));
    await f.experiments.cancel(manager, created.orchestration.id); await new RunRepository(pool).finalizeUnleasedCancellations();
    const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("CANCELLED"); expect(done.candidates.every(c => c.status === "CANCELLED")).toBe(true); expect(await f.worker().tick()).toBe(false);
    await expect(f.experiments.cancel(manager, created.orchestration.id)).rejects.toThrow(/already finalized/);
  });
  it("cancels a running candidate and keeps finished results ranked", async () => {
    // Candidates run in claim order (not list order), so some may finish before the hanging one starts.
    const backend = new FakeBackend(); backend.hang = "gradient_boosting_classifier"; const f = await fixture(backend);
    const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline));
    const running = new Promise<void>(r => { backend.started = r; }); const loop = f.drain(); await running;
    await f.experiments.cancel(manager, created.orchestration.id); await loop; await new RunRepository(pool).finalizeUnleasedCancellations();
    const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("CANCELLED"); expect(done.candidates.find(c => c.label === "gradient_boosting_classifier")!.status).toBe("CANCELLED");
    expect(done.outcome.counts.completed + done.outcome.counts.cancelled).toBe(4); expect(done.ranking.entries).toHaveLength(done.outcome.counts.completed);
    expect(done.ranking.entries.map(e => e.primary.value)).toEqual([...done.ranking.entries.map(e => e.primary.value)].sort((x, y) => y - x));
  });
  it("survives stale-worker recovery and still finalizes", async () => {
    const f = await fixture(); const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline)); const runs = new RunRepository(pool);
    const stale = (await runs.claimNext("dead", 30, "ml"))!; await runs.workerTransition(stale.id, "dead", "running");
    const attempt = await f.training.begin(stale.id, "dead");
    await pool.query("UPDATE runs SET lease_expires_at=now()-interval '1 second' WHERE id=$1", [stale.id]); expect(await runs.recoverExpired()).toBe(1);
    expect((await f.experiments.get(manager, created.orchestration.id)).orchestration.status).toBe("RUNNING");
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.candidates.find(c => c.runId === stale.id)!.attempt).toBe(2);
    expect((await pool.query("SELECT status FROM ml_training_runs WHERE id=$1", [attempt.id])).rows[0].status).toBe("FAILED");
  });
  it("gives the same ranking with one worker or several concurrent workers", async () => {
    const f = await fixture(); (f.backend as FakeBackend).delayMs = 40;
    const sequential = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline)); await f.drain();
    const concurrent = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline, { name: "concurrent" }));
    await Promise.all(["w1", "w2", "w3"].map(id => f.drain(f.worker(id))));
    const a = await f.experiments.get(manager, sequential.orchestration.id), b = await f.experiments.get(manager, concurrent.orchestration.id);
    expect(b.orchestration.status).toBe("COMPLETED"); expect(labels(b)).toEqual(labels(a));
    expect(b.ranking.entries.map(e => e.primary.value)).toEqual(a.ranking.entries.map(e => e.primary.value));
  });
  it("refuses to rank candidates that did not share one split", async () => {
    const backend = new FakeBackend(); backend.skewSplit.add("svc"); const f = await fixture(backend);
    const created = await f.experiments.benchmark(manager, input(f.dataset, f.pipeline)); await f.drain();
    const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.ranking.entries).toEqual([]); expect(done.ranking.unranked).toHaveLength(4); expect(done.recommendation).toBeNull();
  });
  it("validates definitions atomically and enforces tenant, role and idempotency rules", async () => {
    const f = await fixture(); const count = async () => [(await pool.query("SELECT count(*)::int AS n FROM ml_orchestrations")).rows[0].n, (await pool.query("SELECT count(*)::int AS n FROM ml_training_jobs")).rows[0].n];
    const bad = (extra: Record<string, unknown>) => f.experiments.benchmark(manager, input(f.dataset, f.pipeline, extra));
    await expect(bad({ candidates: [{ algorithm: "ridge" }, { algorithm: "svc" }] })).rejects.toThrow(/regression model/);
    await expect(bad({ candidates: [{ algorithm: "svc" }, { algorithm: "logistic_regression" }], evaluation: { primaryMetric: "log_loss", cv: { strategy: "kfold", folds: 3 } } })).rejects.toThrow(/cannot produce/);
    await expect(bad({ candidates: [{ algorithm: "svc" }, { algorithm: "svc" }] })).rejects.toThrow(/labels must be unique/);
    await expect(bad({ candidates: [{ algorithm: "svc" }, { algorithm: "magic_forest" }] })).rejects.toThrow(/Unsupported/);
    await expect(bad({ candidates: [{ algorithm: "svc" }, { algorithm: "logistic_regression", hyperparameters: { n_jobs: 8 } }] })).rejects.toThrow();
    await expect(bad({ candidates: [{ algorithm: "svc" }] })).rejects.toThrow();
    await expect(bad({ evaluation: { primaryMetric: "f1", secondaryMetrics: ["f1"] } })).rejects.toThrow(/distinct/);
    await expect(bad({ taskType: "regression", candidates: [{ algorithm: "ridge" }, { algorithm: "lasso" }], evaluation: { primaryMetric: "rmse" } })).rejects.toThrow(/numeric target/);
    await expect(bad({ datasetVersionId: randomUUID() })).rejects.toThrow(/not found/);
    expect(await count()).toEqual([0, 0]);
    await expect(f.experiments.benchmark(viewer, input(f.dataset, f.pipeline))).rejects.toThrow(/tenant_admin/);
    const first = await bad({ idempotencyKey: "retry-1" }), again = await bad({ idempotencyKey: "retry-1" });
    expect(again.orchestration.id).toBe(first.orchestration.id); expect(await count()).toEqual([1, 4]);
    await expect(bad({ idempotencyKey: "retry-1", seed: 6 })).rejects.toThrow(/different definition/);
    await expect(f.experiments.get(outsider, first.orchestration.id)).rejects.toThrow(/not found/);
    expect((await f.experiments.get(viewer, first.orchestration.id)).candidates).toHaveLength(4);
    await expect(pool.query("UPDATE ml_orchestrations SET config='{}' WHERE id=$1", [first.orchestration.id])).rejects.toThrow(/immutable/);
    await expect(pool.query("DELETE FROM ml_orchestration_members WHERE orchestration_id=$1", [first.orchestration.id])).rejects.toThrow();
  });
  it("exposes the catalog and benchmark workflow through the API", async () => {
    const f = await fixture(); const app = new Hono<{ Variables: { principal: Principal; database: pg.Pool; correlationId: string } }>();
    app.use("*", async (c, next) => { c.set("principal", manager); c.set("database", pool); await next(); });
    registerMlRoutes(app, { enabled: true, backends: f.backends });
    const catalog = await (await app.request("/ml/models")).json(); expect(catalog.models).toHaveLength(15); expect(catalog.metrics.f1.direction).toBe("higher");
    const post = await app.request("/ml/benchmarks", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(input(f.dataset, f.pipeline)) });
    expect(post.status).toBe(202); const created = await post.json(); await f.drain();
    const got = await (await app.request(`/ml/benchmarks/${created.orchestration.id}`)).json(); expect(got.orchestration.status).toBe("COMPLETED");
    const ranking = await (await app.request(`/ml/experiments/${created.orchestration.id}/ranking`)).json(); expect(ranking.final).toBe(true); expect(ranking.ranking.entries[0].label).toBe("random_forest_classifier");
    expect((await (await app.request("/ml/benchmarks")).json())).toHaveLength(1);
  });
});

const searchInput = (ds: { id: string }, pl: { id: string }, extra: Record<string, unknown> = {}) => ({
  name: "forest search", datasetVersionId: ds.id, featurePipelineId: pl.id, taskType: "classification", seed: 8, algorithm: "random_forest_classifier", strategy: "grid", maxCandidates: 10,
  space: { max_depth: { type: "choice", values: [4, 8] }, n_estimators: { type: "choice", values: [100, 300] } }, evaluation: { primaryMetric: "f1", cv: { strategy: "stratified_kfold", folds: 3 } }, ...extra });

describe("hyperparameter search experiments", () => {
  it("runs each generated configuration as a normal child job and recommends without registering", async () => {
    const f = await fixture(); const created = await f.experiments.search(manager, searchInput(f.dataset, f.pipeline));
    expect(created.orchestration.kind).toBe("search"); expect(created.candidates.map(c => c.label)).toEqual([1, 2, 3, 4].map(i => `random_forest_classifier#${i}`));
    expect(created.candidates.map(c => c.requestedHyperparameters)).toEqual([{ max_depth: 4, n_estimators: 100 }, { max_depth: 4, n_estimators: 300 }, { max_depth: 8, n_estimators: 100 }, { max_depth: 8, n_estimators: 300 }]);
    expect(created.orchestration.config).toMatchObject({ definition: { strategy: "grid", maxCandidates: 10, algorithm: "random_forest_classifier" }, shared: { seed: 8 } });
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id, "search");
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.recommendation).toMatchObject({ label: "random_forest_classifier#4" });
    expect(labels(done)).toEqual(["random_forest_classifier#4", "random_forest_classifier#3", "random_forest_classifier#2", "random_forest_classifier#1"]);
    const child = await f.training.getJob(manager, done.candidates[0]!.jobId);
    expect(child.job.snapshot).toMatchObject({ datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, seed: 8, hyperparameters: { max_depth: 4, n_estimators: 100 }, evaluation: { cv: { folds: 3 } } });
    expect(child.attempts[0].performance.fitSeconds).toBe(.5); expect(child.artifacts).toHaveLength(1);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
    await expect(f.experiments.get(manager, created.orchestration.id, "benchmark")).rejects.toThrow(/not found/);
  });
  it("generates the same random candidates for the same seed and keeps rankings reproducible", async () => {
    const f = await fixture(); const space = { n_estimators: { type: "int", min: 10, max: 400 }, max_depth: { type: "choice", values: [3, 6, 9, 12] } };
    const run = async (seed: number, name: string) => { const e = await f.experiments.search(manager, searchInput(f.dataset, f.pipeline, { strategy: "random", maxCandidates: 6, space, seed, name })); await f.drain(); return f.experiments.get(manager, e.orchestration.id); };
    const a = await run(21, "a"), b = await run(21, "b"), c = await run(22, "c");
    const configs = (v: typeof a) => v.candidates.map(x => x.requestedHyperparameters);
    expect(configs(a)).toEqual(configs(b)); expect(configs(a)).not.toEqual(configs(c)); expect(a.candidates).toHaveLength(6);
    expect(a.ranking.entries.map(e => e.hyperparameters)).toEqual(b.ranking.entries.map(e => e.hyperparameters));
  });
  it("keeps the other candidates when one configuration fails", async () => {
    const backend = new FakeBackend(); backend.failWhen = s => s.hyperparameters.max_depth === 8 && s.hyperparameters.n_estimators === 300; const f = await fixture(backend);
    const created = await f.experiments.search(manager, searchInput(f.dataset, f.pipeline)); await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("PARTIALLY_COMPLETED"); expect(done.ranking.entries).toHaveLength(3); expect(done.recommendation).toMatchObject({ label: "random_forest_classifier#3" });
  });
  it("rejects invalid spaces before creating any run", async () => {
    const f = await fixture(); const count = async () => (await pool.query("SELECT (SELECT count(*) FROM ml_orchestrations)::int AS o,(SELECT count(*) FROM ml_training_jobs)::int AS j")).rows[0];
    for (const bad of [{ space: {} }, { space: { n_jobs: { type: "choice", values: [1] } } }, { space: { max_depth: { type: "int", min: 9, max: 3 } } }, { maxCandidates: 2 }, { strategy: "bayesian" }, { algorithm: "ridge" }, { evaluation: { primaryMetric: "rmse" } }, { fixedHyperparameters: { max_depth: 3 } }]) await expect(f.experiments.search(manager, searchInput(f.dataset, f.pipeline, bad)), JSON.stringify(bad)).rejects.toThrow();
    await expect(f.experiments.search(viewer, searchInput(f.dataset, f.pipeline))).rejects.toThrow(/tenant_admin/);
    expect(await count()).toEqual({ o: 0, j: 0 });
  });
});

const autoInput = (ds: { id: string }, extra: Record<string, unknown> = {}) => ({ name: "auto churn", datasetVersionId: ds.id, target: "label", taskType: "classification", metric: "f1", secondaryMetrics: ["accuracy"], seed: 12, budget: { maxCandidateRuns: 10, maxSearchCandidatesPerModel: 2 }, ...extra });

describe("bounded AutoML experiments", () => {
  it("plans within budget, runs benchmark and search children, ranks across them and only recommends", async () => {
    const f = await fixture(); const created = await f.experiments.automl(manager, autoInput(f.dataset));
    expect(created.orchestration.kind).toBe("automl"); expect(created.children.map(c => c.kind)).toEqual(["benchmark", "search", "search"]);
    expect(created.candidates).toHaveLength(9); expect(created.candidates.length).toBeLessThanOrEqual(10);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_jobs")).rows[0].n).toBe(9);
    expect(created.orchestration.config).toMatchObject({ plan: { totalCandidates: 9 }, shared: { evaluation: { cv: { strategy: "stratified_kfold", folds: 5 } } } });
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id, "automl");
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.children.every(c => c.status === "COMPLETED")).toBe(true); expect(done.ranking.entries).toHaveLength(9);
    expect(done.recommendation).toMatchObject({ basis: "f1 on cv" }); expect(done.recommendation!.note).toMatch(/not registered, promoted or deployed/);
    expect(done.ranking.entries[0]!.label).toMatch(/^(random_forest_classifier|gradient_boosting_classifier)#/);
    const lineage = done.candidates.map(c => c.lineage!); expect(new Set(lineage.map(l => l.featurePipelineId)).size).toBe(1); expect(new Set(lineage.map(l => l.rootOrchestrationId))).toEqual(new Set([created.orchestration.id]));
    expect(new Set(done.candidates.map(c => c.orchestrationId)).size).toBe(3);
    const pipeline = (await pool.query("SELECT document FROM ml_feature_pipelines WHERE id=$1", [lineage[0]!.featurePipelineId])).rows[0].document;
    expect(pipeline.definition).toEqual({ features: ["x"], steps: [{ operation: "impute_median", parameters: {} }, { operation: "standard_scale", parameters: {} }] });
    for (const table of ["ml_registry_entries", "ml_model_versions", "ml_endpoints"]) expect((await pool.query(`SELECT count(*)::int AS n FROM ${table}`)).rows[0].n).toBe(0);
    await expect(f.experiments.get(manager, created.orchestration.id, "search")).rejects.toThrow(/not found/);
  });
  it("is reproducible for the same dataset version, configuration, seed and budget", async () => {
    const f = await fixture(); const run = async (name: string, extra: Record<string, unknown> = {}) => { const e = await f.experiments.automl(manager, autoInput(f.dataset, { name, ...extra })); await f.drain(); return f.experiments.get(manager, e.orchestration.id); };
    const a = await run("first"), b = await run("second"), other = await run("other seed", { seed: 13 });
    expect(b.candidates.map(c => [c.label, c.requestedHyperparameters])).toEqual(a.candidates.map(c => [c.label, c.requestedHyperparameters]));
    expect(b.ranking.entries.map(e => e.label)).toEqual(a.ranking.entries.map(e => e.label)); expect(b.recommendation!.label).toBe(a.recommendation!.label);
    expect(other.candidates.map(c => c.requestedHyperparameters)).not.toEqual(a.candidates.map(c => c.requestedHyperparameters));
  });
  it("honours small budgets and rejects invalid requests without creating anything", async () => {
    const f = await fixture(); const small = await f.experiments.automl(manager, autoInput(f.dataset, { budget: { maxCandidateRuns: 3, maxSearchCandidatesPerModel: 5 } }));
    expect(small.candidates.length).toBeLessThanOrEqual(3);
    const count = async () => (await pool.query("SELECT (SELECT count(*) FROM ml_orchestrations)::int AS o,(SELECT count(*) FROM ml_training_jobs)::int AS j,(SELECT count(*) FROM ml_feature_pipelines)::int AS p")).rows[0];
    const before = await count();
    for (const bad of [{ budget: { maxCandidateRuns: 41 } }, { metric: "rmse" }, { target: "x" }, { taskType: "regression", target: "label", metric: "rmse" }, { cvFolds: 11 }, { datasetVersionId: randomUUID() }]) await expect(f.experiments.automl(manager, autoInput(f.dataset, bad)), JSON.stringify(bad)).rejects.toThrow();
    await expect(f.experiments.automl(viewer, autoInput(f.dataset))).rejects.toThrow(/tenant_admin/);
    expect(await count()).toEqual(before);
  });
  it("supports idempotent creation and parent cancellation across children", async () => {
    const f = await fixture();
    const first = await f.experiments.automl(manager, autoInput(f.dataset, { idempotencyKey: "auto-1" })), again = await f.experiments.automl(manager, autoInput(f.dataset, { idempotencyKey: "auto-1", name: "renamed" }));
    expect(again.orchestration.id).toBe(first.orchestration.id); expect((await pool.query("SELECT count(*)::int AS n FROM ml_training_jobs")).rows[0].n).toBe(9);
    await expect(f.experiments.automl(manager, autoInput(f.dataset, { idempotencyKey: "auto-1", seed: 99 }))).rejects.toThrow(/different definition/);
    await f.experiments.cancel(manager, first.orchestration.id); await new RunRepository(pool).finalizeUnleasedCancellations();
    const cancelled = await f.experiments.get(manager, first.orchestration.id);
    expect(cancelled.orchestration.status).toBe("CANCELLED"); expect(cancelled.children.every(c => c.status === "CANCELLED")).toBe(true); expect(cancelled.candidates.every(c => c.status === "CANCELLED")).toBe(true);
  });
  it("keeps AutoML results when one family fails, and reports FAILED when nothing succeeds", async () => {
    const backend = new FakeBackend(); backend.failWhen = s => s.algorithm === "gradient_boosting_classifier"; const f = await fixture(backend);
    const created = await f.experiments.automl(manager, autoInput(f.dataset)); await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("PARTIALLY_COMPLETED"); expect(done.children.map(c => c.status).sort()).toEqual(["COMPLETED", "FAILED", "PARTIALLY_COMPLETED"]);
    expect(done.ranking.entries.length).toBeGreaterThan(0); expect(done.ranking.entries.every(e => e.algorithm !== "gradient_boosting_classifier")).toBe(true);
    const doomed = new FakeBackend(); doomed.failWhen = () => true; const g = await fixture(doomed);
    const none = await g.experiments.automl(manager, autoInput(g.dataset)); await g.drain(); const failed = await g.experiments.get(manager, none.orchestration.id);
    expect(failed.orchestration.status).toBe("FAILED"); expect(failed.recommendation).toBeNull();
  });
  it("is available through the API", async () => {
    const f = await fixture(); const app = new Hono<{ Variables: { principal: Principal; database: pg.Pool; correlationId: string } }>();
    app.use("*", async (c, next) => { c.set("principal", manager); c.set("database", pool); await next(); });
    registerMlRoutes(app, { enabled: true, backends: f.backends });
    const post = await app.request("/ml/automl", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(autoInput(f.dataset)) });
    expect(post.status).toBe(202); const created = await post.json(); await f.drain();
    const got = await (await app.request(`/ml/automl/${created.orchestration.id}`)).json(); expect(got.orchestration.status).toBe("COMPLETED"); expect(got.recommendation).not.toBeNull();
    expect((await (await app.request("/ml/automl")).json())).toHaveLength(1);
  });
});

describe.runIf(Boolean(python))("real scikit-learn benchmark fixtures", () => {
  const classificationRows = Array.from({ length: 150 }, (_, i) => { const hi = i % 3 === 0; return { a: (i * 37) % 17 + (hi ? 6 : 0), b: (i * 11) % 13, label: hi ? "churn" : "stay" }; });
  async function real(rows: object[], columns: object[], target: string, features: string[], steps: object[]) {
    const f = await fixture(new SklearnBackend(python!));
    const dataset = await f.training.ingest(manager, { name: "real", schema: { id: randomUUID(), columns, target }, rows }); const pipeline = await f.training.pipeline(manager, { name: "real", definition: { features, steps } });
    return { ...f, dataset, pipeline };
  }
  it("benchmarks four classifiers with stratified 5-fold CV, ranks by F1, reproduces, and leaves registration to the user", async () => {
    const f = await real(classificationRows, [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "label", type: "string" }], "label", ["a", "b"], [{ operation: "impute_median" }, { operation: "standard_scale" }]);
    const definition = { ...input(f.dataset, f.pipeline), candidates: [{ algorithm: "logistic_regression" }, { algorithm: "random_forest_classifier", hyperparameters: { n_estimators: 25 } }, { algorithm: "gradient_boosting_classifier", hyperparameters: { n_estimators: 30 } }, { algorithm: "svc" }], evaluation: { primaryMetric: "f1", secondaryMetrics: ["accuracy"], cv: { strategy: "stratified_kfold", folds: 5 } } };
    const first = await f.experiments.benchmark(manager, definition); await f.drain(); const second = await f.experiments.benchmark(manager, { ...definition, name: "repeat" }); await f.drain();
    const a = await f.experiments.get(manager, first.orchestration.id), b = await f.experiments.get(manager, second.orchestration.id);
    expect(a.orchestration.status).toBe("COMPLETED"); expect(a.ranking.entries).toHaveLength(4);
    for (const c of a.candidates) { expect(c.metrics.filter(m => m.partition === "cv_fold" && m.name === "f1")).toHaveLength(5); expect(c.artifactBytes).toBeGreaterThan(0); expect(Object.keys(c.performance)).toEqual(expect.arrayContaining(["fitSeconds", "predictSeconds", "cvFitSeconds"])); }
    expect(labels(b)).toEqual(labels(a));
    a.ranking.entries.forEach((e, i) => { expect(b.ranking.entries[i]!.primary.value).toBeCloseTo(e.primary.value, 9); expect(b.ranking.entries[i]!.primary.std).toBeCloseTo(e.primary.std!, 9); });
    const winner = a.recommendation!; expect(winner.basis).toBe("f1 on cv");
    // Registration, readiness and the endpoint are separate explicit steps.
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_model_versions")).rows[0].n).toBe(0);
    const registry = await f.models.registry(manager, { name: "churn" }); const version = await f.models.register(manager, registry.id, { trainingRunId: winner.trainingRunId });
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "churn-endpoint", modelVersionId: version.id });
    const prediction = await f.models.predict(manager, endpoint.id, [{ a: 20, b: 3 }, { a: 1, b: 3 }], new AbortController().signal); expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toEqual(["churn", "stay"]);
  }, 400_000);
  it("benchmarks regressors with k-fold CV and ranks by RMSE (lower is better)", async () => {
    const rows = Array.from({ length: 120 }, (_, i) => ({ a: (i * 7) % 20, b: (i * 3) % 9, y: 2 * ((i * 7) % 20) + ((i * 3) % 9) + ((i % 5) - 2) * .1 }));
    const f = await real(rows, [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "y", type: "number" }], "y", ["a", "b"], [{ operation: "impute_mean" }, { operation: "min_max_scale" }]);
    const created = await f.experiments.benchmark(manager, { name: "regression", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, taskType: "regression", seed: 3, split: { train: .6, validation: .2, test: .2 },
      evaluation: { primaryMetric: "rmse", secondaryMetrics: ["r2", "mae"], cv: { strategy: "kfold", folds: 3 } }, candidates: [{ algorithm: "linear_regression" }, { algorithm: "ridge" }, { algorithm: "decision_tree_regressor" }, { algorithm: "k_neighbors_regressor" }] });
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("COMPLETED"); const values = done.ranking.entries.map(e => e.primary.value);
    expect(values).toEqual([...values].sort((x, y) => x - y)); expect(done.ranking.entries[0]!.primary.direction).toBe("lower"); expect(["linear_regression", "ridge"]).toContain(done.ranking.entries[0]!.label);
  }, 300_000);
  it("searches a logistic-regression grid with real cross-validation", async () => {
    const f = await real(classificationRows, [{ name: "a", type: "number" }, { name: "b", type: "number" }, { name: "label", type: "string" }], "label", ["a", "b"], [{ operation: "standard_scale" }]);
    const created = await f.experiments.search(manager, { name: "C search", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, taskType: "classification", seed: 4, algorithm: "logistic_regression", strategy: "grid", maxCandidates: 5,
      space: { C: { type: "float", min: .001, max: 10, log: true, steps: 3 } }, fixedHyperparameters: { max_iter: 500 }, evaluation: { primaryMetric: "f1", secondaryMetrics: ["log_loss"], cv: { strategy: "stratified_kfold", folds: 3 } } });
    await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.candidates.map(c => c.resolvedHyperparameters!.C)).toEqual([.001, .1, 10]); expect(done.candidates.every(c => c.resolvedHyperparameters!.max_iter === 500)).toBe(true);
    const f1 = done.ranking.entries.map(e => e.primary.value); expect(f1).toEqual([...f1].sort((x, y) => y - x));
  }, 200_000);
  it("runs a small AutoML over mixed numeric and categorical features end to end", async () => {
    const columns = [{ name: "a", type: "number" }, { name: "kind", type: "string", nullable: true }, { name: "flag", type: "boolean" }, { name: "label", type: "string" }];
    const rows = Array.from({ length: 90 }, (_, i) => ({ a: i % 9, kind: i % 4 === 0 ? null : ["red", "blue", "green"][i % 3]!, flag: i % 2 === 0, label: i % 9 >= 5 ? "hi" : "lo" }));
    const f = await fixture(new SklearnBackend(python!)); const dataset = await f.training.ingest(manager, { name: "mixed", schema: { id: randomUUID(), columns, target: "label" }, rows });
    const created = await f.experiments.automl(manager, { name: "real auto", datasetVersionId: dataset.id, target: "label", taskType: "classification", metric: "f1", seed: 3, cvFolds: 3, budget: { maxCandidateRuns: 4, maxSearchCandidatesPerModel: 2 } });
    expect(created.candidates.length).toBeLessThanOrEqual(4); await f.drain(); const done = await f.experiments.get(manager, created.orchestration.id);
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.recommendation).not.toBeNull(); expect(done.ranking.entries[0]!.primary.value).toBeGreaterThan(.9);
    const pipeline = (await pool.query("SELECT document FROM ml_feature_pipelines WHERE id=$1", [done.candidates[0]!.lineage!.featurePipelineId])).rows[0].document;
    expect(pipeline.definition.steps.map((x: { operation: string }) => x.operation)).toEqual(["impute_median", "standard_scale", "impute_most_frequent", "one_hot_encode"]);
  }, 300_000);
});
