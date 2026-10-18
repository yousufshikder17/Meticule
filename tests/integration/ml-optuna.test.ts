import { resolve } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import type { MlBackend, TrainedModel } from "../../src/ml/backend.js";
import type { Snapshot } from "../../src/ml/domain.js";
import { MlExperimentService } from "../../src/ml/experiments.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { callRunner } from "../../src/ml/runner-process.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { json, LinearTestBackend, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

const python = process.env.ML_TEST_MLOPS_PYTHON;
const optuna = python ? { python, timeoutMs: 60_000 } : null;
beforeEach(truncate);
afterAll(() => pool.end());

/** Fails every trial whose alpha exceeds 6, so the study sees FAIL trials from real failed jobs. */
class PickyBackend extends LinearTestBackend {
  override async train(s: Snapshot): Promise<TrainedModel> { if (Number(s.hyperparameters.alpha) > 6) throw new Error("diverged"); return super.train(s); }
}

async function study(store: Parameters<typeof trainedModel>[0], backend: MlBackend, overrides: Record<string, unknown> = {}) {
  const f = await trainedModel(store, backend); // reuses the fixture's dataset and pipeline; its own job is not part of the study
  const service = new MlExperimentService(pool, store, f.backends, { optuna });
  const worker = new LifecycleWorker(pool, { workerId: "optuna-test", leaseSeconds: 30, kind: "ml" }, new TrainingProcessor(pool, store, f.backends, { optuna }));
  const created = await service.optuna(manager, { name: "alpha study", experimentId: f.experiment.id, datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: backend.id, taskType: "regression", seed: 11,
    evaluation: { primaryMetric: "mae" }, algorithm: "linear", space: { alpha: { type: "float", min: 0, max: 10 } }, sampler: "tpe", maxTrials: 12, parallelism: 2, ...overrides });
  const run = async (limit = 40) => { for (let i = 0; i < limit && await worker.tick(); i++); return service.get(manager, created.orchestration.id); };
  return { f, service, worker, created, run };
}

describe.runIf(Boolean(python))("Optuna studies", () => {
  it("runs a bounded TPE study through durable jobs, records provenance and registers the best trial", () => withStore(async store => {
    const s = await study(store, new LinearTestBackend());
    expect(s.created.candidates).toHaveLength(2); // parallelism 2: two trials asked up front
    const done = await s.run();
    expect(done.orchestration.status).toBe("COMPLETED");
    const trials = done.study!.trials;
    expect(trials).toHaveLength(12); expect(trials.every(t => t.state === "COMPLETE")).toBe(true);
    expect(trials.map(t => t.number)).toEqual([...Array(12).keys()]);
    expect(new Set(trials.map(t => t.params.alpha)).size).toBe(12);
    expect(done.study).toMatchObject({ sampler: "tpe", direction: "minimize", objective: { metric: "mae", partition: "validation" }, maxTrials: 12, parallelism: 2 });
    const values = trials.map(t => t.value!);
    expect(done.study!.best!.value).toBe(Math.min(...values));
    expect(done.recommendation!.trainingRunId).toBe(done.study!.best!.trainingRunId);
    const members = (await pool.query("SELECT candidate FROM ml_orchestration_members WHERE orchestration_id=$1 ORDER BY ordinal", [s.created.orchestration.id])).rows.map(r => r.candidate);
    expect(members.every(m => m.origin === "optuna" && m.trial.optuna === "5.0.0" && m.trial.sampler === "TPESampler")).toBe(true);
    expect(new Set(members.map(m => m.trial.seed)).size).toBeGreaterThan(1); // each rebuild draws from a fresh seed
    // TPE takes over after its 10 random start-up trials; its proposals must be closer to the optimum (alpha = 3) on average.
    const tpe = trials.slice(10).map(t => Math.abs(Number(t.params.alpha) - 3)), startup = trials.slice(0, 10).map(t => Math.abs(Number(t.params.alpha) - 3));
    expect(tpe.reduce((a, b) => a + b) / tpe.length).toBeLessThan(startup.reduce((a, b) => a + b) / startup.length);
    // Registration from a selected completed trial uses the ordinary registry.
    const models = new MlModelService(pool, store, s.f.backends);
    const registry = await models.registry(manager, { name: "tuned" });
    const version = await models.register(manager, registry.id, { trainingRunId: done.study!.best!.trainingRunId! });
    expect(version.training_run_id).toBe(done.study!.best!.trainingRunId);
    expect((await s.service.get(viewer, s.created.orchestration.id)).study!.trials).toHaveLength(12);
    await expect(s.service.get(outsider, s.created.orchestration.id)).rejects.toThrow(/not found/);
  }), 300_000);

  it("feeds failed trials back as FAIL, stops at cancellation, and never adds trials to a cancelled study", () => withStore(async store => {
    const s = await study(store, new PickyBackend(), { sampler: "random", maxTrials: 30 });
    for (let i = 0; i < 6; i++) await s.worker.tick();
    const mid = await s.service.get(manager, s.created.orchestration.id);
    expect(mid.study!.trials.some(t => t.state === "FAIL")).toBe(true);
    expect(mid.study!.trials.filter(t => t.state === "FAIL").every(t => Number(t.params.alpha) > 6)).toBe(true);
    const cancelled = await s.service.cancel(manager, s.created.orchestration.id);
    expect(cancelled.orchestration.cancelRequestedAt).not.toBeNull();
    const after = await s.run();
    expect(after.orchestration.status).toBe("CANCELLED");
    const count = after.study!.trials.length; expect(count).toBeLessThan(30);
    expect(after.study!.trials.filter(t => t.state === "RUNNING")).toHaveLength(0);
    await s.service.refresh(manager.tenantId, s.created.orchestration.id);
    expect((await s.service.get(manager, s.created.orchestration.id)).study!.trials).toHaveLength(count);
    const job = (await pool.query("SELECT job_id FROM ml_orchestration_members WHERE orchestration_id=$1 LIMIT 1", [s.created.orchestration.id])).rows[0].job_id;
    await expect(pool.query("INSERT INTO ml_orchestration_members(tenant_id,orchestration_id,ordinal,job_id,label,candidate) VALUES($1,$2,999,$3,'x','{}')", [manager.tenantId, s.created.orchestration.id, job])).rejects.toThrow();
  }), 300_000);

  it("starts no trial after its time limit and finishes with the trials it has", () => withStore(async store => {
    const s = await study(store, new LinearTestBackend(), { timeoutSeconds: 60 });
    const client = await pool.connect();
    try { await client.query("SET session_replication_role=replica"); await client.query("UPDATE ml_orchestrations SET created_at=now()-interval '2 minutes' WHERE id=$1", [s.created.orchestration.id]); }
    finally { await client.query("RESET session_replication_role"); client.release(); }
    const done = await s.run();
    expect(done.orchestration.status).toBe("COMPLETED"); expect(done.study!.trials).toHaveLength(2);
  }), 300_000);

  it("lets the worker sweep advance and finalize a study whose terminal hook could not ask", () => withStore(async store => {
    const f = await trainedModel(store); // f.worker has no Optuna configuration, like a misconfigured worker
    const service = new MlExperimentService(pool, store, f.backends, { optuna });
    const created = await service.optuna(manager, { name: "swept", experimentId: f.experiment.id, datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: "test-p5", taskType: "regression", seed: 2,
      evaluation: { primaryMetric: "mae" }, algorithm: "linear", space: { alpha: { type: "float", min: 0, max: 10 } }, maxTrials: 2 });
    await f.worker.tick();
    const stalled = await service.read(manager.tenantId, created.orchestration.id);
    expect(stalled.orchestration.status).toBe("RUNNING"); expect(stalled.study!.trials).toHaveLength(1);
    // An older study whose trial is still open needs nothing from the sweep, so it cannot take the only sweep slot.
    const busy = await service.optuna(manager, { name: "busy", experimentId: f.experiment.id, datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: "test-p5", taskType: "regression", seed: 3,
      evaluation: { primaryMetric: "mae" }, algorithm: "linear", space: { alpha: { type: "float", min: 0, max: 10 } }, maxTrials: 1 });
    await pool.query(`ALTER TABLE ml_orchestrations DISABLE TRIGGER ml_orchestration_update; UPDATE ml_orchestrations SET created_at=created_at-interval '1 minute' WHERE id='${busy.orchestration.id}'; ALTER TABLE ml_orchestrations ENABLE TRIGGER ml_orchestration_update`);
    await service.sweepStudies(1);
    expect((await service.read(manager.tenantId, created.orchestration.id)).study!.trials).toHaveLength(2);
    while (await f.worker.tick()); await service.sweepStudies();
    expect((await service.read(manager.tenantId, created.orchestration.id)).orchestration.status).toBe("COMPLETED");
    expect((await service.read(manager.tenantId, busy.orchestration.id)).orchestration.status).toBe("COMPLETED");
  }), 300_000);

  it("asks deterministically for a given history and seed, and refuses spaces outside the allowlist", () => withStore(async store => {
    const ask = (trials: unknown[]) => callRunner({ label: "optuna", python: python!, script: resolve("ml/optuna_runner.py"), timeoutMs: 60_000, signal: new AbortController().signal,
      request: { command: "ask", sampler: "tpe", direction: "minimize", seed: 5, space: { alpha: { type: "float", min: 0, max: 10 }, k: { type: "choice", values: ["a", "b"] } }, count: 2, trials } });
    const history = [{ params: { alpha: 1, k: "a" }, state: "COMPLETE", value: .3 }];
    expect(await ask(history)).toEqual(await ask(history));
    expect((await ask(history) as { trials: unknown[] }).trials).not.toEqual((await ask([...history, { params: { alpha: 2, k: "b" }, state: "FAIL", value: null }]) as { trials: unknown[] }).trials);
    await expect(ask([{ params: { alpha: 50, k: "a" }, state: "COMPLETE", value: 1 }])).rejects.toMatchObject({ category: "invalid_request" });
    const f = await trainedModel(store); const service = new MlExperimentService(pool, store, f.backends, { optuna });
    const base = { name: "bad", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: "test-p5", taskType: "regression", seed: 1, evaluation: { primaryMetric: "mae" }, algorithm: "linear", maxTrials: 3 };
    await expect(service.optuna(manager, { ...base, space: { gamma: { type: "float", min: 0, max: 1 } } })).rejects.toThrow(/unsupported search parameter/);
    await expect(service.optuna(manager, { ...base, space: { alpha: { type: "float", min: 0, max: 1000 } } })).rejects.toThrow(/invalid value/);
    await expect(service.optuna(viewer, { ...base, space: { alpha: { type: "float", min: 0, max: 1 } } })).rejects.toThrow(/role/);
    await expect(service.optuna(manager, { ...base, maxTrials: 51, space: { alpha: { type: "float", min: 0, max: 1 } } })).rejects.toThrow();
  }), 120_000);
});

describe("Optuna feature flag", () => {
  it("returns 503 when Optuna is not configured", () => withStore(async store => {
    const f = await trainedModel(store);
    const response = await testApp(store, f.backends).request("/ml/optuna-studies", json({ name: "x", datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: "test-p5", taskType: "regression", seed: 1, evaluation: { primaryMetric: "mae" }, algorithm: "linear", space: { alpha: { type: "float", min: 0, max: 1 } }, maxTrials: 2 }));
    expect(response.status).toBe(503);
  }));
});
