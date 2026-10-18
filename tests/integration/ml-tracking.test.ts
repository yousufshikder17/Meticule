import { randomUUID } from "node:crypto";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { withTenantSession } from "../../src/db/tenant-session.js";
import { MlTrackingService } from "../../src/ml/tracking.js";
import { MlLineageService } from "../../src/ml/lineage.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { dropBlob, json, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

beforeEach(truncate);
afterAll(() => pool.end());

describe("Phase 5 experiment tracking", () => {
  it("records metadata, filters by project, tag, name and derived status, and isolates tenants", () => withStore(async store => {
    const f = await trainedModel(store); const app = testApp(store, f.backends);
    const child = await (await app.request("/ml/experiments", json({ name: "100% tuned_v2", description: "child", project: "mlops", tags: { team: "core", stage: "tuning" }, parentExperimentId: f.experiment.id }))).json();
    expect(child).toMatchObject({ project: "mlops", parent_experiment_id: f.experiment.id, tags: { stage: "tuning" } });
    expect((await app.request("/ml/experiments", json({ name: "orphan", parentExperimentId: randomUUID() }))).status).toBe(404);
    expect((await app.request("/ml/experiments", json({ name: "bad", unknown: 1 }))).status).toBe(400);
    const list = async (query: string) => (await (await app.request(`/ml/experiments?${query}`)).json()).map((e: { name: string }) => e.name).sort();
    expect(await list("project=mlops")).toEqual(["100% tuned_v2", "phase5"]);
    expect(await list("tag=team:core&tag=stage:tuning")).toEqual(["100% tuned_v2"]);
    expect(await list("nameContains=0%25")).toEqual(["100% tuned_v2"]); // % is literal, not a wildcard
    expect(await list("nameContains=_")).toEqual(["100% tuned_v2"]);
    expect(await list("status=COMPLETED")).toEqual(["phase5"]);
    expect(await list("status=EMPTY")).toEqual(["100% tuned_v2"]);
    expect(await list(`parentExperimentId=${f.experiment.id}`)).toEqual(["100% tuned_v2"]);
    const detail = await (await app.request(`/ml/experiments/${f.experiment.id}`)).json();
    expect(detail.experiment).toMatchObject({ status: "COMPLETED", summary: { total: 1, completed: 1 } }); expect(detail.children).toHaveLength(1); expect(detail.jobs).toHaveLength(1);
    const patched = await (await app.request(`/ml/experiments/${child.id}`, json({ tags: { stage: "done" } }, "PATCH"))).json();
    expect(patched.tags).toEqual({ stage: "done" });
    await expect(pool.query("UPDATE ml_experiments SET name='renamed' WHERE id=$1", [child.id])).rejects.toThrow(/immutable/);
    expect((await testApp(store, f.backends, viewer).request(`/ml/experiments/${child.id}`, json({ description: "x" }, "PATCH"))).status).toBe(403);
    expect(await (await testApp(store, f.backends, outsider).request("/ml/experiments")).json()).toEqual([]);
    expect((await testApp(store, f.backends, outsider).request(`/ml/experiments/${child.id}`)).status).toBe(404);
    // Under the API role, row-level security hides the rows and the column grant allows only description and tags.
    await withTenantSession(pool, outsider.tenantId, async db => { expect((await db.query("SELECT * FROM ml_experiments")).rowCount).toBe(0); });
    await withTenantSession(pool, manager.tenantId, async db => { await expect(db.query("UPDATE ml_experiments SET project='other' WHERE id=$1", [child.id])).rejects.toThrow(/permission denied/); });
  }));

  it("lists run history, compares runs and reproduces a job with an evidence-based verdict", () => withStore(async store => {
    const f = await trainedModel(store); const tracking = new MlTrackingService(pool, store, f.backends);
    const reproduction = await tracking.reproduce(manager, f.job.id);
    expect(reproduction.job.reproduces_job_id).toBe(f.job.id); expect(reproduction.caveats.length).toBeGreaterThan(0);
    expect(reproduction.job.snapshot.split).toEqual(f.job.snapshot.split);
    await f.worker.tick();
    const second = (await f.training.getJob(manager, reproduction.job.id)).attempts[0];
    const verdict = await tracking.reproducibility(manager, f.attempt.id, second.id);
    expect(verdict).toMatchObject({ verdict: "identical", sameSpecification: true, sameSplit: true, environmentDifferences: [] });
    // A different seed is a different specification; its metrics are reported but never called a reproduction.
    const other = await f.training.queue(manager, { experimentId: f.experiment.id, spec: { ...f.spec, seed: 8, hyperparameters: { alpha: 5 } } }); await f.worker.tick();
    const third = (await f.training.getJob(manager, other.id)).attempts[0];
    expect((await tracking.reproducibility(manager, f.attempt.id, third.id)).verdict).toBe("not_comparable");
    const runs = await tracking.runs(manager, f.experiment.id);
    expect(runs).toHaveLength(3); expect(runs.find(r => r.job_id === reproduction.job.id).reproduces_job_id).toBe(f.job.id);
    expect(runs[0].metrics.length).toBe(3);
    const same = await tracking.compare(manager, { trainingRunIds: [f.attempt.id, second.id] });
    expect(same.comparable).toBe(true);
    const mixed = await tracking.compare(manager, { trainingRunIds: [f.attempt.id, third.id] });
    expect(mixed.comparable).toBe(false); expect(mixed.comparabilityNote).toMatch(/seed/);
    const mae = mixed.metrics.find(m => m.partition === "test" && m.name === "mae")!;
    expect(mae.best).toBe(f.attempt.id); expect(mae.values[third.id]).toBeCloseTo(.21);
    await expect(tracking.compare(outsider, { trainingRunIds: [f.attempt.id, second.id] })).rejects.toThrow(/not found/);
    await expect(tracking.reproduce(outsider, f.job.id)).rejects.toThrow(/not found/);
    await expect(new MlTrackingService(pool, store, new Map()).reproduce(manager, f.job.id)).rejects.toThrow(/backend is not configured/);
  }));

  it("refuses to reproduce when the frozen dataset bytes are gone", () => withStore(async store => {
    const f = await trainedModel(store);
    await dropBlob(manager.tenantId, f.dataset.content.sha256);
    await expect(new MlTrackingService(pool, store, f.backends).reproduce(manager, f.job.id)).rejects.toThrow(/dataset content is unavailable/);
  }));
});

describe("Phase 5 lineage", () => {
  it("traces a served model to its experiment, dataset and pipeline, and dataset versions to what serves them", () => withStore(async store => {
    const f = await trainedModel(store);
    const registry = await f.models.registry(manager, { name: "linear" }); const version = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id });
    await f.models.transition(manager, version.id, "READY"); const endpoint = await f.models.endpoint(manager, { name: "serve", modelVersionId: version.id });
    const lineage = new MlLineageService(pool, store);
    const doc = await lineage.modelVersion(manager, version.id, true);
    expect(doc.issues).toEqual([]);
    expect(doc.upstream.experiment).toMatchObject({ id: f.experiment.id, project: "mlops" });
    expect(doc.upstream.datasetVersion).toMatchObject({ id: f.dataset.id, sha256: f.dataset.content.sha256, rowCount: 60 });
    expect(doc.upstream.featurePipeline.id).toBe(f.pipeline.id);
    expect(doc.upstream.training).toMatchObject({ backend: "test-p5", algorithm: "linear", seed: 7 });
    expect(doc.upstream.trainingRun.environment).toEqual({ protocol: "test-p5", library: "1.0" });
    expect(doc.downstream.endpoints.map(e => e.id)).toEqual([endpoint.id]);
    const byDataset = await lineage.datasetVersion(manager, f.dataset.id);
    expect(byDataset.jobs.map(j => j.id)).toEqual([f.job.id]); expect(byDataset.downstream.modelVersions.map(m => m.id)).toEqual([version.id]);
    expect((await lineage.trainingRun(manager, f.attempt.id)).downstream.endpoints).toHaveLength(1);
    // Missing bytes and a dangling resume reference are reported, not hidden.
    await dropBlob(manager.tenantId, (await pool.query("SELECT content->>'sha256' AS s FROM ml_artifacts WHERE id=$1", [version.artifact_id])).rows[0].s);
    expect((await lineage.modelVersion(manager, version.id, true)).issues).toEqual([{ code: "artifact_unavailable", subject: "model_artifact", id: version.artifact_id }]);
    expect((await lineage.modelVersion(manager, version.id)).issues).toEqual([]);
    const dangling = await pool.query("INSERT INTO ml_training_runs(tenant_id,job_id,attempt,snapshot,resumed_from_epoch) VALUES($1,$2,2,$3,4) RETURNING id", [manager.tenantId, f.job.id, JSON.stringify(f.job.snapshot)]);
    expect((await lineage.trainingRun(manager, f.attempt.id)).issues).toEqual([{ code: "checkpoint_reference_missing", subject: "training_run", id: dangling.rows[0].id }]);
    await expect(lineage.modelVersion(outsider, version.id)).rejects.toThrow(/not found/);
    await expect(lineage.datasetVersion(outsider, f.dataset.id)).rejects.toThrow(/not found/);
    const app = testApp(store, f.backends, viewer);
    expect((await app.request(`/ml/lineage/model-versions/${version.id}`)).status).toBe(200);
    expect((await testApp(store, f.backends, outsider).request(`/ml/lineage/model-versions/${version.id}`)).status).toBe(404);
  }));

  it("keeps the Phase 1 experiment API shape", () => withStore(async store => {
    const e = await new MlTrainingService(pool, store).experiment(manager, "plain");
    expect(e).toMatchObject({ name: "plain", description: "", project: null, tags: {}, parent_experiment_id: null });
  }));
});
