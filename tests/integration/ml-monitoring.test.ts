import { afterAll, beforeEach, describe, expect, it } from "vitest";
import type { Row, Snapshot } from "../../src/ml/domain.js";
import { MlBackendFailure } from "../../src/ml/contract.js";
import { MlDeploymentService } from "../../src/ml/deployments.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlMonitoringService } from "../../src/ml/monitoring.js";
import { json, LinearTestBackend, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

beforeEach(truncate);
afterAll(() => pool.end());
const signal = () => new AbortController().signal;

/** Fails rows with x = -1, so monitoring sees real failed predictions. */
class FlakyBackend extends LinearTestBackend {
  override async predict(s: Snapshot, a: Buffer, rows: Row[]): Promise<unknown[]> { if (rows.some(r => r.x === -1)) throw new MlBackendFailure("prediction_failed", "boom"); return super.predict(s, a, rows); }
}

async function served(store: Parameters<typeof trainedModel>[0]) {
  const f = await trainedModel(store, new FlakyBackend());
  const registry = await f.models.registry(manager, { name: "monitored" });
  const version = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id }); await f.models.transition(manager, version.id, "READY");
  const deployments = new MlDeploymentService(pool, store, f.backends);
  const d = await deployments.create(manager, { name: "monitored" });
  await deployments.addRevision(manager, d.id, { modelVersionId: version.id }, signal()); await deployments.activate(manager, d.id, {});
  const batch = (xs: number[]) => deployments.predict(manager, d.id, xs.map(x => ({ x })), signal());
  return { f, version, deployments, d, batch, monitoring: new MlMonitoringService(pool, store) };
}

describe("Phase 5 monitoring and drift", () => {
  it("measures drift against the training reference without storing raw inputs", () => withStore(async store => {
    const s = await served(store);
    // Training rows are x = 0..35 (the fixture's first 36 rows); observed traffic first matches them, then moves far away.
    for (let i = 0; i < 4; i++) await s.batch(Array.from({ length: 36 }, (_, x) => x));
    const stable = await s.monitoring.evaluateDrift(manager, s.d.id, {});
    expect(stable).toMatchObject({ status: "stable", observed_rows: "144", reference_rows: 36, model_version_id: s.version.id });
    expect(stable.features[0]).toMatchObject({ name: "x", kind: "numeric", status: "stable" });
    expect(stable.features[0].psi).toBeLessThan(.01); expect(stable.output).toMatchObject({ name: "output", status: "stable" });
    const reference = (await pool.query("SELECT profile FROM ml_drift_references WHERE model_version_id=$1", [s.version.id])).rows[0].profile;
    expect(reference.features[0].edges).toHaveLength(9);
    // Profiles hold integer counts per reference bin and nothing else.
    for (const row of (await pool.query("SELECT profile FROM ml_prediction_profiles")).rows) {
      expect(Object.keys(row.profile).sort()).toEqual(["features", "output"]);
      expect([...row.profile.features.x, ...row.profile.output].every((n: unknown) => Number.isInteger(n))).toBe(true);
    }
    for (let i = 0; i < 3; i++) await s.batch(Array.from({ length: 50 }, (_, x) => 200 + x));
    const drifted = await s.monitoring.evaluateDrift(manager, s.d.id, {});
    expect(drifted.status).toBe("significant"); expect(drifted.features[0].psi).toBeGreaterThan(.25); expect(drifted.output.status).toBe("significant");
    // The last bin holds everything above the top training decile (31.5): 150 shifted rows plus x = 32..35 from each early batch.
    expect(drifted.features[0].observed.at(-2)).toBe(166);
    expect((await s.monitoring.evaluateDrift(manager, s.d.id, { minSamples: 10_000 })).status).toBe("insufficient_data");
    // A persisted window evaluated again over unchanged profiles gives the same evidence.
    const again = await s.monitoring.evaluateDrift(manager, s.d.id, { from: drifted.window_start.toISOString(), to: drifted.window_end.toISOString() });
    expect(again.features).toEqual(drifted.features); expect(again.status).toBe(drifted.status);
    expect((await s.monitoring.evaluations(viewer, s.d.id)).map(e => e.status)).toEqual(["significant", "insufficient_data", "significant", "stable"]);
    await expect(s.monitoring.evaluateDrift(manager, s.d.id, { from: "2026-01-02T00:00:00Z", to: "2026-01-01T00:00:00Z" })).rejects.toThrow(/precede/);
    await expect(s.monitoring.evaluateDrift(manager, s.d.id, { from: "2026-01-01T00:00:00Z", to: "2026-03-01T00:00:00Z" })).rejects.toThrow(/31 days/);
    await expect(s.monitoring.evaluateDrift(viewer, s.d.id, {})).rejects.toThrow(/role/);
    await expect(s.monitoring.evaluateDrift(outsider, s.d.id, {})).rejects.toThrow(/not found/);
    // Retention: the owner role prunes profiles; evaluations keep their evidence.
    expect(Number((await pool.query("SELECT ml_prune_monitoring('0 seconds')")).rows[0].ml_prune_monitoring)).toBe(7);
    expect((await s.monitoring.evaluateDrift(manager, s.d.id, {})).observed_rows).toBe("0");
    expect(await s.monitoring.evaluations(manager, s.d.id)).toHaveLength(5);
  }));

  it("summarises requests, latency, failures, outputs and ground truth per model version", () => withStore(async store => {
    const s = await served(store);
    const ok = await s.batch([1, 2, 3]);
    const failed = await s.batch([-1]);
    expect(failed.status).toBe("FAILED");
    const fb = s.monitoring;
    await expect(fb.feedback(manager, ok.id, { labels: [3, 5] })).rejects.toThrow(/exactly 3/);
    await expect(fb.feedback(manager, ok.id, { labels: ["3", "5", "7"] })).rejects.toThrow(/type/);
    await expect(fb.feedback(manager, failed.id, { labels: [1] })).rejects.toThrow(/completed/);
    await expect(fb.feedback(viewer, ok.id, { labels: [4, 6, 8] })).rejects.toThrow(/role/);
    await fb.feedback(manager, ok.id, { labels: [4, 6, 8] }); // every prediction is off by exactly one
    await expect(fb.feedback(manager, ok.id, { labels: [4, 6, 8] })).rejects.toThrow(/already/);
    const summary = await fb.monitor(viewer, s.d.id, {});
    expect(summary.versions).toHaveLength(1);
    const v = summary.versions[0]!;
    expect(v).toMatchObject({ modelVersionId: s.version.id, task: "regression", requests: 2, rows: 4, completed: 1, failures: 1, pending: 0, errorRate: .5, failuresByCategory: { prediction_failed: 1 } });
    expect(v.outputs).toMatchObject({ count: 3, mean: 5, min: 3, max: 7 });
    expect(v.groundTruth).toEqual({ labelledRows: 3, mae: 1, rmse: 1 });
    expect(v.latencyMs.p50).not.toBeNull(); expect(v.latencyMs.p95).toBeGreaterThanOrEqual(v.latencyMs.p50!);
    // The public endpoint summary keeps its fields and gains the median latency and failure categories.
    const endpoint = await new MlModelService(pool, store, s.f.backends).monitoring(manager, failed.endpoint_id);
    expect(endpoint).toMatchObject({ requests: 2, failed: 1, pending: 0, error_rate: .5, failures_by_category: { prediction_failed: 1 } });
    expect(endpoint.average_latency_ms).not.toBeNull(); expect(endpoint.p95_latency_ms).toBeGreaterThanOrEqual(endpoint.p50_latency_ms);
    await expect(fb.monitor(outsider, s.d.id, {})).rejects.toThrow(/not found/);
    const app = testApp(store, s.f.backends, viewer);
    expect((await app.request(`/ml/deployments/${s.d.id}/monitoring`)).status).toBe(200);
    expect((await app.request(`/ml/deployments/${s.d.id}/drift-evaluations`, json({}))).status).toBe(403);
    expect((await testApp(store, s.f.backends).request(`/ml/deployments/${s.d.id}/drift-evaluations`, json({}))).status).toBe(201);
  }));
});
