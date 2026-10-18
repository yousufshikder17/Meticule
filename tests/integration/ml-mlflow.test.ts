import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { MlMlflowService } from "../../src/ml/mlflow.js";
import { MlTrackingService } from "../../src/ml/tracking.js";
import { json, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

// A real MLflow tracking server (SQLite backend store, proxied artifact store) from the MLOps interpreter. The exporter
// talks to it over HTTP exactly as it would to an operator's server; assertions read MLflow's own REST API.
const python = process.env.ML_TEST_MLOPS_PYTHON;
let server: ChildProcess | undefined, uri = "", directory = "";
const signal = () => new AbortController().signal;
const freePort = () => new Promise<number>(done => { const s = createServer().listen(0, "127.0.0.1", () => { const port = (s.address() as { port: number }).port; s.close(() => done(port)); }); });
const api = async (path: string, body?: unknown) => (await fetch(`${uri}${path}`, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {})).json();
const runsTagged = async (experimentId: string, trainingRunId: string) => ((await api("/api/2.0/mlflow/runs/search", { experiment_ids: [experimentId], filter: `tags.\`meticule.training_run_id\` = '${trainingRunId}'` })).runs ?? []);

beforeAll(async () => {
  if (!python) return;
  directory = await mkdtemp(join(tmpdir(), "meticule-mlflow-")); const port = await freePort(); uri = `http://127.0.0.1:${port}`;
  server = spawn(python, ["-m", "mlflow", "server", "--workers", "1", "--host", "127.0.0.1", "--port", String(port), "--backend-store-uri", `sqlite:///${join(directory, "mlflow.db").replaceAll("\\", "/")}`, "--artifacts-destination", join(directory, "artifacts")], { stdio: "ignore", windowsHide: true, env: { ...process.env, MLFLOW_DISABLE_TELEMETRY: "true", DO_NOT_TRACK: "true" } });
  for (let i = 0; i < 240; i++) { try { if ((await fetch(`${uri}/health`)).ok) return; } catch { /* starting */ } await new Promise(r => setTimeout(r, 500)); }
  throw new Error("MLflow server did not start");
}, 150_000);
// mlflow server starts a waitress child on Windows; the whole tree must go before its SQLite file can be removed.
const stop = (child: ChildProcess) => new Promise<void>(done => { if (process.platform !== "win32") { child.kill(); done(); return; } spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }).once("close", () => done()); });
afterAll(async () => { if (server) await stop(server); await pool.end(); if (directory) await rm(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 500 }).catch(() => undefined); }, 60_000);
beforeEach(truncate);

describe("MLflow configuration", () => {
  it("is off by default and accepts only an operator-configured http(s) server", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).mlflow).toBeNull();
    expect(mlConfiguration({ ML_MLFLOW_ENABLED: "true", ML_MLFLOW_TRACKING_URI: "file:///tmp/x" }).mlflow).toBeNull(); // ML itself disabled
    expect(() => mlConfiguration({ ML_ENABLED: "true", ML_MLFLOW_ENABLED: "true", ML_MLFLOW_TRACKING_URI: "file:///tmp/mlruns" })).toThrow(/http\(s\) URL/);
    expect(mlConfiguration({ ML_ENABLED: "true", ML_MLFLOW_ENABLED: "true", ML_MLFLOW_TRACKING_URI: "http://127.0.0.1:5000" }).mlflow).toMatchObject({ trackingUri: "http://127.0.0.1:5000", prefix: "meticule" });
  });
  it("returns 503 for MLflow routes when the feature is not configured", () => withStore(async store => {
    const f = await trainedModel(store);
    expect((await testApp(store, f.backends).request(`/ml/training-runs/${f.attempt.id}/mlflow-export`, json({}))).status).toBe(503);
  }));
});

describe.runIf(Boolean(python))("MLflow interoperability against a real tracking server", () => {
  const config = () => ({ python: python!, timeoutMs: 120_000, trackingUri: uri, token: "", prefix: "meticule-test" });

  it("exports parameters, metrics, tags and lineage once, reuses the result, and replaces an interrupted export", () => withStore(async store => {
    const f = await trainedModel(store); const service = new MlMlflowService(pool, store, config());
    const first = await service.export(manager, f.attempt.id, signal());
    expect(first).toMatchObject({ status: "COMPLETED", reused: false, attempts: 1, failure: null });
    expect(first.unsupported.map((u: { kind: string }) => u.kind)).toEqual(["model_artifact"]);
    const run = (await api(`/api/2.0/mlflow/runs/get?run_id=${first.mlflow_run_id}`)).run;
    const params = Object.fromEntries(run.data.params.map((p: { key: string; value: string }) => [p.key, p.value]));
    const tags = Object.fromEntries(run.data.tags.map((p: { key: string; value: string }) => [p.key, p.value]));
    expect(params).toMatchObject({ "hp.alpha": "3", "resolved.alpha": "3", backend: "test-p5", algorithm: "linear", seed: "7", dataset_version_id: f.dataset.id });
    expect(run.data.metrics.find((m: { key: string }) => m.key === "test.mae").value).toBeCloseTo(.01);
    expect(tags).toMatchObject({ "meticule.training_run_id": f.attempt.id, "meticule.dataset_sha256": f.dataset.content.sha256, "meticule.export_complete": "true" });
    expect(run.info.status).toBe("FINISHED");
    const lineage = await (await fetch(`${uri}/get-artifact?path=meticule/lineage.json&run_uuid=${first.mlflow_run_id}`)).json();
    expect(lineage).toMatchObject({ schema: "meticule-lineage-v1", datasetVersion: { id: f.dataset.id, sha256: f.dataset.content.sha256 }, experiment: { id: f.experiment.id } });
    expect(lineage.datasetVersion.content).toBeUndefined();

    const again = await service.export(manager, f.attempt.id, signal());
    expect(again).toMatchObject({ status: "COMPLETED", reused: true, mlflow_run_id: first.mlflow_run_id, attempts: 1 });
    expect(await runsTagged(first.mlflow_experiment_id, f.attempt.id)).toHaveLength(1);

    // An export interrupted after MLflow created its run (and before Meticule committed) leaves an incomplete tagged run.
    const other = await new MlTrackingService(pool, store, f.backends).reproduce(manager, f.job.id); await f.worker.tick();
    const second = (await f.training.getJob(manager, other.job.id)).attempts[0];
    const orphan = (await api("/api/2.0/mlflow/runs/create", { experiment_id: first.mlflow_experiment_id, tags: [{ key: "meticule.training_run_id", value: second.id }] })).run.info.run_id;
    const replaced = await service.export(manager, second.id, signal());
    expect(replaced.status).toBe("COMPLETED"); expect(replaced.mlflow_run_id).not.toBe(orphan);
    expect((await runsTagged(first.mlflow_experiment_id, second.id)).map((r: { info: { run_id: string } }) => r.info.run_id)).toEqual([replaced.mlflow_run_id]);
  }), 120_000);

  it("records a failed export without throwing, retries it, and keeps tenants and roles apart", () => withStore(async store => {
    const f = await trainedModel(store);
    const dead = new MlMlflowService(pool, store, { ...config(), trackingUri: `http://127.0.0.1:${await freePort()}`, timeoutMs: 60_000 });
    const failed = await dead.export(manager, f.attempt.id, signal());
    expect(failed).toMatchObject({ status: "FAILED", mlflow_run_id: null, attempts: 1, failure: { category: "backend_unavailable" } });
    expect(JSON.stringify(failed.failure)).not.toContain("127.0.0.1");
    expect((await dead.export(manager, f.attempt.id, signal())).attempts).toBe(2);
    await expect(new MlMlflowService(pool, store, config()).export(outsider, f.attempt.id, signal())).rejects.toThrow(/not found/);
    await expect(new MlMlflowService(pool, store, config()).export(viewer, f.attempt.id, signal())).rejects.toThrow(/role/);
    const app = testApp(store, f.backends, manager, { mlflow: { ...dead["config"] } });
    expect((await app.request(`/ml/training-runs/${f.attempt.id}/mlflow-export`, json({}))).status).toBe(502);
    expect(await (await app.request(`/ml/training-runs/${f.attempt.id}/mlflow-exports`)).json()).toHaveLength(1);
  }), 120_000);

  it("imports run metadata as data only, once per experiment", () => withStore(async store => {
    const f = await trainedModel(store); const service = new MlMlflowService(pool, store, config());
    const exported = await service.export(manager, f.attempt.id, signal());
    const target = await f.training.experiment(manager, "imported");
    const imported = await service.import(manager, { experimentId: target.id, mlflowRunId: exported.mlflow_run_id }, signal());
    expect(imported).toMatchObject({ reused: false, source: "mlflow", external_run_id: exported.mlflow_run_id });
    expect(imported.document).toMatchObject({ status: "FINISHED", params: { "hp.alpha": "3" }, tags: { "meticule.training_run_id": f.attempt.id } });
    expect(imported.document.metrics["test.mae"]).toBeCloseTo(.01);
    expect(Object.keys(imported.document.tags).some(k => k.startsWith("mlflow."))).toBe(false);
    expect((await service.import(manager, { experimentId: target.id, mlflowRunId: exported.mlflow_run_id }, signal())).reused).toBe(true);
    expect(await service.externalRuns(manager, target.id)).toHaveLength(1);
    await expect(service.import(manager, { experimentId: target.id, mlflowRunId: "../../etc" }, signal())).rejects.toThrow();
    // Runs Meticule exported for another tenant are refused, by experiment name and by identity tag.
    const foreignExperiment = (await api("/api/2.0/mlflow/experiments/create", { name: `meticule-test/${outsider.tenantId}/elsewhere` })).experiment_id;
    const foreign = (await api("/api/2.0/mlflow/runs/create", { experiment_id: foreignExperiment })).run.info.run_id;
    await expect(service.import(manager, { experimentId: target.id, mlflowRunId: foreign }, signal())).rejects.toThrow(/not found/);
    const tagged = (await api("/api/2.0/mlflow/runs/create", { experiment_id: "0", tags: [{ key: "meticule.tenant_id", value: outsider.tenantId }] })).run.info.run_id;
    await expect(service.import(manager, { experimentId: target.id, mlflowRunId: tagged }, signal())).rejects.toThrow(/not found/);
    const external = (await api("/api/2.0/mlflow/runs/create", { experiment_id: "0" })).run.info.run_id;
    expect((await service.import(manager, { experimentId: target.id, mlflowRunId: external }, signal())).reused).toBe(false);
    await expect(service.import(outsider, { experimentId: target.id, mlflowRunId: exported.mlflow_run_id }, signal())).rejects.toThrow(/not found/);
    const app = testApp(store, f.backends, manager, { mlflow: config() });
    expect((await app.request("/ml/mlflow/imports", json({ experimentId: target.id, mlflowRunId: "0".repeat(32) }))).status).toBe(502);
  }), 120_000);
});
