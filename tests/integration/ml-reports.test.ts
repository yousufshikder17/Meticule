import { createHash, randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import type { ChunkResult, MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import type { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import type { MetricValue, Row, Snapshot } from "../../src/ml/domain.js";
import { MlDeploymentService } from "../../src/ml/deployments.js";
import { MlExperimentService } from "../../src/ml/experiments.js";
import { ForecastingBackend } from "../../src/ml/forecasting-backend.js";
import { defineModel, registerModels } from "../../src/ml/model-catalog.js";
import { MlMonitoringService } from "../../src/ml/monitoring.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlReportService } from "../../src/ml/reports.js";
import { callRunner } from "../../src/ml/runner-process.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { ANOMALY_ROWS, SERIES_ROWS } from "../conformance/modality-rows.js";
import { json, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

const mlops = process.env.ML_TEST_MLOPS_PYTHON, sklearn = process.env.ML_TEST_PYTHON;
const config = () => ({ python: mlops!, timeoutMs: 120_000 });
const signal = () => new AbortController().signal;
beforeEach(truncate);
afterAll(() => pool.end());

registerModels([defineModel({ backend: "test-curves", id: "curve_mlp", taskType: "regression", displayName: "Iterative test model", parameters: { epochs: { kind: "int", min: 1, max: 50, default: 6 } }, capabilities: { scores: "none", scaleSensitive: false, seeded: true, iterative: true } })]);
/** Iterative in-process backend: loss 1/epoch (validation 1/epoch + 0.1), so the curve is known exactly. */
class CurveBackend implements MlBackend {
  readonly id = "test-curves";
  readonly capabilities = { artifactFormats: ["test-curve-v1"], checkpointFormats: ["test-curve-ckpt"], devices: ["cpu"], precisions: ["fp32"], preprocessing: [], iterative: true, tasks: ["regression"], inputs: ["number"] } as const;
  validate(): void {}
  async prepare(_s: Snapshot, rows: Row[]): Promise<PreparedData> { const all = rows.map((_, i) => i); return { indices: { train: all.slice(0, 36), validation: all.slice(36, 48), test: all.slice(48) }, environment: { protocol: "curve" } }; }
  async train(): Promise<TrainedModel> { throw new Error("iterative only"); }
  async trainChunk(_s: Snapshot, _r: Row[], _p: PreparedData, state: Buffer | null, epochs: number): Promise<ChunkResult> {
    const start = state ? Number(state.toString()) : 0, done = start + epochs;
    return { checkpoint: Buffer.from(String(done)), checkpointFormat: "test-curve-ckpt", checkpointSchema: 1, completedEpoch: done, environment: { protocol: "curve" },
      epochs: Array.from({ length: epochs }, (_, i) => ({ epoch: start + i + 1, trainLoss: 1 / (start + i + 1), validationLoss: 1 / (start + i + 1) + .1, elapsedSeconds: .01 })) };
  }
  async exportModel(): Promise<TrainedModel> { return { bytes: Buffer.from("curve"), format: "test-curve-v1", environment: { protocol: "curve" }, resolvedHyperparameters: {} }; }
  async evaluate(): Promise<MetricValue[]> { return ["train", "validation", "test"].map(p => ({ name: "mae", partition: p as MetricValue["partition"], value: .5, direction: "lower" })); }
  async predict(_s: Snapshot, _a: Buffer, rows: Row[]): Promise<unknown[]> { return rows.map(() => 0); }
}

/** Trains one job through the durable worker with a given backend and returns its completed attempt. */
async function train(store: PostgresArtifactStore, backend: MlBackend, columns: { name: string; type: string; nullable?: boolean }[], rows: Row[], target: string | undefined, features: string[], spec: Record<string, unknown>) {
  const backends = new Map([[backend.id, backend]]), training = new MlTrainingService(pool, store);
  const dataset = await training.ingest(manager, { name: `d-${randomUUID().slice(0, 6)}`, schema: { id: randomUUID(), columns, ...(target ? { target } : {}) }, rows });
  const pipeline = await training.pipeline(manager, { name: "p", definition: { features, steps: [] } });
  const experiment = await training.experiment(manager, `e-${randomUUID().slice(0, 6)}`);
  const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id, seed: 4, ...spec } });
  const worker = new LifecycleWorker(pool, { workerId: "reports-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, backends));
  while (await worker.tick());
  const result = await training.getJob(manager, job.id);
  expect(result.job.status).toBe("COMPLETED");
  return { job: result.job, attempt: result.attempts.at(-1), metrics: result.metrics as { name: string; partition: string; value: number }[], backends };
}
const metric = (metrics: { name: string; partition: string; value: number }[], name: string) => metrics.find(m => m.name === name && m.partition === "test")!.value;
const png = (bytes: Buffer) => ({ magic: bytes.subarray(0, 8).toString("hex"), width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) });

describe.runIf(Boolean(mlops))("Matplotlib reports", () => {
  it("renders training curves, comparisons, optimization history and drift as valid, deterministic PNG and SVG", () => withStore(async store => {
    const curves = await train(store, new CurveBackend(), [{ name: "x", type: "number" }, { name: "y", type: "number" }], Array.from({ length: 60 }, (_, i) => ({ x: i, y: i })), "y", ["x"], { algorithm: "curve_mlp", split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2 } });
    const reports = new MlReportService(pool, store, curves.backends, config());
    const first = await reports.create(manager, { kind: "training_curves", jobId: curves.job.id }, signal());
    expect(first.summary).toMatchObject({ epochs: 6, finalTrainLoss: 1 / 6, bestValidationLoss: 1 / 6 + .1 });
    expect(first.producer).toEqual({ renderer: "matplotlib", matplotlib: "3.11.2" });
    const image = (await reports.content(viewer, first.id)).bytes;
    expect(png(image)).toEqual({ magic: "89504e470d0a1a0a", width: 900, height: 540 });
    const second = await reports.create(manager, { kind: "training_curves", jobId: curves.job.id }, signal());
    expect(second.content.sha256).toBe(first.content.sha256); // same request, same environment, same bytes
    const svg = (await reports.content(manager, (await reports.create(manager, { kind: "training_curves", jobId: curves.job.id, format: "svg", width: 400, height: 300 }, signal())).id)).bytes.toString("utf8");
    expect(svg.startsWith("<?xml")).toBe(true); expect(svg).toContain("<svg"); expect(svg).not.toMatch(/<script|<foreignObject|href="http/i);

    const f = await trainedModel(store), g = await trainedModel(store, undefined, { alpha: 5 });
    const compared = await reports.create(manager, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, g.attempt.id], metric: "mae", partition: "test" }, signal());
    expect(compared.summary.values[g.attempt.id]).toBeCloseTo(.21); expect(compared.summary.direction).toBe("lower");
    await expect(reports.create(manager, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, g.attempt.id], metric: "r2", partition: "test" }, signal())).rejects.toThrow(/needs the test\/r2/);

    // A study name full of markup, mathtext and control characters is drawn as literal text and cannot reach the SVG as markup.
    const experiments = new MlExperimentService(pool, store, f.backends, { optuna: config() });
    const study = await experiments.optuna(manager, { name: "$\\frac{1}{2}$ <script>alert(1)</script>\u0007", experimentId: f.experiment.id, datasetVersionId: f.dataset.id, featurePipelineId: f.pipeline.id, backend: "test-p5", taskType: "regression", seed: 3, evaluation: { primaryMetric: "mae" }, algorithm: "linear", space: { alpha: { type: "float", min: 0, max: 10 } }, maxTrials: 3 });
    // The study's worker needs the Optuna configuration to ask for the next trial when one finishes.
    const studyWorker = new LifecycleWorker(pool, { workerId: "reports-study", leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, f.backends, { optuna: config() }));
    for (let i = 0; i < 5 && await studyWorker.tick(); i++);
    const history = await reports.create(manager, { kind: "optimization_history", orchestrationId: study.orchestration.id, format: "svg" }, signal());
    expect(history.summary).toMatchObject({ completedTrials: 3, direction: "minimize" });
    const historySvg = (await reports.content(manager, history.id)).bytes.toString("utf8");
    expect(historySvg).not.toMatch(/<script/i);

    const registry = await f.models.registry(manager, { name: "drifting" });
    const version = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id }); await f.models.transition(manager, version.id, "READY");
    const deployments = new MlDeploymentService(pool, store, f.backends), d = await deployments.create(manager, { name: "d" });
    await deployments.addRevision(manager, d.id, { modelVersionId: version.id }, signal()); await deployments.activate(manager, d.id, {});
    const monitoring = new MlMonitoringService(pool, store);
    await deployments.predict(manager, d.id, Array.from({ length: 36 }, (_, x) => ({ x })), signal());
    await monitoring.evaluateDrift(manager, d.id, { minSamples: 1 });
    await deployments.predict(manager, d.id, Array.from({ length: 100 }, (_, x) => ({ x: x + 500 })), signal());
    const latest = await monitoring.evaluateDrift(manager, d.id, { minSamples: 1 });
    const drift = await reports.create(manager, { kind: "drift", deploymentId: d.id }, signal());
    expect(drift.sources.evaluationId).toBe(latest.id); expect(drift.summary.status).toBe("significant"); expect(drift.summary.psi.x).toBeGreaterThan(.25);
    const trend = await reports.create(manager, { kind: "drift_trend", deploymentId: d.id }, signal());
    expect(trend.summary.statuses).toEqual(["stable", "significant"]);
    await expect(reports.create(manager, { kind: "training_curves", jobId: f.job.id }, signal())).rejects.toThrow(/no epoch history/);
    expect((await reports.list(viewer, "drift")).map(r => r.id)).toEqual([drift.id]);
  }), 600_000);

  it("validates chart data in the runner and keeps tenants and roles apart", () => withStore(async store => {
    const render = (chart: unknown, extra: Record<string, unknown> = {}) => callRunner({ label: "reports", python: mlops!, script: resolve("ml/report_runner.py"), timeoutMs: 60_000, signal: signal(), request: { command: "render", format: "png", width: 400, height: 300, title: "t", chart, ...extra } });
    for (const [chart, extra] of [[{ type: "lines", series: [{ label: "a", x: [1, 2], y: [1, "NaN"] }] }, {}], [{ type: "lines", series: [{ label: "a", x: [1], y: [1, 2] }] }, {}],
      [{ type: "exec", code: "import os" }, {}], [{ type: "bars", categories: ["a"], series: [{ label: "a", values: [1] }] }, { width: 99_999 }], [{ type: "histogram", bins: 1, series: [{ label: "a", values: [1] }] }, {}]] as const) {
      await expect(render(chart, extra)).rejects.toMatchObject({ category: "invalid_request", runnerCode: "invalid_report_data" });
    }
    const f = await trainedModel(store), reports = new MlReportService(pool, store, f.backends, config());
    const report = await reports.create(manager, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, randomUUID()], metric: "mae", partition: "test" }, signal()).catch(e => e);
    expect(report).toBeInstanceOf(Error); // an unknown run is not found
    const g = await trainedModel(store, undefined, { alpha: 4 });
    const ok = await reports.create(manager, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, g.attempt.id], metric: "mae", partition: "test" }, signal());
    await expect(reports.create(viewer, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, g.attempt.id], metric: "mae", partition: "test" }, signal())).rejects.toThrow(/role/);
    await expect(reports.create(outsider, { kind: "metric_comparison", trainingRunIds: [f.attempt.id, g.attempt.id], metric: "mae", partition: "test" }, signal())).rejects.toThrow(/not found/);
    await expect(reports.get(outsider, ok.id)).rejects.toThrow(/not found/);
    const app = testApp(store, f.backends, viewer, { reports: config() });
    const content = await app.request(`/ml/reports/${ok.id}/content`);
    expect(content.status).toBe(200); expect(content.headers.get("content-type")).toBe("image/png");
    expect(content.headers.get("x-content-type-options")).toBe("nosniff"); expect(content.headers.get("content-security-policy")).toContain("sandbox");
    expect((await testApp(store, f.backends).request("/ml/reports", json({ kind: "drift", deploymentId: randomUUID() }))).status).toBe(503);
    await expect(pool.query("UPDATE ml_reports SET summary='{}' WHERE id=$1", [ok.id])).rejects.toThrow();
  }), 300_000);
});

describe.runIf(Boolean(mlops && sklearn))("evaluation reports from real models", () => {
  it("draws confusion matrices, residuals, anomaly scores, ROC/PR curves and forecasts consistent with the stored metrics", () => withStore(async store => {
    const sk = new SklearnBackend(sklearn!), split = (stratify: boolean) => ({ id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify });
    const rows = Array.from({ length: 80 }, (_, i) => { const a = Math.sin(i) * 3, b = Math.cos(i * 1.3) * 2; return { a, b, label: 2 * a - b > 0 ? "high" : "low", y: 2 * a - b + Math.sin(i * 7) * .3 }; });
    const columns = [{ name: "a", type: "number" }, { name: "b", type: "number" }];
    const cls = await train(store, sk, [...columns, { name: "label", type: "string" }], rows.map(({ y: _y, ...r }) => r), "label", ["a", "b"], { algorithm: "logistic_regression", split: split(true), evaluation: { metrics: ["accuracy"] } });
    const reports = new MlReportService(pool, store, new Map<string, MlBackend>([["sklearn", sk], ["forecasting", new ForecastingBackend(sklearn!)]]), config());
    const cm = await reports.create(manager, { kind: "confusion_matrix", trainingRunId: cls.attempt.id }, signal());
    expect(cm.summary.classes).toEqual(["high", "low"]); expect(cm.summary.matrix.flat().reduce((a: number, b: number) => a + b)).toBe(16);
    expect(cm.summary.accuracy).toBeCloseTo(metric(cls.metrics, "accuracy"), 12);
    await expect(reports.create(manager, { kind: "roc_curve", trainingRunId: cls.attempt.id }, signal())).rejects.toThrow(/need model scores/);
    await expect(reports.create(manager, { kind: "residuals", trainingRunId: cls.attempt.id }, signal())).rejects.toThrow(/regression model/);

    const reg = await train(store, sk, [...columns, { name: "y", type: "number" }], rows.map(({ label: _l, ...r }) => r), "y", ["a", "b"], { algorithm: "ridge", split: split(false), evaluation: { metrics: ["mae"] } });
    const residuals = await reports.create(manager, { kind: "residuals", trainingRunId: reg.attempt.id }, signal());
    expect(residuals.summary.mae).toBeCloseTo(metric(reg.metrics, "mae"), 9);

    const anomaly = await train(store, sk, [...columns, { name: "outlier", type: "boolean" }], ANOMALY_ROWS, "outlier", ["a", "b"], { algorithm: "isolation_forest", hyperparameters: { contamination: .1 }, split: split(true), evaluation: { metrics: ["roc_auc", "average_precision"] } });
    const roc = await reports.create(manager, { kind: "roc_curve", trainingRunId: anomaly.attempt.id }, signal());
    expect(roc.summary.auc).toBeCloseTo(metric(anomaly.metrics, "roc_auc"), 9);
    const pr = await reports.create(manager, { kind: "precision_recall", trainingRunId: anomaly.attempt.id }, signal());
    expect(pr.summary.averagePrecision).toBeCloseTo(metric(anomaly.metrics, "average_precision"), 9);
    expect((await reports.create(manager, { kind: "anomaly_scores", trainingRunId: anomaly.attempt.id }, signal())).summary).toMatchObject({ rows: 24, labelled: true });

    const series = await train(store, new ForecastingBackend(sklearn!), [{ name: "at", type: "timestamp" }, { name: "y", type: "number" }], SERIES_ROWS, "y", ["at"], { algorithm: "ridge_forecaster", hyperparameters: { lags: 4, windows: [6], horizon: 5 }, split: { id: randomUUID(), strategy: "chronological", timeColumn: "at", train: .6, validation: .2, test: .2 }, evaluation: { metrics: ["mae"] } });
    const forecast = await reports.create(manager, { kind: "forecast", trainingRunId: series.attempt.id, format: "svg" }, signal());
    expect(forecast.summary).toMatchObject({ horizon: 5, compared: 5 }); expect(forecast.summary.mae).toBeGreaterThanOrEqual(0);
    expect(createHash("sha256").update((await reports.content(manager, forecast.id)).bytes).digest("hex")).toBe(forecast.content.sha256);
  }), 600_000);
});
