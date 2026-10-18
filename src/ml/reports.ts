import type pg from "pg";
import { resolve } from "node:path";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { MlBackendFailure } from "./contract.js";
import { TrainingSnapshot, type Row } from "./domain.js";
import { timestampMs } from "./media.js";
import { MlExperimentService } from "./experiments.js";
import { sequence } from "./lineage.js";
import { getModel } from "./model-catalog.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { callRunner } from "./runner-process.js";
import { predictionCount, validatePredictions } from "./tasks.js";

export interface ReportsConfig { python: string; timeoutMs: number }
/** Test rows a prediction-based report scores at most; forecasting history is bounded the same way. */
export const MAX_REPORT_ROWS = 5000;
const Id = z.uuid(), common = { format: z.enum(["png", "svg"]).default("png"), width: z.int().min(320).max(1600).default(900), height: z.int().min(240).max(1200).default(540) };
export const ReportInput = z.discriminatedUnion("kind", [
  z.strictObject({ kind: z.literal("training_curves"), jobId: Id, ...common }),
  z.strictObject({ kind: z.literal("metric_comparison"), trainingRunIds: z.array(Id).min(2).max(10).refine(x => new Set(x).size === x.length, "Duplicate runs"), metric: z.string().min(1).max(60), partition: z.enum(["train", "validation", "test", "cv"]), ...common }),
  z.strictObject({ kind: z.literal("optimization_history"), orchestrationId: Id, ...common }),
  ...(["confusion_matrix", "residuals", "roc_curve", "precision_recall", "forecast", "anomaly_scores"] as const).map(kind => z.strictObject({ kind: z.literal(kind), trainingRunId: Id, ...common })),
  z.strictObject({ kind: z.literal("drift"), deploymentId: Id, evaluationId: Id.optional(), ...common }),
  z.strictObject({ kind: z.literal("drift_trend"), deploymentId: Id, ...common }),
]);
type Input = z.infer<typeof ReportInput>;
type Chart = Record<string, unknown> & { type: "lines" | "bars" | "heatmap" | "scatter" | "histogram" };
interface Built { title: string; chart: Chart; sources: Record<string, unknown>; summary: Record<string, unknown> }
const label = (v: unknown) => String(v).replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 60);

/** Confusion counts with classes in sorted order; rows are actual classes, columns predicted. */
export function confusionMatrix(actual: unknown[], predicted: unknown[]) {
  const classes = [...new Set([...actual, ...predicted].map(String))].sort();
  const matrix = classes.map(() => classes.map(() => 0));
  actual.forEach((a, i) => { matrix[classes.indexOf(String(a))]![classes.indexOf(String(predicted[i]))]!++; });
  return { classes, matrix, accuracy: actual.length ? actual.filter((a, i) => String(a) === String(predicted[i])).length / actual.length : 0 };
}
/** Thresholds at each distinct score, highest first (as scikit-learn's roc_curve and precision_recall_curve). */
function sweep(scores: number[], labels: boolean[]) {
  const order = scores.map((s, i) => [s, labels[i]!] as const).sort((a, b) => b[0] - a[0]);
  const positives = labels.filter(Boolean).length, negatives = labels.length - positives, steps: { tp: number; fp: number }[] = [];
  let tp = 0, fp = 0;
  order.forEach(([s, positive], i) => { if (positive) tp++; else fp++; if (i === order.length - 1 || order[i + 1]![0] !== s) steps.push({ tp, fp }); });
  return { steps, positives, negatives };
}
export function rocCurve(scores: number[], labels: boolean[]) {
  const { steps, positives, negatives } = sweep(scores, labels);
  if (!positives || !negatives) throw new ConflictError("ROC needs both positive and negative labels");
  const points = [{ fpr: 0, tpr: 0 }, ...steps.map(s => ({ fpr: s.fp / negatives, tpr: s.tp / positives }))];
  const auc = points.slice(1).reduce((area, p, i) => area + (p.fpr - points[i]!.fpr) * (p.tpr + points[i]!.tpr) / 2, 0);
  return { points, auc };
}
export function precisionRecallCurve(scores: number[], labels: boolean[]) {
  const { steps, positives } = sweep(scores, labels);
  if (!positives) throw new ConflictError("Precision-recall needs positive labels");
  const points = steps.map(s => ({ recall: s.tp / positives, precision: s.tp / (s.tp + s.fp) }));
  const averagePrecision = points.reduce((ap, p, i) => ap + (p.recall - (i ? points[i - 1]!.recall : 0)) * p.precision, 0);
  return { points: [{ recall: 0, precision: 1 }, ...points], averagePrecision };
}

/**
 * Matplotlib reports. The platform computes every number from durable records (epochs, metrics, trials, drift
 * evaluations) or from the registered model's own predictions on its held-out test rows; the runner only draws them.
 */
export class MlReportService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>, private readonly config: ReportsConfig, private readonly script = resolve(process.cwd(), "ml/report_runner.py")) {}

  async create(principal: Principal, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal, true); const input = ReportInput.parse(raw);
    const built = await this.build(principal.tenantId, input, signal);
    const { kind: _kind, format, width, height, ...subject } = input;
    const rendered = z.object({ image: z.string().min(1), matplotlib: z.string().max(40), bytes: z.int().positive() }).safeParse(await callRunner({ label: "reports", python: this.config.python, script: this.script, timeoutMs: this.config.timeoutMs, signal,
      request: { command: "render", format, width, height, title: built.title, chart: built.chart } }));
    if (!rendered.success) throw new MlBackendFailure("invalid_response", "Report runner response did not match the contract");
    const content = await this.artifacts.put(principal.tenantId, Buffer.from(rendered.data.image, "base64"), format === "png" ? "image/png" : "image/svg+xml");
    return transaction(this.pool, async c => {
      const r = await c.query("INSERT INTO ml_reports(tenant_id,kind,format,spec,sources,summary,content,producer,created_by) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *",
        [principal.tenantId, input.kind, format, JSON.stringify({ ...subject, width, height, title: built.title }), JSON.stringify(built.sources), JSON.stringify(built.summary), JSON.stringify(content), JSON.stringify({ renderer: "matplotlib", matplotlib: rendered.data.matplotlib }), principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.report_created", { reportId: r.rows[0].id, kind: input.kind }); return r.rows[0];
    });
  }

  async list(principal: Principal, kind?: string) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT id,kind,format,spec,sources,summary,producer,created_by,created_at FROM ml_reports WHERE tenant_id=$1 AND ($2::text IS NULL OR kind=$2) ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, kind ?? null])).rows;
  }
  async get(principal: Principal, id: string) {
    authorizeMl(principal);
    const r = (await this.pool.query("SELECT * FROM ml_reports WHERE tenant_id=$1 AND id=$2", [principal.tenantId, id])).rows[0];
    if (!r) throw new NotFoundError("Report not found"); return r;
  }
  /** Image bytes, re-verified against their checksum. */
  async content(principal: Principal, id: string) {
    const r = await this.get(principal, id);
    return { bytes: await this.artifacts.get(principal.tenantId, r.content), mediaType: r.format === "png" ? "image/png" : "image/svg+xml" };
  }

  private async build(tenantId: string, input: Input, signal: AbortSignal): Promise<Built> {
    switch (input.kind) {
      case "training_curves": {
        const job = (await this.pool.query("SELECT snapshot FROM ml_training_jobs WHERE tenant_id=$1 AND id=$2", [tenantId, input.jobId])).rows[0];
        if (!job) throw new NotFoundError("Training job not found");
        const epochs = (await this.pool.query("SELECT DISTINCT ON (e.epoch) e.epoch,e.train_loss,e.validation_loss,t.attempt FROM ml_training_epochs e JOIN ml_training_runs t ON t.tenant_id=e.tenant_id AND t.id=e.training_run_id WHERE t.tenant_id=$1 AND t.job_id=$2 ORDER BY e.epoch,t.attempt DESC", [tenantId, input.jobId])).rows;
        if (!epochs.length) throw new ConflictError("The job has no epoch history; only iterative models record one");
        const validation = epochs.filter(e => e.validation_loss !== null);
        return { title: `Training curves: ${label(job.snapshot.algorithm)}`, sources: { jobId: input.jobId, attempts: [...new Set(epochs.map(e => e.attempt))] },
          summary: { epochs: epochs.length, finalTrainLoss: epochs.at(-1).train_loss, bestValidationLoss: validation.length ? Math.min(...validation.map(e => e.validation_loss)) : null },
          chart: { type: "lines", xLabel: "epoch", yLabel: "loss", series: [{ label: "train", x: epochs.map(e => e.epoch), y: epochs.map(e => e.train_loss) }, ...(validation.length ? [{ label: "validation", x: validation.map(e => e.epoch), y: validation.map(e => e.validation_loss) }] : [])] } };
      }
      case "metric_comparison": {
        const rows = (await this.pool.query("SELECT t.id,t.attempt,t.snapshot->>'algorithm' AS algorithm,m.value,m.direction FROM ml_training_runs t LEFT JOIN ml_metrics m ON m.tenant_id=t.tenant_id AND m.training_run_id=t.id AND m.name=$3 AND m.partition=$4 AND m.fold IS NULL WHERE t.tenant_id=$1 AND t.id=ANY($2::uuid[])", [tenantId, input.trainingRunIds, input.metric, input.partition])).rows;
        if (rows.length !== input.trainingRunIds.length) throw new NotFoundError("Training run not found");
        if (rows.some(r => r.value === null)) throw new ConflictError(`Every run needs the ${input.partition}/${input.metric} metric`);
        const ordered = input.trainingRunIds.map(id => rows.find(r => r.id === id)!);
        return { title: `${label(input.metric)} on ${input.partition}`, sources: { trainingRunIds: input.trainingRunIds },
          summary: { values: Object.fromEntries(ordered.map(r => [r.id, r.value])), direction: ordered[0].direction },
          chart: { type: "bars", yLabel: `${label(input.metric)} (${ordered[0].direction} is better)`, categories: ordered.map(r => `${label(r.algorithm)} ${r.id.slice(0, 8)}`), series: [{ label: input.metric, values: ordered.map(r => r.value) }] } };
      }
      case "optimization_history": {
        const view = await new MlExperimentService(this.pool, this.artifacts, this.backends).read(tenantId, input.orchestrationId, this.pool, "optuna");
        const study = view.study!, done = study.trials.filter(t => t.state === "COMPLETE");
        if (!done.length) throw new ConflictError("The study has no completed trials yet");
        let best = done[0]!.value!; const running = done.map(t => (best = study.direction === "maximize" ? Math.max(best, t.value!) : Math.min(best, t.value!)));
        return { title: `Optimization history: ${label(view.orchestration.name)}`, sources: { orchestrationId: input.orchestrationId, trials: study.trials.length },
          summary: { completedTrials: done.length, best: study.best, direction: study.direction, objective: study.objective },
          chart: { type: "lines", xLabel: "trial", yLabel: `${label(study.objective.metric)} (${study.objective.partition})`, series: [{ label: "trial value", x: done.map(t => t.number), y: done.map(t => t.value!), style: "points" }, { label: "best so far", x: done.map(t => t.number), y: running, style: "step" }] } };
      }
      case "drift": case "drift_trend": {
        if (!(await this.pool.query("SELECT 1 FROM ml_deployments WHERE tenant_id=$1 AND id=$2", [tenantId, input.deploymentId])).rowCount) throw new NotFoundError("Deployment not found");
        const evaluations = (await this.pool.query("SELECT * FROM ml_drift_evaluations WHERE tenant_id=$1 AND deployment_id=$2 AND ($3::uuid IS NULL OR id=$3) ORDER BY created_at DESC,id DESC LIMIT 50", [tenantId, input.deploymentId, input.kind === "drift" ? input.evaluationId ?? null : null])).rows.reverse();
        if (!evaluations.length) throw new ConflictError("The deployment has no drift evaluations");
        if (input.kind === "drift") {
          const e = evaluations.at(-1), items = [...e.features, ...(e.output ? [e.output] : [])];
          return { title: `Drift (PSI) for evaluation ${e.id.slice(0, 8)}: ${e.status}`, sources: { deploymentId: input.deploymentId, evaluationId: e.id, modelVersionId: e.model_version_id },
            summary: { status: e.status, observedRows: Number(e.observed_rows), psi: Object.fromEntries(items.map((f: { name: string; psi: number }) => [f.name, f.psi])), thresholds: e.thresholds },
            chart: { type: "bars", yLabel: "population stability index", categories: items.map((f: { name: string }) => label(f.name)), series: [{ label: "PSI", values: items.map((f: { psi: number }) => f.psi) }] } };
        }
        const worst = evaluations.map(e => Math.max(...[...e.features, ...(e.output ? [e.output] : [])].map((f: { psi: number }) => f.psi), 0));
        return { title: "Drift trend (highest feature PSI per evaluation)", sources: { deploymentId: input.deploymentId, evaluationIds: evaluations.map(e => e.id) },
          summary: { evaluations: evaluations.length, statuses: evaluations.map(e => e.status), maxPsi: worst },
          chart: { type: "lines", xLabel: "evaluation (oldest first)", yLabel: "highest PSI", series: [{ label: "highest PSI", x: worst.map((_, i) => i + 1), y: worst }, { label: "moderate", x: [1, Math.max(2, worst.length)], y: [evaluations.at(-1).thresholds.psiModerate, evaluations.at(-1).thresholds.psiModerate], style: "dashed" }, { label: "significant", x: [1, Math.max(2, worst.length)], y: [evaluations.at(-1).thresholds.psiSignificant, evaluations.at(-1).thresholds.psiSignificant], style: "dashed" }] } };
      }
      default: return this.fromPredictions(tenantId, input, signal);
    }
  }

  /** Reports that need the model's own predictions on its held-out test rows (at most MAX_REPORT_ROWS). */
  private async fromPredictions(tenantId: string, input: Extract<Input, { trainingRunId: string }>, signal: AbortSignal): Promise<Built> {
    const t = (await this.pool.query("SELECT t.id,t.status,t.snapshot,t.environment,t.split_indices,a.content FROM ml_training_runs t JOIN ml_artifacts a ON a.tenant_id=t.tenant_id AND a.training_run_id=t.id AND a.kind='model' WHERE t.tenant_id=$1 AND t.id=$2 AND t.status='COMPLETED'", [tenantId, input.trainingRunId])).rows[0];
    if (!t) throw new NotFoundError("Completed training run not found");
    const snapshot = TrainingSnapshot.parse(t.snapshot), model = getModel(snapshot.backend, snapshot.algorithm), task = model.taskType;
    const need: Record<string, string> = { confusion_matrix: "classification", residuals: "regression", forecast: "forecasting", anomaly_scores: "anomaly_detection", roc_curve: "anomaly_detection", precision_recall: "anomaly_detection" };
    if (need[input.kind] !== task) throw new ConflictError(input.kind === "roc_curve" || input.kind === "precision_recall"
      ? "ROC and precision-recall curves need model scores; Meticule models expose scores for anomaly detection only" : `${input.kind} reports need a ${need[input.kind]} model`);
    const backend = this.backends.get(snapshot.backend); if (!backend) throw new ConflictError("The model's backend is not configured");
    const [data, artifact] = await sequence([async () => JSON.parse((await this.artifacts.get(tenantId, snapshot.dataset.content)).toString("utf8")) as Row[], () => this.artifacts.get(tenantId, t.content)]);
    const target = snapshot.dataset.schema.target, sources = { trainingRunId: t.id, datasetVersionId: snapshot.datasetVersionId, datasetSha256: snapshot.dataset.content.sha256, modelSha256: t.content.sha256 };
    const predict = async (rows: Row[]) => validatePredictions(task, await backend.predict(snapshot, artifact, rows, t.environment, signal), predictionCount(model, snapshot, rows.length), t.environment);
    if (task === "forecasting") {
      const time = snapshot.split.strategy === "chronological" ? snapshot.split.timeColumn : snapshot.pipeline.definition.features[0]!;
      const byTime = (indices: number[]) => indices.map(i => data[i]!).sort((a, b) => (timestampMs(String(a[time])) ?? 0) - (timestampMs(String(b[time])) ?? 0));
      const history = byTime([...t.split_indices.train, ...t.split_indices.validation]).slice(-MAX_REPORT_ROWS), observed = byTime(t.split_indices.test);
      const forecast = await predict(history) as { step: number; value: number }[];
      const tail = history.slice(-50).map(r => Number(r[target!])), compared = observed.slice(0, forecast.length).map(r => Number(r[target!]));
      const errors = compared.map((v, i) => Math.abs(v - forecast[i]!.value));
      return { title: `Forecast vs observed: ${label(snapshot.algorithm)}`, sources, summary: { horizon: forecast.length, compared: compared.length, mae: errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null },
        chart: { type: "lines", xLabel: "step (0 = last training observation)", yLabel: label(target), series: [{ label: "history", x: tail.map((_, i) => i - tail.length + 1), y: tail }, { label: "forecast", x: forecast.map(f => f.step), y: forecast.map(f => f.value), style: "dashed" }, ...(compared.length ? [{ label: "observed", x: compared.map((_, i) => i + 1), y: compared, style: "points" }] : [])] } };
    }
    const test = (t.split_indices.test as number[]).slice(0, MAX_REPORT_ROWS).map(i => data[i]!);
    if (!test.length) throw new ConflictError("The training run has no test rows");
    const outputs = await predict(test.map(r => Object.fromEntries(Object.entries(r).filter(([k]) => k !== target)) as Row));
    const actual = target ? test.map(r => r[target]) : null;
    if (input.kind === "confusion_matrix") {
      const cm = confusionMatrix(actual!, outputs);
      return { title: `Confusion matrix: ${label(snapshot.algorithm)} (test)`, sources, summary: { rows: test.length, ...cm },
        chart: { type: "heatmap", xLabel: "predicted", yLabel: "actual", matrix: cm.matrix, xTicks: cm.classes.map(label), yTicks: cm.classes.map(label), annotate: true } };
    }
    if (input.kind === "residuals") {
      const predicted = outputs as number[], residuals = predicted.map((p, i) => Number(actual![i]) - p);
      const mae = residuals.reduce((a, r) => a + Math.abs(r), 0) / residuals.length;
      return { title: `Residuals: ${label(snapshot.algorithm)} (test)`, sources, summary: { rows: test.length, mae, meanResidual: residuals.reduce((a, b) => a + b, 0) / residuals.length },
        chart: { type: "scatter", xLabel: "predicted", yLabel: "actual - predicted", x: predicted, y: residuals, hline: 0 } };
    }
    const scores = (outputs as { score: number }[]).map(o => o.score);
    if (input.kind === "anomaly_scores") {
      const groups = actual ? [{ label: "normal", values: scores.filter((_, i) => actual[i] === false) }, { label: "anomalous", values: scores.filter((_, i) => actual[i] === true) }] : [{ label: "score", values: scores }];
      return { title: `Anomaly scores: ${label(snapshot.algorithm)} (test)`, sources, summary: { rows: test.length, flagged: (outputs as { anomaly: boolean }[]).filter(o => o.anomaly).length, labelled: Boolean(actual) },
        chart: { type: "histogram", xLabel: "anomaly score (higher is more anomalous)", yLabel: "rows", bins: 30, series: groups } };
    }
    if (!actual || actual.some(v => typeof v !== "boolean")) throw new ConflictError("ROC and precision-recall curves need boolean anomaly labels");
    if (input.kind === "roc_curve") {
      const roc = rocCurve(scores, actual as boolean[]);
      return { title: `ROC: ${label(snapshot.algorithm)} (test), AUC ${roc.auc.toFixed(3)}`, sources, summary: { rows: test.length, auc: roc.auc, points: roc.points.length },
        chart: { type: "lines", xLabel: "false positive rate", yLabel: "true positive rate", diagonal: true, series: [{ label: "ROC", x: roc.points.map(p => p.fpr), y: roc.points.map(p => p.tpr) }] } };
    }
    const pr = precisionRecallCurve(scores, actual as boolean[]);
    return { title: `Precision-recall: ${label(snapshot.algorithm)} (test), AP ${pr.averagePrecision.toFixed(3)}`, sources, summary: { rows: test.length, averagePrecision: pr.averagePrecision, points: pr.points.length },
      chart: { type: "lines", xLabel: "recall", yLabel: "precision", series: [{ label: "precision-recall", x: pr.points.map(p => p.recall), y: pr.points.map(p => p.precision), style: "step" }] } };
  }
}
