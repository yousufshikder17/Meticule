import { ConflictError } from "../domain/errors.js";
import type { ModelDefinition } from "./model-catalog.js";
import type { TaskType } from "./tasks.js";

// "higher" = maximize, "lower" = minimize. Ranking reads this, never the metric name.
export type MetricDirection = "higher" | "lower";
/**
 * `labels`: needs the target (refused on unlabelled data). `needs`: the per-class or anomaly score the estimator must expose.
 * `diagnostic`: describes a model's outputs (threshold transfer, anisotropy), not its quality, so it never ranks models.
 * ml/ml_runtime_common.py mirrors tasks, direction, labels and diagnostic (a unit test compares the two).
 */
export interface MetricDefinition { tasks: readonly TaskType[]; direction: MetricDirection; needs?: "probability" | "scores"; labels: boolean; diagnostic?: true }
const definitions = {
  accuracy: { tasks: ["classification"], direction: "higher", labels: true },
  precision: { tasks: ["classification"], direction: "higher", labels: true }, // weighted by class support, zero_division=0
  recall: { tasks: ["classification"], direction: "higher", labels: true },
  f1: { tasks: ["classification"], direction: "higher", labels: true },
  // classification: binary, or multiclass one-vs-rest with probabilities. anomaly_detection: anomaly scores against the labels.
  roc_auc: { tasks: ["classification", "anomaly_detection"], direction: "higher", needs: "scores", labels: true },
  log_loss: { tasks: ["classification"], direction: "lower", needs: "probability", labels: true },
  mae: { tasks: ["regression", "forecasting"], direction: "lower", labels: true },
  mse: { tasks: ["regression"], direction: "lower", labels: true },
  rmse: { tasks: ["regression", "forecasting"], direction: "lower", labels: true },
  r2: { tasks: ["regression"], direction: "higher", labels: true },
  smape: { tasks: ["forecasting"], direction: "lower", labels: true }, // 0..200 percent; a term with |actual| + |forecast| = 0 counts 0
  mase: { tasks: ["forecasting"], direction: "lower", labels: true }, // scaled by the in-sample seasonal-naive error; a zero scale is refused
  average_precision: { tasks: ["anomaly_detection"], direction: "higher", needs: "scores", labels: true },
  // Positive class = anomaly; zero_division = 0.
  anomaly_precision: { tasks: ["anomaly_detection"], direction: "higher", labels: true },
  anomaly_recall: { tasks: ["anomaly_detection"], direction: "higher", labels: true },
  anomaly_f1: { tasks: ["anomaly_detection"], direction: "higher", labels: true },
  // |flagged fraction - contamination|: whether the training threshold transfers, NOT detection quality.
  threshold_calibration_error: { tasks: ["anomaly_detection"], direction: "lower", labels: false, diagnostic: true },
  // Leave-one-out cosine nearest neighbours within the partition; relevant = same label.
  retrieval_precision_at_1: { tasks: ["embedding"], direction: "higher", labels: true },
  retrieval_mrr: { tasks: ["embedding"], direction: "higher", labels: true },
  mean_pairwise_cosine: { tasks: ["embedding"], direction: "lower", labels: false, diagnostic: true }, // anisotropy
  // Corpus-level: summed edits over summed reference length, on lower-cased whitespace-collapsed text; refused when empty.
  wer: { tasks: ["transcription"], direction: "lower", labels: true },
  cer: { tasks: ["transcription"], direction: "lower", labels: true },
} as const satisfies Record<string, MetricDefinition>;
export type MetricName = keyof typeof definitions;
export const METRICS: Readonly<Record<MetricName, MetricDefinition>> = definitions;
export const isMetricName = (name: string): name is MetricName => Object.hasOwn(METRICS, name);

// What a snapshot without `evaluation.metrics` computes, for labelled and unlabelled data (null: the task needs a target).
const DEFAULTS: Readonly<Record<TaskType, { labelled: readonly MetricName[]; unlabelled: readonly MetricName[] | null }>> = {
  classification: { labelled: ["accuracy"], unlabelled: null },
  regression: { labelled: ["mae", "mse"], unlabelled: null },
  forecasting: { labelled: ["mae", "rmse"], unlabelled: null },
  anomaly_detection: { labelled: ["average_precision"], unlabelled: ["threshold_calibration_error"] },
  embedding: { labelled: ["retrieval_precision_at_1"], unlabelled: ["mean_pairwise_cosine"] },
  transcription: { labelled: ["wer", "cer"], unlabelled: null },
};
export function defaultMetrics(task: TaskType, labelled: boolean): MetricName[] {
  const chosen = labelled ? DEFAULTS[task].labelled : DEFAULTS[task].unlabelled;
  if (!chosen) throw new ConflictError(`${task} requires a target`);
  return [...chosen];
}
// Phase 1 behaviour: a classification or regression spec without `evaluation.metrics` computes exactly these.
export const DEFAULT_METRICS = { classification: defaultMetrics("classification", true), regression: defaultMetrics("regression", true) };

/** Rejects metrics the task, the data or the estimator cannot produce, instead of silently omitting them later. */
export function assertMetricsSupported(model: ModelDefinition, metrics: readonly string[], labelled = true): void {
  for (const name of metrics) {
    if (!isMetricName(name)) throw new ConflictError(`Unknown metric: ${name}`);
    const metric = METRICS[name];
    if (!metric.tasks.includes(model.taskType)) throw new ConflictError(`Metric ${name} does not apply to ${model.taskType}`);
    if (metric.labels && !labelled) throw new ConflictError(`Metric ${name} needs a target`);
    if (metric.needs === "probability" && model.capabilities.scores !== "probability") throw new ConflictError(`${model.id} cannot produce ${name}`);
    if (metric.needs === "scores" && model.capabilities.scores === "none") throw new ConflictError(`${model.id} cannot produce ${name}`);
  }
}

/** Experiments, AutoML and ranking order models by quality, which a diagnostic does not measure. */
export function assertRankingMetrics(metrics: readonly string[]): void {
  for (const name of metrics) if (isMetricName(name) && METRICS[name].diagnostic) throw new ConflictError(`Metric ${name} is a diagnostic and cannot rank models`);
}
