import { ConflictError } from "../domain/errors.js";
import type { ModelDefinition, TaskType } from "./model-catalog.js";

// "higher" = maximize, "lower" = minimize. Ranking reads this, never the metric name.
export type MetricDirection = "higher" | "lower";
export interface MetricDefinition { task: TaskType; direction: MetricDirection; needs?: "probability" | "scores" }
export const METRICS = {
  accuracy: { task: "classification", direction: "higher" },
  precision: { task: "classification", direction: "higher" }, // weighted by class support, zero_division=0
  recall: { task: "classification", direction: "higher" },
  f1: { task: "classification", direction: "higher" },
  roc_auc: { task: "classification", direction: "higher", needs: "scores" }, // binary, or multiclass one-vs-rest with probabilities
  log_loss: { task: "classification", direction: "lower", needs: "probability" },
  mae: { task: "regression", direction: "lower" },
  mse: { task: "regression", direction: "lower" },
  rmse: { task: "regression", direction: "lower" },
  r2: { task: "regression", direction: "higher" },
} as const satisfies Record<string, MetricDefinition>;
export type MetricName = keyof typeof METRICS;
export const isMetricName = (name: string): name is MetricName => Object.hasOwn(METRICS, name);
// Phase 1 behaviour: a spec without `evaluation.metrics` computes exactly these.
export const DEFAULT_METRICS: Record<TaskType, MetricName[]> = { classification: ["accuracy"], regression: ["mae", "mse"] };

/** Rejects metrics the task or estimator cannot produce, instead of silently omitting them later. */
export function assertMetricsSupported(model: ModelDefinition, metrics: readonly string[]): void {
  for (const name of metrics) {
    if (!isMetricName(name)) throw new ConflictError(`Unknown metric: ${name}`);
    const metric: MetricDefinition = METRICS[name];
    if (metric.task !== model.taskType) throw new ConflictError(`Metric ${name} does not apply to ${model.taskType}`);
    if (metric.needs === "probability" && model.capabilities.scores !== "probability") throw new ConflictError(`${model.id} cannot produce ${name}`);
    if (metric.needs === "scores" && model.capabilities.scores === "none") throw new ConflictError(`${model.id} cannot produce ${name}`);
  }
}
