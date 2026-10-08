import type { Evaluation, MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import { METRICS, type MetricName } from "../../src/ml/metrics.js";
import type { MetricValue, Row, Snapshot } from "../../src/ml/domain.js";

/** Deterministic stand-in: scores come from a table so ranking, failure and cancellation can be asserted exactly. */
export class FakeBackend implements MlBackend {
  readonly id = "sklearn";
  scores: Record<string, number> = { logistic_regression: .8, random_forest_classifier: .9, gradient_boosting_classifier: .85, svc: .7, k_neighbors_classifier: .6, decision_tree_classifier: .5 };
  fail = new Set<string>(); failWhen: ((s: Snapshot) => boolean) | null = null; delayMs = 0; hang: string | null = null; started: (() => void) | null = null; skewSplit = new Set<string>();
  validate(): void {}
  async prepare(s: Snapshot, rows: Row[]): Promise<PreparedData> {
    const all = rows.map((_, i) => i); const train = all.slice(0, 36), validation = all.slice(36, 48), test = all.slice(48);
    if (this.skewSplit.has(s.algorithm)) [validation[0], test[0]] = [test[0]!, validation[0]!];
    const k = s.evaluation.cv?.folds;
    return { indices: { train, validation, test }, folds: k ? Array.from({ length: k }, (_, f) => train.filter((_, i) => i % k === f)) : undefined, environment: { protocol: "fake" } };
  }
  async train(s: Snapshot, _r: Row[], _p: PreparedData, signal: AbortSignal): Promise<TrainedModel> {
    if (this.hang === s.algorithm) { this.started?.(); await new Promise<void>((_, reject) => signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true })); }
    if (this.delayMs) await new Promise(r => setTimeout(r, this.delayMs));
    if (this.fail.has(s.algorithm) || this.failWhen?.(s)) throw new Error("boom");
    return { bytes: Buffer.from(`model:${s.algorithm}`), format: "fake-v1", environment: { protocol: "fake" }, resolvedHyperparameters: { ...s.hyperparameters }, performance: { fitSeconds: .5 } };
  }
  async evaluate(s: Snapshot, r: Row[], p: PreparedData, m: TrainedModel, signal: AbortSignal) { return (await this.evaluateDetailed(s, r, p, m, signal)).metrics; }
  async evaluateDetailed(s: Snapshot, _r: Row[], p: PreparedData, _m: TrainedModel, _signal: AbortSignal): Promise<Evaluation> {
    const names = (s.evaluation.metrics ?? ["accuracy"]) as MetricName[]; const base = (this.scores[s.algorithm] ?? .5) + Number(s.hyperparameters.max_depth ?? 0) / 100 + Number(s.hyperparameters.n_estimators ?? 0) / 100_000; const metrics: MetricValue[] = [];
    const value = (n: MetricName, shift = 0) => METRICS[n].direction === "higher" ? base - shift : 1 - base + shift;
    for (const n of names) for (const partition of ["train", "validation", "test"] as const) metrics.push({ name: n, partition, value: value(n, partition === "train" ? -.05 : 0), direction: METRICS[n].direction });
    if (s.evaluation.cv) for (const n of names) {
      const folds = p.folds!.map((_, f) => value(n, (f - 1) * .01));
      folds.forEach((v, fold) => metrics.push({ name: n, partition: "cv_fold", value: v, direction: METRICS[n].direction, fold }));
      metrics.push({ name: n, partition: "cv", value: value(n), direction: METRICS[n].direction, std: .01 });
    }
    return { metrics, performance: { predictSeconds: .1, ...(s.evaluation.cv ? { cvFitSeconds: 1 } : {}) } };
  }
  async predict(_s: Snapshot, _a: Buffer, rows: Row[]): Promise<unknown[]> { return rows.map(() => "even"); }
}

