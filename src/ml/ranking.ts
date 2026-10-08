import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { MetricValue } from "./domain.js";
import { METRICS, isMetricName } from "./metrics.js";

// "Best" always means best under this persisted objective, not a universal claim about a model.
export const RankingConfig = z.object({
  partition: z.enum(["cv", "validation"]),
  primaryMetric: z.string().min(1).max(40),
  secondaryMetrics: z.array(z.string().min(1).max(40)).max(9).default([]),
  tolerance: z.number().min(0).max(1e-3).default(0),
});
export type RankingConfigType = z.infer<typeof RankingConfig>;
/** Applied in order after the primary metric: lower fold variance, secondary metrics, lower fit time, then run ID. */
export const TIE_BREAKERS = ["primary_std", "secondary_metrics", "fit_seconds", "training_run_id"] as const;

export interface RankableCandidate {
  jobId: string; trainingRunId: string; label: string; algorithm: string; hyperparameters: Record<string, unknown>;
  metrics: MetricValue[]; performance: Record<string, number>; artifactBytes: number | null;
}
export interface RankedEntry {
  rank: number; jobId: string; trainingRunId: string; label: string; algorithm: string; hyperparameters: Record<string, unknown>;
  primary: { name: string; value: number; std: number | null; direction: "higher" | "lower" };
  secondary: { name: string; value: number; std: number | null; direction: "higher" | "lower" }[];
  fitSeconds: number | null; predictSeconds: number | null; artifactBytes: number | null;
}
export interface Ranking { config: RankingConfigType; tieBreakers: typeof TIE_BREAKERS; entries: RankedEntry[]; unranked: { jobId: string; label: string; reason: string }[] }

function pick(metrics: MetricValue[], name: string, partition: string) {
  const m = metrics.find(x => x.name === name && x.partition === partition && x.fold == null);
  if (!m) return null;
  if (isMetricName(name) && METRICS[name].direction !== m.direction) throw new ConflictError(`Metric ${name} direction disagrees with the metric registry`);
  return { name, value: m.value, std: m.std ?? null, direction: m.direction };
}

export function rankCandidates(candidates: readonly RankableCandidate[], rawConfig: z.input<typeof RankingConfig>): Ranking {
  const config = RankingConfig.parse(rawConfig);
  const scored: (RankedEntry & { runId: string })[] = []; const unranked: Ranking["unranked"] = [];
  for (const c of candidates) {
    const primary = pick(c.metrics, config.primaryMetric, config.partition);
    if (!primary) { unranked.push({ jobId: c.jobId, label: c.label, reason: `missing ${config.partition}/${config.primaryMetric}` }); continue; }
    const secondary = config.secondaryMetrics.flatMap(n => { const m = pick(c.metrics, n, config.partition); return m ? [m] : []; });
    scored.push({ rank: 0, runId: c.trainingRunId, jobId: c.jobId, trainingRunId: c.trainingRunId, label: c.label, algorithm: c.algorithm, hyperparameters: c.hyperparameters, primary, secondary,
      fitSeconds: c.performance.fitSeconds ?? null, predictSeconds: c.performance.predictSeconds ?? null, artifactBytes: c.artifactBytes });
  }
  const tie = (a: number, b: number) => Math.abs(a - b) <= config.tolerance * Math.max(1, Math.abs(a), Math.abs(b));
  // Negative when `a` ranks ahead of `b`. Missing optional evidence ranks behind present evidence.
  const order = (x: number | null, y: number | null, direction: "higher" | "lower" = "lower") => {
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1;
    if (tie(x, y)) return 0;
    return direction === "higher" ? y - x : x - y;
  };
  scored.sort((a, b) => order(a.primary.value, b.primary.value, a.primary.direction)
    || order(a.primary.std, b.primary.std)
    || config.secondaryMetrics.reduce((r, n) => r || order(a.secondary.find(m => m.name === n)?.value ?? null, b.secondary.find(m => m.name === n)?.value ?? null, METRICS[n as keyof typeof METRICS]?.direction ?? "higher"), 0)
    || order(a.fitSeconds, b.fitSeconds)
    || (a.runId < b.runId ? -1 : a.runId > b.runId ? 1 : 0));
  const entries = scored.map(({ runId: _runId, ...entry }, index) => ({ ...entry, rank: index + 1 }));
  return { config, tieBreakers: TIE_BREAKERS, entries, unranked };
}
