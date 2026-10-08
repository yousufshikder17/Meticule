import { describe, expect, it } from "vitest";
import { rankCandidates, type RankableCandidate } from "../../src/ml/ranking.js";
import type { MetricValue } from "../../src/ml/domain.js";

const metric = (name: string, value: number, direction: "higher" | "lower", std: number | null = null, partition: MetricValue["partition"] = "cv"): MetricValue => ({ name, partition, value, direction, std });
const candidate = (id: string, metrics: MetricValue[], fitSeconds?: number): RankableCandidate => ({ jobId: `job-${id}`, trainingRunId: id, label: id, algorithm: "x", hyperparameters: {}, metrics, performance: fitSeconds === undefined ? {} : { fitSeconds }, artifactBytes: 1 });
const config = { partition: "cv" as const, primaryMetric: "f1", secondaryMetrics: ["accuracy", "log_loss"] };
const ids = (r: ReturnType<typeof rankCandidates>) => r.entries.map(e => e.trainingRunId);

describe("candidate ranking", () => {
  it("orders by the metric's declared direction, never by name", () => {
    const f1 = (id: string, v: number) => candidate(id, [metric("f1", v, "higher")]);
    expect(ids(rankCandidates([f1("a", .7), f1("b", .9), f1("c", .8)], config))).toEqual(["b", "c", "a"]);
    const loss = (id: string, v: number) => candidate(id, [metric("log_loss", v, "lower")]);
    expect(ids(rankCandidates([loss("a", .7), loss("b", .2), loss("c", .5)], { ...config, primaryMetric: "log_loss" }))).toEqual(["b", "c", "a"]);
  });
  it("breaks ties by fold variance, secondary metrics, fit time and finally run ID", () => {
    const base = [metric("f1", .8, "higher", .05), metric("accuracy", .8, "higher"), metric("log_loss", .5, "lower")];
    const tweak = (id: string, edits: Partial<Record<string, MetricValue>>, fit: number) => candidate(id, base.map(m => edits[m.name] ?? m), fit);
    const field = [
      tweak("e", {}, 2), tweak("d", {}, 1),
      tweak("c", { log_loss: metric("log_loss", .4, "lower") }, 9),
      tweak("b", { accuracy: metric("accuracy", .85, "higher") }, 9),
      tweak("a", { f1: metric("f1", .8, "higher", .01) }, 9),
      tweak("z", { f1: metric("f1", .81, "higher", .5) }, 9),
    ];
    expect(ids(rankCandidates(field, config))).toEqual(["z", "a", "b", "c", "d", "e"]);
    expect(ids(rankCandidates([tweak("y", {}, 1), tweak("x", {}, 1)], config))).toEqual(["x", "y"]);
  });
  it("is independent of input order and persists its configuration", () => {
    const field = ["a", "b", "c", "d", "e"].map((id, i) => candidate(id, [metric("f1", i % 2 ? .8 : .9, "higher", .1)], i));
    const expected = ids(rankCandidates(field, config));
    for (const shuffled of [[...field].reverse(), [field[2]!, field[4]!, field[0]!, field[3]!, field[1]!]]) expect(ids(rankCandidates(shuffled, config))).toEqual(expected);
    const ranking = rankCandidates(field, config);
    expect(ranking.config).toEqual({ ...config, tolerance: 0 }); expect(ranking.tieBreakers).toEqual(["primary_std", "secondary_metrics", "fit_seconds", "training_run_id"]);
    expect(ranking.entries.map(e => e.rank)).toEqual([1, 2, 3, 4, 5]);
  });
  it("leaves candidates without the primary metric unranked and rejects direction drift", () => {
    const r = rankCandidates([candidate("a", [metric("f1", .5, "higher")]), candidate("b", [metric("accuracy", .9, "higher")]), candidate("c", [metric("f1", .6, "higher", null, "validation")])], config);
    expect(ids(r)).toEqual(["a"]); expect(r.unranked.map(u => u.jobId)).toEqual(["job-b", "job-c"]);
    expect(() => rankCandidates([candidate("a", [metric("f1", .5, "lower")])], config)).toThrow(/direction/);
    expect(rankCandidates([], config).entries).toEqual([]);
  });
});
