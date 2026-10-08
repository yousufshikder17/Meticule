import { describe, expect, it } from "vitest";
import { AutoMlInput, MAX_CANDIDATE_RUNS, planAutoMl } from "../../src/ml/automl.js";

const numeric = { columns: [{ name: "a", type: "number" as const }, { name: "b", type: "number" as const }, { name: "label", type: "string" as const }], target: "label" };
const mixed = { columns: [{ name: "a", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "flag", type: "boolean" as const }, { name: "y", type: "number" as const }], target: "y" };
const input = (extra: Record<string, unknown> = {}) => AutoMlInput.parse({ name: "auto", datasetVersionId: "11111111-1111-4111-8111-111111111111", target: "label", taskType: "classification", metric: "f1", seed: 1, ...extra });

describe("AutoML V1 planning", () => {
  it("is deterministic and covers the default classification neighbourhood", () => {
    const plan = planAutoMl(input(), numeric, 200);
    expect(planAutoMl(input(), numeric, 200)).toEqual(plan);
    expect(plan.benchmark).toEqual(["logistic_regression", "decision_tree_classifier", "random_forest_classifier", "gradient_boosting_classifier", "k_neighbors_classifier", "svc"]);
    expect(plan.searches.map(s => [s.algorithm, s.maxCandidates])).toEqual([["random_forest_classifier", 3], ["gradient_boosting_classifier", 3]]);
    expect(plan.totalCandidates).toBe(12); expect(plan.steps.map(s => s.operation)).toEqual(["impute_median", "standard_scale"]); expect(plan.features).toEqual(["a", "b"]);
  });
  it("never exceeds the candidate budget, for any budget the schema accepts", () => {
    for (let runs = 2; runs <= MAX_CANDIDATE_RUNS; runs++) for (const per of [1, 3, 10]) for (const task of ["classification", "regression"] as const) {
      const plan = planAutoMl(input({ budget: { maxCandidateRuns: runs, maxSearchCandidatesPerModel: per }, taskType: task, metric: task === "regression" ? "rmse" : "f1", target: task === "regression" ? "y" : "label" }), task === "regression" ? mixed : numeric, 500);
      expect(plan.totalCandidates, `${task} ${runs}/${per}`).toBeLessThanOrEqual(runs); expect(plan.benchmark.length).toBeGreaterThanOrEqual(2);
      for (const s of plan.searches) expect(s.maxCandidates).toBeLessThanOrEqual(per);
    }
    expect(() => input({ budget: { maxCandidateRuns: MAX_CANDIDATE_RUNS + 1 } })).toThrow(); expect(() => input({ budget: { maxSearchCandidatesPerModel: 11 } })).toThrow(); expect(() => input({ budget: { maxCandidateRuns: 1 } })).toThrow();
  });
  it("derives preprocessing from column types and excludes models that cannot produce the metric", () => {
    const regression = planAutoMl(input({ taskType: "regression", target: "y", metric: "r2" }), mixed, 100);
    expect(regression.steps.map(s => s.operation)).toEqual(["impute_median", "standard_scale", "impute_most_frequent", "one_hot_encode"]);
    expect(planAutoMl(input({ metric: "log_loss" }), numeric, 100).skipped.map(s => s.algorithm)).toEqual(["svc"]);
    expect(planAutoMl(input({ metric: "roc_auc" }), numeric, 100).benchmark).toContain("svc");
    const categoricalOnly = planAutoMl(input({ target: "y", taskType: "regression", metric: "mae" }), { columns: [{ name: "kind", type: "string" }, { name: "y", type: "number" }], target: "y" }, 100);
    expect(categoricalOnly.steps.map(s => s.operation)).toEqual(["impute_most_frequent", "one_hot_encode"]);
  });
  it("rejects invalid tasks, targets, metrics and tiny datasets", () => {
    expect(() => planAutoMl(input({ target: "a" }), numeric, 100)).toThrow(/declared target/);
    expect(() => planAutoMl(input({ taskType: "regression", target: "label", metric: "rmse" }), numeric, 100)).toThrow(/numeric target/);
    expect(() => planAutoMl(input({ metric: "rmse" }), numeric, 100)).toThrow(/does not apply/);
    expect(() => planAutoMl(input({ metric: "magic" }), numeric, 100)).toThrow(/unknown/);
    expect(() => planAutoMl(input({ secondaryMetrics: ["f1"] }), numeric, 100)).toThrow(/distinct/);
    expect(() => planAutoMl(input({ cvFolds: 10 }), numeric, 99)).toThrow(/at least 100 rows/);
    expect(() => planAutoMl(input(), { columns: [{ name: "label", type: "string" }, { name: "b", type: "number" }], target: "label" }, 10)).toThrow(/at least/);
  });
});
