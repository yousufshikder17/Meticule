import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import { assertMetricsSupported, isMetricName, METRICS } from "./metrics.js";
import { getModel, type TaskType } from "./model-catalog.js";
import type { SearchSpaceType } from "./search.js";

const Name = z.string().trim().min(1).max(120);
// Hard ceilings: AutoML V1 explores a small, fixed neighbourhood and never generates unbounded work.
export const MAX_CANDIDATE_RUNS = 40, MAX_SEARCH_PER_MODEL = 10;
export const AutoMlInput = z.object({
  name: Name, experimentId: z.uuid().optional(), idempotencyKey: z.string().min(1).max(200).optional(),
  datasetVersionId: z.uuid(), target: Name, taskType: z.enum(["classification", "regression"]), metric: Name, secondaryMetrics: z.array(Name).max(5).default([]),
  backend: Name.default("sklearn"), seed: z.int().min(0).max(4_294_967_295), cvFolds: z.int().min(2).max(10).default(5),
  budget: z.object({ maxCandidateRuns: z.int().min(2).max(MAX_CANDIDATE_RUNS).default(12), maxSearchCandidatesPerModel: z.int().min(1).max(MAX_SEARCH_PER_MODEL).default(3) }).default({ maxCandidateRuns: 12, maxSearchCandidatesPerModel: 3 }),
  source: z.object({ revision: z.string().min(1).max(200), repository: z.string().max(500).optional(), dirty: z.boolean() }).nullable().default(null),
});
export type AutoMlInputType = z.infer<typeof AutoMlInput>;

// Cheap, interpretable families first, so small budgets still cover the useful baselines.
const FAMILIES: Record<TaskType, string[]> = {
  classification: ["logistic_regression", "decision_tree_classifier", "random_forest_classifier", "gradient_boosting_classifier", "k_neighbors_classifier", "svc"],
  regression: ["ridge", "linear_regression", "decision_tree_regressor", "random_forest_regressor", "gradient_boosting_regressor", "k_neighbors_regressor", "lasso", "elastic_net", "svr"],
};
const choice = (...values: (number | null)[]) => ({ type: "choice" as const, values });
const SEARCHABLE: Record<string, SearchSpaceType> = {
  random_forest_classifier: { n_estimators: choice(50, 100, 200), max_depth: choice(3, 6, 12, null), min_samples_leaf: choice(1, 2, 5) },
  random_forest_regressor: { n_estimators: choice(50, 100, 200), max_depth: choice(3, 6, 12, null), min_samples_leaf: choice(1, 2, 5) },
  gradient_boosting_classifier: { n_estimators: choice(50, 100, 200), learning_rate: choice(.03, .1, .3), max_depth: choice(2, 3, 5) },
  gradient_boosting_regressor: { n_estimators: choice(50, 100, 200), learning_rate: choice(.03, .1, .3), max_depth: choice(2, 3, 5) },
};
interface Column { name: string; type: "number" | "string" | "boolean" }
export interface AutoMlPlan {
  features: string[]; steps: { operation: string; parameters: Record<string, never> }[];
  benchmark: string[]; searches: { algorithm: string; space: SearchSpaceType; maxCandidates: number }[];
  skipped: { algorithm: string; reason: string }[]; totalCandidates: number;
}

/** Pure and deterministic: same schema, row count, input and budget always yield the same plan. */
export function planAutoMl(input: AutoMlInputType, schema: { columns: Column[]; target: string }, rowCount: number): AutoMlPlan {
  if (input.target !== schema.target) throw new ConflictError(`target must be the dataset's declared target column (${schema.target})`);
  const targetType = schema.columns.find(c => c.name === schema.target)!.type;
  if (input.taskType === "regression" && targetType !== "number") throw new ConflictError("Regression requires a numeric target");
  if (rowCount < Math.max(30, input.cvFolds * 10)) throw new ConflictError(`AutoML needs at least ${Math.max(30, input.cvFolds * 10)} rows for ${input.cvFolds}-fold CV; the dataset has ${rowCount}`);
  const features = schema.columns.filter(c => c.name !== schema.target);
  if (!features.length) throw new ConflictError("Dataset has no feature columns");
  const metrics = [input.metric, ...input.secondaryMetrics];
  if (new Set(metrics).size !== metrics.length) throw new ConflictError("Primary and secondary metrics must be distinct");
  for (const name of metrics) if (!isMetricName(name) || METRICS[name].task !== input.taskType) throw new ConflictError(`Metric ${name} is unknown or does not apply to ${input.taskType}`);
  const skipped: AutoMlPlan["skipped"] = []; const usable: string[] = [];
  for (const algorithm of FAMILIES[input.taskType]) {
    try { assertMetricsSupported(getModel(input.backend, algorithm), metrics); usable.push(algorithm); }
    catch (error) { if (!(error instanceof ConflictError)) throw error; skipped.push({ algorithm, reason: error.message }); }
  }
  if (usable.length < 2) throw new ConflictError(`Metric ${input.metric} is supported by fewer than two model families for ${input.taskType}`);
  const runs = input.budget.maxCandidateRuns;
  const benchmark = usable.slice(0, Math.min(usable.length, Math.max(2, Math.ceil(runs / 2))));
  const searchable = benchmark.filter(a => a in SEARCHABLE); const remaining = runs - benchmark.length;
  const each = searchable.length ? Math.min(input.budget.maxSearchCandidatesPerModel, Math.floor(remaining / searchable.length)) : 0;
  const searches = each > 0 ? searchable.map(algorithm => ({ algorithm, space: SEARCHABLE[algorithm]!, maxCandidates: each })) : [];
  const numeric = features.some(c => c.type === "number"), categorical = features.some(c => c.type !== "number");
  const steps = [...(numeric ? ["impute_median", "standard_scale"] : []), ...(categorical ? ["impute_most_frequent", "one_hot_encode"] : [])].map(operation => ({ operation, parameters: {} }));
  return { features: features.map(c => c.name), steps, benchmark, searches, skipped, totalCandidates: benchmark.length + searches.reduce((n, s) => n + s.maxCandidates, 0) };
}
