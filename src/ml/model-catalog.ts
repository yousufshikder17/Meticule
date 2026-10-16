import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import { TABULAR_COLUMN_TYPES, type ColumnType } from "./domain.js";
import type { TaskType } from "./tasks.js";

type Scalar = string | number | boolean | null;
// Defaults may be scalars or bounded integer lists (neural-network layer widths).
type DefaultValue = Scalar | number[];
// One declaration drives validation, discovery and search-space checks; unknown parameters are never forwarded.
export type ParameterSpec =
  | { kind: "int"; min: number; max: number; nullable?: boolean; default: number | null }
  | { kind: "float"; min: number; max: number; openMin?: boolean; nullable?: boolean; default: number | null }
  | { kind: "choice"; values: readonly Scalar[]; default: Scalar }
  | { kind: "bool"; default: boolean }
  // Bounded list of integers, e.g. hidden layer widths. Only neural definitions use it; search spaces reject it.
  | { kind: "intList"; minItems: number; maxItems: number; min: number; max: number; default: number[] }
  // A local model reference (pretrained weights provisioned under the model root): a bounded identifier, never a path.
  // Nullable with default null; search spaces and AutoML reject it.
  | { kind: "name"; default: null };
const MODEL_NAME = /^[a-z0-9][a-z0-9._-]{0,63}$/;
/** Column types a model reads as features, and how many feature columns it takes. */
export interface ModelInputs { readonly types: readonly ColumnType[]; readonly min: number; readonly max: number }
/** supervised fits with labels; unsupervised fits without; pretrained applies fixed weights and fits nothing (a pretrained classifier only binds its label set). */
export type Learning = "supervised" | "unsupervised" | "pretrained";
const TABULAR_INPUTS: ModelInputs = { types: TABULAR_COLUMN_TYPES, min: 1, max: 199 };
export interface ModelCapabilities {
  /** Which per-class score the estimator exposes; gates roc_auc (probability|decision, binary only for decision) and log_loss (probability). */
  scores: "probability" | "decision" | "none";
  scaleSensitive: boolean;
  /** Estimator consumes random_state, so the seed changes its fit. */
  seeded: boolean;
  /** Trained by repeated passes over the data (epochs), so the training configuration includes an epoch count. */
  iterative?: boolean | undefined;
  /** Devices the backend can execute on. The caller must request one explicitly; none is chosen silently. */
  devices?: readonly string[] | undefined;
  /** Whether the model can train with mixed precision (CUDA only). */
  mixedPrecision?: boolean | undefined;
}
export interface ModelDefinition {
  id: string;
  backend: string;
  taskType: TaskType;
  displayName: string;
  parameters: Record<string, ParameterSpec>;
  parameterSchema: z.ZodType<Record<string, unknown>>;
  defaults: Record<string, DefaultValue>;
  capabilities: ModelCapabilities;
  inputs: ModelInputs;
  learning: Learning;
  /** Cross-parameter rule; returns a message when the combination is impossible or silently ignored. */
  check?: ((params: Record<string, unknown>) => string | null) | undefined;
}

function schemaFor(parameters: Record<string, ParameterSpec>): z.ZodType<Record<string, unknown>> {
  const shape: Record<string, z.ZodType> = {};
  for (const [name, spec] of Object.entries(parameters)) {
    let type: z.ZodType;
    if (spec.kind === "int") { type = z.int().min(spec.min).max(spec.max); if (spec.nullable) type = type.nullable(); }
    else if (spec.kind === "float") { type = spec.openMin ? z.number().gt(spec.min).max(spec.max) : z.number().min(spec.min).max(spec.max); if (spec.nullable) type = type.nullable(); }
    else if (spec.kind === "bool") type = z.boolean();
    else if (spec.kind === "intList") type = z.array(z.int().min(spec.min).max(spec.max)).min(spec.minItems).max(spec.maxItems);
    else if (spec.kind === "name") type = z.string().regex(MODEL_NAME).nullable();
    else type = z.union(spec.values.map(v => z.literal(v)) as unknown as [z.ZodType, z.ZodType]);
    shape[name] = type.optional();
  }
  return z.strictObject(shape);
}
/**
 * Backend-neutral definition builder: the catalog is data, and a backend's definitions are registered, not branched on.
 * Without `inputs` a model reads 1-199 tabular columns; without `learning` it is supervised.
 */
export function defineModel(model: { backend: string; id: string; taskType: TaskType; displayName: string; parameters: Record<string, ParameterSpec>; capabilities: ModelCapabilities; inputs?: ModelInputs; learning?: Learning; check?: ModelDefinition["check"] }): ModelDefinition {
  return { ...model, inputs: model.inputs ?? TABULAR_INPUTS, learning: model.learning ?? "supervised", parameterSchema: schemaFor(model.parameters), defaults: Object.fromEntries(Object.entries(model.parameters).map(([k, v]) => [k, v.default])) };
}
function define(id: string, taskType: TaskType, displayName: string, parameters: Record<string, ParameterSpec>, capabilities: ModelCapabilities, check?: ModelDefinition["check"]): ModelDefinition {
  return defineModel({ backend: "sklearn", id, taskType, displayName, parameters, capabilities, check });
}

const depth = (max = 100): ParameterSpec => ({ kind: "int", min: 1, max, nullable: true, default: null });
const leaf: ParameterSpec = { kind: "int", min: 1, max: 1000, default: 1 };
const trees: ParameterSpec = { kind: "int", min: 1, max: 1000, default: 100 };
const C: ParameterSpec = { kind: "float", min: 0, max: 1e6, openMin: true, default: 1 };
const kernel: ParameterSpec = { kind: "choice", values: ["linear", "rbf", "poly", "sigmoid"], default: "rbf" };
const svm = { kernel, gamma: { kind: "choice", values: ["scale", "auto"], default: "scale" } as ParameterSpec, degree: { kind: "int", min: 1, max: 10, default: 3 } as ParameterSpec };
const classWeight: ParameterSpec = { kind: "choice", values: [null, "balanced"], default: null };
const boosting: Record<string, ParameterSpec> = { n_estimators: trees, learning_rate: { kind: "float", min: 0, max: 10, openMin: true, default: .1 }, max_depth: { kind: "int", min: 1, max: 20, default: 3 }, subsample: { kind: "float", min: 0, max: 1, openMin: true, default: 1 }, min_samples_leaf: leaf };
const knn: Record<string, ParameterSpec> = { n_neighbors: { kind: "int", min: 1, max: 200, default: 5 }, weights: { kind: "choice", values: ["uniform", "distance"], default: "uniform" }, p: { kind: "choice", values: [1, 2], default: 2 } };
const alpha = (min = 0, openMin = true): ParameterSpec => ({ kind: "float", min, max: 1e6, openMin, default: 1 });
const iterations = (d: number): ParameterSpec => ({ kind: "int", min: 1, max: 10_000, default: d });
const kernelDegree = (p: Record<string, unknown>) => p.degree !== undefined && p.degree !== 3 && (p.kernel ?? "rbf") !== "poly" ? "degree only applies to the poly kernel" : null;
const forest: Record<string, ParameterSpec> = { n_estimators: trees, max_depth: depth(), min_samples_leaf: leaf };
const probability = { scores: "probability", scaleSensitive: false, seeded: false } as const;

export const SKLEARN_MODELS: readonly ModelDefinition[] = [
  define("logistic_regression", "classification", "Logistic regression", { C, max_iter: iterations(100), class_weight: classWeight }, { ...probability, scaleSensitive: true }),
  define("random_forest_classifier", "classification", "Random forest classifier", { ...forest, max_features: { kind: "choice", values: ["sqrt", "log2", null], default: "sqrt" }, class_weight: classWeight }, { ...probability, seeded: true }),
  define("gradient_boosting_classifier", "classification", "Gradient boosting classifier", boosting, { ...probability, seeded: true }),
  define("svc", "classification", "Support vector classifier", { C, ...svm, class_weight: classWeight }, { scores: "decision", scaleSensitive: true, seeded: false }, kernelDegree),
  define("k_neighbors_classifier", "classification", "k-nearest neighbors classifier", knn, { ...probability, scaleSensitive: true }),
  define("decision_tree_classifier", "classification", "Decision tree classifier", { max_depth: depth(), min_samples_leaf: leaf, min_samples_split: { kind: "int", min: 2, max: 1000, default: 2 }, criterion: { kind: "choice", values: ["gini", "entropy"], default: "gini" } }, { ...probability, seeded: true }),
  define("linear_regression", "regression", "Ordinary least squares", { fit_intercept: { kind: "bool", default: true } }, { scores: "none", scaleSensitive: false, seeded: false }),
  define("ridge", "regression", "Ridge regression", { alpha: alpha(0, false) }, { scores: "none", scaleSensitive: true, seeded: false }),
  define("lasso", "regression", "Lasso regression", { alpha: alpha(), max_iter: iterations(1000) }, { scores: "none", scaleSensitive: true, seeded: false }),
  define("elastic_net", "regression", "Elastic net regression", { alpha: alpha(), l1_ratio: { kind: "float", min: 0, max: 1, default: .5 }, max_iter: iterations(1000) }, { scores: "none", scaleSensitive: true, seeded: false }),
  define("random_forest_regressor", "regression", "Random forest regressor", forest, { scores: "none", scaleSensitive: false, seeded: true }),
  define("gradient_boosting_regressor", "regression", "Gradient boosting regressor", boosting, { scores: "none", scaleSensitive: false, seeded: true }),
  define("svr", "regression", "Support vector regressor", { C, ...svm, epsilon: { kind: "float", min: 0, max: 1e6, default: .1 } }, { scores: "none", scaleSensitive: true, seeded: false }, kernelDegree),
  define("k_neighbors_regressor", "regression", "k-nearest neighbors regressor", knn, { scores: "none", scaleSensitive: true, seeded: false }),
  define("decision_tree_regressor", "regression", "Decision tree regressor", { max_depth: depth(), min_samples_leaf: leaf, min_samples_split: { kind: "int", min: 2, max: 1000, default: 2 }, criterion: { kind: "choice", values: ["squared_error", "absolute_error"], default: "squared_error" } }, { scores: "none", scaleSensitive: false, seeded: true }),
  // Anomaly detection (Phase 4): unsupervised, scored by -score_samples (higher = more anomalous). contamination is the
  // expected anomaly fraction that sets the threshold; it is numeric here (scikit-learn's own default is "auto").
  ...anomalyModels(),
];

function anomalyModels(): ModelDefinition[] {
  const contamination: ParameterSpec = { kind: "float", min: 0, max: 0.5, openMin: true, default: 0.1 };
  const scored = (seeded: boolean, scaleSensitive: boolean) => ({ scores: "decision", scaleSensitive, seeded } as const);
  const anomaly = (id: string, displayName: string, parameters: Record<string, ParameterSpec>, capabilities: ModelCapabilities) =>
    defineModel({ backend: "sklearn", id, taskType: "anomaly_detection", displayName, learning: "unsupervised", parameters, capabilities });
  return [
    anomaly("isolation_forest", "Isolation forest", { n_estimators: trees, max_samples: { kind: "int", min: 1, max: 100_000, nullable: true, default: null }, contamination }, scored(true, false)),
    anomaly("local_outlier_factor", "Local outlier factor (novelty)", { n_neighbors: { kind: "int", min: 1, max: 200, default: 20 }, contamination }, scored(false, true)),
    anomaly("robust_covariance", "Robust covariance (elliptic envelope)", { support_fraction: { kind: "float", min: 0, max: 1, openMin: true, nullable: true, default: null }, contamination }, scored(true, true)),
  ];
}

// Backends register their definitions here; an optional XGBoost/LightGBM package would add its own array without touching callers.
const catalog = new Map<string, ModelDefinition>();
export function registerModels(models: readonly ModelDefinition[]): void {
  for (const m of models) { const key = `${m.backend}:${m.id}`; if (catalog.has(key)) throw new Error(`Duplicate model definition ${key}`); catalog.set(key, m); }
}
registerModels(SKLEARN_MODELS);
export const listModels = (backend?: string): ModelDefinition[] => [...catalog.values()].filter(m => !backend || m.backend === backend);
export function getModel(backend: string, id: string): ModelDefinition {
  const model = catalog.get(`${backend}:${id}`); if (!model) throw new ConflictError(`Unsupported ${backend} algorithm`);
  return model;
}
/** Allowlist validation plus cross-parameter rules; returns the parsed parameters. */
export function validateHyperparameters(model: ModelDefinition, raw: unknown): Record<string, unknown> {
  const params = model.parameterSchema.parse(raw); const problem = model.check?.(params);
  if (problem) throw new ConflictError(`${model.id}: ${problem}`);
  return params;
}
export const describeModel = ({ id, backend, taskType, displayName, parameters, defaults, capabilities, inputs, learning }: ModelDefinition) => ({ id, backend, taskType, displayName, parameters, defaults, capabilities, inputs, learning });
