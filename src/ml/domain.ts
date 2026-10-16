import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { MEDIA_BUDGETS, MEDIA_LIMITS, MediaBudget, decodeBase64, imageHeader, timestampMs, wavLayout } from "./media.js";

const Id = z.uuid();
const Name = z.string().trim().min(1).max(120);
export const JsonObject = z.record(z.string(), z.json());
const Identity = z.object({ id: Id, tenantId: Id, createdAt: z.iso.datetime() });
// Phase 1-3 datasets use only the tabular types. text, timestamp, image and audio cells are strings with their own rules
// (validateRows, media.ts), and any of them makes a snapshot need contract v2.
export const TABULAR_COLUMN_TYPES = ["number", "string", "boolean"] as const;
export const COLUMN_TYPES = [...TABULAR_COLUMN_TYPES, "text", "timestamp", "image", "audio"] as const;
export type ColumnType = (typeof COLUMN_TYPES)[number];
export const isTabularType = (type: string): boolean => (TABULAR_COLUMN_TYPES as readonly string[]).includes(type);
const TARGET_TYPES: readonly ColumnType[] = ["number", "string", "boolean", "text"];
// The target is optional: embedding corpora and unsupervised anomaly detection have none, so one column is enough there.
export const DatasetSchema = z.object({
  id: Id,
  columns: z.array(z.object({ name: Name, type: z.enum(COLUMN_TYPES), nullable: z.boolean().default(false) })).min(1).max(200),
  target: Name.optional(),
}).superRefine((s, c) => {
  if (new Set(s.columns.map(x => x.name)).size !== s.columns.length) c.addIssue({ code: "custom", message: "Duplicate columns" });
  if (s.target === undefined) return;
  const target = s.columns.find(x => x.name === s.target);
  if (!target) c.addIssue({ code: "custom", message: "Target must be a column" });
  else if (!TARGET_TYPES.includes(target.type)) c.addIssue({ code: "custom", message: "Target must be a number, string, boolean or text column" });
  if (s.columns.length < 2) c.addIssue({ code: "custom", message: "A dataset with a target needs another column" });
});
/** Recorded as a dataset version's validator: unchanged for tabular datasets, versioned once any Phase 4 column type appears. */
export const datasetValidator = (schema: DatasetSchemaType): string => schema.columns.every(c => isTabularType(c.type)) ? "meticule-tabular-v1" : "meticule-dataset-v2";
export const ArtifactReferenceSchema = z.object({ key: z.string().regex(/^[a-f0-9]{64}$/), sha256: z.string().regex(/^[a-f0-9]{64}$/), size: z.int().nonnegative(), mediaType: Name });
export const Dataset = Identity.extend({ name: Name });
export const DatasetVersion = Identity.extend({ datasetId: Id, version: z.int().positive(), schema: DatasetSchema, content: ArtifactReferenceSchema, rowCount: z.int().positive(), validation: JsonObject });
export const PipelineDefinition = z.object({
  features: z.array(Name).min(1).max(199).refine(x => new Set(x).size === x.length, "Duplicate features"),
  steps: z.array(z.object({ operation: Name, parameters: JsonObject.default({}) })).max(50),
});
export const FeaturePipeline = Identity.extend({ logicalId: Id, version: z.int().positive(), name: Name, definition: PipelineDefinition });
const Fraction = z.number().gt(0).lt(1);
// chronological: rows ordered by the UTC instant of a timestamp column; train is the earliest block, then validation, then test.
// It has no stratify, and an unknown key is refused rather than dropped.
export const SplitDefinition = z.discriminatedUnion("strategy", [
  z.object({ id: Id, strategy: z.literal("random"), train: Fraction, validation: Fraction, test: Fraction, stratify: z.boolean().default(false) }),
  z.strictObject({ id: Id, strategy: z.literal("chronological"), timeColumn: Name, train: Fraction, validation: Fraction, test: Fraction }),
]).refine(x => Math.abs(x.train + x.validation + x.test - 1) < 1e-9, "Split fractions must sum to one");
export type SplitDefinitionType = z.infer<typeof SplitDefinition>;
export type SplitStrategy = SplitDefinitionType["strategy"];
/**
 * Partition sizes for n rows, computed as the runners split (ml/ml_runtime_common.py split_partitions). random:
 * scikit-learn's train_test_split twice, floor(n * train) and then floor(rest * validation / (validation + test)).
 * chronological: floor(n * train), floor(n * validation), and the rest is test. A size may be 0; callers refuse that.
 */
export function partitionSizes(split: SplitDefinitionType, n: number): { train: number; validation: number; test: number } {
  const train = Math.floor(n * split.train);
  const validation = split.strategy === "chronological" ? Math.floor(n * split.validation) : Math.floor((n - train) * (split.validation / (split.validation + split.test)));
  return { train, validation, test: n - train - validation };
}
// Cross-validation runs over the training partition only; validation and test stay untouched for later scoring.
export const CvSpec = z.object({ strategy: z.enum(["kfold", "stratified_kfold"]), folds: z.int().min(2).max(20) });
export const EvaluationSpec = z.object({ metrics: z.array(Name).min(1).max(10).refine(x => new Set(x).size === x.length, "Duplicate metrics").optional(), cv: CvSpec.optional() });
// Framework-neutral training policy (Phase 3B). Meticule owns it; backends only run the bounded chunks it asks for.
// Epoch history and checkpoints are bounded by the catalog epoch limit, so policy stays small and explicit.
export const EarlyStoppingPolicy = z.strictObject({ patience: z.int().min(1).max(50), minDelta: z.number().min(0).max(1e6).default(0) });
// The device and precision vocabulary is declared once here; backend capabilities and the catalog use the same values.
export const DEVICES = ["cpu", "cuda"] as const;
export const PRECISIONS = ["fp32", "mixed"] as const;
export const TrainingPolicy = z.strictObject({
  // Epochs per checkpoint. Each chunk is one bounded backend call; 1 gives per-epoch durability.
  checkpointEveryEpochs: z.int().min(1).max(50).default(5),
  // Monitors validation loss (lower is better). Restores the best-epoch model when enabled.
  earlyStopping: EarlyStoppingPolicy.nullable().default(null),
  // Explicit device. CUDA is never chosen silently: if it is requested and unavailable, the job fails.
  device: z.enum(DEVICES).default("cpu"),
  // Mixed precision is explicit and CUDA-only. CPU training stays fp32.
  precision: z.enum(PRECISIONS).default("fp32"),
}).refine(p => p.precision === "fp32" || p.device === "cuda", "Mixed precision requires device cuda").nullable().default(null);
export const TrainingSpec = z.object({ datasetVersionId: Id, featurePipelineId: Id, backend: Name, algorithm: Name, hyperparameters: JsonObject.default({}), seed: z.int().min(0).max(4_294_967_295), training: TrainingPolicy, split: SplitDefinition, evaluation: EvaluationSpec.default({}), source: z.object({ revision: z.string().min(1).max(200), repository: z.string().max(500).optional(), dirty: z.boolean() }).nullable().default(null) });
export const TrainingSnapshot = TrainingSpec.extend({ dataset: DatasetVersion, pipeline: FeaturePipeline });
export const TRAINING_STATES = ["QUEUED", "PREPARING", "TRAINING", "EVALUATING", "SAVING", "COMPLETED", "FAILED", "CANCELLED"] as const;
export type TrainingState = typeof TRAINING_STATES[number];
export const MODEL_STATES = ["REGISTERED", "READY", "RETIRED", "REJECTED"] as const;
export type ModelState = typeof MODEL_STATES[number];
const trainingEdges: Record<TrainingState, TrainingState[]> = { QUEUED: ["PREPARING", "CANCELLED", "FAILED"], PREPARING: ["TRAINING", "FAILED", "CANCELLED"], TRAINING: ["EVALUATING", "FAILED", "CANCELLED"], EVALUATING: ["SAVING", "FAILED", "CANCELLED"], SAVING: ["COMPLETED", "FAILED", "CANCELLED"], COMPLETED: [], FAILED: [], CANCELLED: [] };
const modelEdges: Record<ModelState, ModelState[]> = { REGISTERED: ["READY", "REJECTED"], READY: ["RETIRED"], RETIRED: [], REJECTED: [] };
export function assertTrainingTransition(from: TrainingState, to: TrainingState): void { if (!trainingEdges[from].includes(to)) throw new ConflictError(`Invalid training transition: ${from} -> ${to}`); }
export function assertModelTransition(from: ModelState, to: ModelState): void { if (!modelEdges[from].includes(to)) throw new ConflictError(`Invalid model transition: ${from} -> ${to}`); }
// partition "cv" is the across-fold mean (with std); "cv_fold" is one fold's validation score.
export const Metric = z.object({ id: Id, name: Name, partition: z.enum(["train", "validation", "test", "cv", "cv_fold"]), value: z.number().finite(), direction: z.enum(["higher", "lower"]), fold: z.int().nonnegative().nullish(), std: z.number().finite().nonnegative().nullish() });
export function assertMetricShape(m: MetricValue): void { if ((m.partition === "cv_fold") !== (m.fold != null)) throw new ConflictError("Only cv_fold metrics carry a fold"); }
export const Experiment = Identity.extend({ name: Name });
export const TrainingJob = Identity.extend({ runId: Id, experimentId: Id, snapshot: TrainingSnapshot, status: z.enum(TRAINING_STATES) });
export const TrainingRun = Identity.extend({ jobId: Id, attempt: z.int().positive(), snapshot: TrainingSnapshot, environment: JsonObject, splitIndices: z.record(z.string(), z.array(z.int().nonnegative())), startedAt: z.iso.datetime(), endedAt: z.iso.datetime().nullable(), status: z.enum(TRAINING_STATES), metrics: z.array(Metric), failure: JsonObject.nullable(), artifactIds: z.array(Id) });
export const ModelArtifact = Identity.extend({ trainingRunId: Id, kind: z.enum(["model", "checkpoint", "manifest"]), format: Name, content: ArtifactReferenceSchema });
export const ModelRegistryEntry = Identity.extend({ name: Name });
export const ModelVersion = Identity.extend({ registryEntryId: Id, trainingRunId: Id, artifactId: Id, version: z.int().positive(), status: z.enum(MODEL_STATES) });
export const InferenceEndpoint = Identity.extend({ name: Name, modelVersionId: Id, status: z.enum(["ACTIVE", "DISABLED"]) });
export const PredictionRecord = Identity.extend({ endpointId: Id, modelVersionId: Id, status: z.enum(["PENDING", "COMPLETED", "FAILED"]), inputHash: z.string().regex(/^[a-f0-9]{64}$/), count: z.int().positive(), latencyMs: z.int().nonnegative().nullable(), output: z.json().nullable(), failure: JsonObject.nullable() });
// Strings may be as long as a base64 media cell; validateRows then holds every non-media string to 10,000 characters.
export const Rows = z.array(z.record(z.string(), z.union([z.number().finite(), z.string().max(MEDIA_LIMITS.maxBase64Chars), z.boolean(), z.null()]))).min(1).max(50_000);
export type DatasetSchemaType = z.infer<typeof DatasetSchema>;
export type Snapshot = z.infer<typeof TrainingSnapshot>;
export type Row = z.infer<typeof Rows>[number];
export type ArtifactReference = z.infer<typeof ArtifactReferenceSchema>;
export type MetricValue = Omit<z.infer<typeof Metric>, "id">;
export type CvSpecType = z.infer<typeof CvSpec>;
/** Two attempts are a fair cohort only if everything except the estimator and its hyperparameters matches. */
export function cohortKey(raw: unknown, indices: unknown): string {
  const s = TrainingSnapshot.parse(raw);
  return canonicalJsonHash({ dataset: s.dataset.content.sha256, schema: s.dataset.schema, pipeline: s.pipeline.definition, split: { ...s.split, id: null }, seed: s.seed, cv: s.evaluation.cv ?? null, indices });
}

// The JSON type of each column type's cells.
const CELL_TYPES: Record<ColumnType, "number" | "string" | "boolean"> = { number: "number", string: "string", boolean: "boolean", text: "string", timestamp: "string", image: "string", audio: "string" };
export const MAX_TEXT_CHARS = 10_000;
const TextCell = z.string().max(MAX_TEXT_CHARS);

/**
 * Rows at the ingest or prediction boundary: exactly the schema's columns (prediction rows leave out the target unless
 * `withTarget`, as forecasting history needs), each cell of its column's type. string/text cells hold at most 10,000
 * characters, timestamps follow C4, images and audio are checked by their header walks and charged to the request's
 * media budget (C5; prediction has the smaller one).
 */
export function validateRows(schema: DatasetSchemaType, raw: unknown, prediction = false, withTarget = !prediction): Row[] {
  const rows = Rows.parse(raw); const columns = schema.columns.filter(c => withTarget || c.name !== schema.target);
  // Phase 1-3 refused any string over 10,000 characters in the Rows schema itself (a validation error); only image and
  // audio cells may be longer now, and every other long string is refused with that same error.
  const media = new Set(schema.columns.filter(c => c.type === "image" || c.type === "audio").map(c => c.name));
  for (const [index, row] of rows.entries()) for (const [key, v] of Object.entries(row)) {
    if (typeof v === "string" && v.length > MAX_TEXT_CHARS && !media.has(key)) throw new z.ZodError(TextCell.safeParse(v).error!.issues.map(issue => ({ ...issue, path: [index, key] })));
  }
  const budget = new MediaBudget(prediction ? MEDIA_BUDGETS.prediction : MEDIA_BUDGETS.ingest);
  for (const [index, row] of rows.entries()) {
    if (Object.keys(row).length !== columns.length) throw new ConflictError(`Row ${index}: unexpected or missing columns`);
    for (const c of columns) {
      const v = row[c.name];
      if (v === undefined || (v === null ? (!c.nullable || c.name === schema.target) : typeof v !== CELL_TYPES[c.type])) throw new ConflictError(`Row ${index}: invalid ${c.name}`);
      if (typeof v !== "string") continue;
      if (c.type === "timestamp" && timestampMs(v) === undefined) throw new ConflictError(`Row ${index}: invalid ${c.name}: use ISO-8601 with an explicit zone`);
      if (c.type !== "image" && c.type !== "audio") continue;
      try {
        const bytes = decodeBase64(v);
        if (c.type === "image") budget.image(imageHeader(bytes)); else budget.audio(wavLayout(bytes));
      } catch (error) {
        throw error instanceof ConflictError ? new ConflictError(`Row ${index}: invalid ${c.name}: ${error.message}`) : error;
      }
    }
  }
  return rows;
}

export function validateSnapshot(snapshot: Snapshot): void {
  if (snapshot.dataset.id !== snapshot.datasetVersionId || snapshot.pipeline.id !== snapshot.featurePipelineId) throw new ConflictError("Snapshot version mismatch");
  const columns = new Set(snapshot.dataset.schema.columns.map(c => c.name));
  if (snapshot.pipeline.definition.features.some(f => f === snapshot.dataset.schema.target || !columns.has(f))) throw new ConflictError("Features must exist and exclude the target");
}

export function compareMetrics(baseline: MetricValue[], candidate: MetricValue[]) {
  return baseline.map(b => {
    const c = candidate.find(c => c.name === b.name && c.partition === b.partition && (c.fold ?? null) === (b.fold ?? null));
    if (!c || c.direction !== b.direction) throw new ConflictError(`Missing compatible metric: ${b.partition}/${b.name}`);
    const delta = c.value - b.value;
    return { name: b.name, partition: b.partition, fold: b.fold ?? null, baseline: b.value, candidate: c.value, delta, regressed: b.direction === "higher" ? delta < 0 : delta > 0 };
  });
}
