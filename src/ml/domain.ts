import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";

const Id = z.uuid();
const Name = z.string().trim().min(1).max(120);
export const JsonObject = z.record(z.string(), z.json());
const Identity = z.object({ id: Id, tenantId: Id, createdAt: z.iso.datetime() });
export const DatasetSchema = z.object({
  id: Id,
  columns: z.array(z.object({ name: Name, type: z.enum(["number", "string", "boolean"]), nullable: z.boolean().default(false) })).min(2).max(200),
  target: Name,
}).superRefine((s, c) => {
  if (new Set(s.columns.map(x => x.name)).size !== s.columns.length) c.addIssue({ code: "custom", message: "Duplicate columns" });
  if (!s.columns.some(x => x.name === s.target)) c.addIssue({ code: "custom", message: "Target must be a column" });
});
export const ArtifactReferenceSchema = z.object({ key: z.string().regex(/^[a-f0-9]{64}$/), sha256: z.string().regex(/^[a-f0-9]{64}$/), size: z.int().nonnegative(), mediaType: Name });
export const Dataset = Identity.extend({ name: Name });
export const DatasetVersion = Identity.extend({ datasetId: Id, version: z.int().positive(), schema: DatasetSchema, content: ArtifactReferenceSchema, rowCount: z.int().positive(), validation: JsonObject });
export const PipelineDefinition = z.object({
  features: z.array(Name).min(1).max(199).refine(x => new Set(x).size === x.length, "Duplicate features"),
  steps: z.array(z.object({ operation: Name, parameters: JsonObject.default({}) })).max(50),
});
export const FeaturePipeline = Identity.extend({ logicalId: Id, version: z.int().positive(), name: Name, definition: PipelineDefinition });
export const SplitDefinition = z.object({ id: Id, strategy: z.literal("random"), train: z.number().gt(0).lt(1), validation: z.number().gt(0).lt(1), test: z.number().gt(0).lt(1), stratify: z.boolean().default(false) })
  .refine(x => Math.abs(x.train + x.validation + x.test - 1) < 1e-9, "Split fractions must sum to one");
// Cross-validation runs over the training partition only; validation and test stay untouched for later scoring.
export const CvSpec = z.object({ strategy: z.enum(["kfold", "stratified_kfold"]), folds: z.int().min(2).max(20) });
export const EvaluationSpec = z.object({ metrics: z.array(Name).min(1).max(10).refine(x => new Set(x).size === x.length, "Duplicate metrics").optional(), cv: CvSpec.optional() });
export const TrainingSpec = z.object({ datasetVersionId: Id, featurePipelineId: Id, backend: Name, algorithm: Name, hyperparameters: JsonObject.default({}), seed: z.int().min(0).max(4_294_967_295), split: SplitDefinition, evaluation: EvaluationSpec.default({}), source: z.object({ revision: z.string().min(1).max(200), repository: z.string().max(500).optional(), dirty: z.boolean() }).nullable().default(null) });
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
export const Rows = z.array(z.record(z.string(), z.union([z.number().finite(), z.string().max(10_000), z.boolean(), z.null()]))).min(1).max(50_000);
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

export function validateRows(schema: DatasetSchemaType, raw: unknown, prediction = false): Row[] {
  const rows = Rows.parse(raw); const columns = schema.columns.filter(c => !prediction || c.name !== schema.target);
  for (const [index, row] of rows.entries()) {
    if (Object.keys(row).length !== columns.length) throw new ConflictError(`Row ${index}: unexpected or missing columns`);
    for (const c of columns) {
      const v = row[c.name];
      if (v === undefined || (v === null ? (!c.nullable || c.name === schema.target) : typeof v !== c.type)) throw new ConflictError(`Row ${index}: invalid ${c.name}`);
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
