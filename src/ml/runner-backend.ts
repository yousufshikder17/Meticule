import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { BackendCapabilities, Evaluation, MlBackend, PreparedData, TrainedModel } from "./backend.js";
import { contractFor, ML_BACKEND_CONTRACT_V2, MlBackendFailure } from "./contract.js";
import { isTabularType, Metric, partitionSizes, validateSnapshot, type Row, type Snapshot } from "./domain.js";
import { assertMetricsSupported, defaultMetrics } from "./metrics.js";
import { getModel, validateHyperparameters } from "./model-catalog.js";
import { validatePipeline } from "./preprocessing.js";
import { callRunner } from "./runner-process.js";
import { effectivePolicy } from "./training-policy.js";
import { TASKS, predictionCount, validatePredictions } from "./tasks.js";

// Response shapes shared by every Python-runner backend. A runner's output is parsed here, at the platform boundary.
const Environment = z.record(z.string(), z.string());
const Indices = z.array(z.int().nonnegative());
const Prepared = z.object({ indices: z.object({ train: Indices, validation: Indices, test: Indices }), folds: z.array(Indices).optional(), environment: Environment });
const Performance = z.record(z.string(), z.number().finite().nonnegative());
export const RunnerResponse = { Environment, Prepared, Performance };

export interface RunnerBackendOptions {
  id: string;
  capabilities: BackendCapabilities;
  /** Interpreter with the backend's pinned requirements installed. Separate from every other backend's interpreter. */
  python: string;
  script: string;
  timeoutMs: number;
}

/**
 * The platform side of a backend whose computation runs in a Python runner. It owns everything that is not numerical:
 * validation against the declared capabilities and the catalog, the runner call, and parsing every response. Concrete
 * backends supply capabilities and a script. Framework code never reaches this class.
 */
export abstract class RunnerBackend implements MlBackend {
  readonly id: string;
  readonly capabilities: BackendCapabilities;
  protected readonly python: string;
  protected readonly script: string;
  protected readonly timeoutMs: number;

  protected constructor(options: RunnerBackendOptions) {
    if (options.capabilities.artifactFormats.length !== 1) throw new Error(`${options.id} must declare exactly one artifact format`);
    this.id = options.id;
    this.capabilities = options.capabilities;
    this.python = options.python;
    this.script = options.script;
    this.timeoutMs = options.timeoutMs;
  }

  /**
   * Refuses, before any runner starts, anything this backend or its models cannot honour. Every rule reads the declared
   * capabilities or the catalog; none names a backend.
   */
  validate(snapshot: Snapshot): void {
    validateSnapshot(snapshot);
    if (snapshot.backend !== this.id) throw new ConflictError("Backend mismatch");
    const model = getModel(this.id, snapshot.algorithm);
    validateHyperparameters(model, snapshot.hyperparameters);
    // A v1 snapshot always declares a target, so `labelled` only varies under v2.
    const labelled = snapshot.dataset.schema.target !== undefined;
    // The Phase 4 rules (task registry, modalities, chronological split) apply only to snapshots that need contract v2, so a
    // stored v1 model (e.g. a boolean-target ridge) keeps validating, serving and resuming exactly as in Phase 3 (A4).
    // A model's declared inputs hold under every contract: a text, image or audio model never runs on a tabular column, even
    // when the snapshot alone would select v1. For Phase 3 models (tabular inputs, 1..199 features) this is the schema's own
    // rule, so v1 behaviour is unchanged. A missing feature is left to validatePipeline, which reports it by name.
    const schema = snapshot.dataset.schema, featureNames = snapshot.pipeline.definition.features;
    const featureColumns = featureNames.map(name => schema.columns.find(c => c.name === name)).filter(c => c !== undefined);
    if (featureNames.length < model.inputs.min || featureNames.length > model.inputs.max) throw new ConflictError("Feature count is outside the model input bounds");
    if (featureColumns.some(c => !model.inputs.types.includes(c.type) || !this.capabilities.inputs.includes(c.type))) throw new ConflictError("Model or backend does not support the feature column types");
    if (contractFor(snapshot, model.taskType) === ML_BACKEND_CONTRACT_V2) {
      const task = TASKS[model.taskType], features = featureColumns;
      if (!this.capabilities.tasks.includes(model.taskType)) throw new ConflictError("Backend does not support this task");
      const target = schema.columns.find(c => c.name === schema.target);
      if (!target && task.target === "required") throw new ConflictError(`${model.taskType} requires a target`);
      if (target && !task.targetTypes.includes(target.type)) throw new ConflictError("Target type is not supported for this task");
      if (model.taskType === "transcription" && target?.nullable) throw new ConflictError("Transcription needs a non-nullable reference transcript");
      if (!task.splits.includes(snapshot.split.strategy)) throw new ConflictError("Split strategy is not supported for this task");
      if (snapshot.evaluation.cv && snapshot.split.strategy === "chronological") throw new ConflictError("Cross-validation is not supported with a chronological split");
      if (snapshot.evaluation.cv && !task.crossValidation) throw new ConflictError("Cross-validation is not supported for this task");
      // A pretrained model fits nothing, so there is nothing for cross-validation to refit per fold.
      if (snapshot.evaluation.cv && model.learning === "pretrained") throw new ConflictError("Cross-validation is not supported for pretrained models");
      if (!labelled && snapshot.split.strategy === "random" && snapshot.split.stratify) throw new ConflictError("Stratification requires a target");
      const sizes = Object.values(partitionSizes(snapshot.split, snapshot.dataset.rowCount));
      if (snapshot.split.strategy === "chronological") {
        const timeColumn = snapshot.split.timeColumn, time = schema.columns.find(c => c.name === timeColumn);
        if (!time || time.type !== "timestamp" || time.nullable) throw new ConflictError("Chronological split requires a non-nullable timestamp column");
        if (sizes.some(n => n < 1)) throw new ConflictError("Chronological partitions must each contain at least one row");
        if (model.taskType === "forecasting" && (features.length !== 1 || features[0]!.name !== time.name || target?.nullable)) throw new ConflictError("Forecasting requires only the time column as a feature and a non-nullable target");
      }
      if (model.taskType === "embedding" && sizes.some(n => n > 5000)) throw new ConflictError("Embedding partitions cannot exceed 5000 rows");
      if (snapshot.pipeline.definition.steps.length && !model.inputs.types.every(isTabularType)) throw new ConflictError("Preprocessing steps require a tabular model");
      predictionCount(model, snapshot, 0); // refuses a forecasting model without a usable horizon (E3)
    }
    validatePipeline(snapshot);
    for (const step of snapshot.pipeline.definition.steps) if (!this.capabilities.preprocessing.includes(step.operation)) throw new ConflictError(`${this.id} backend does not support preprocessing ${step.operation}`);
    // The training policy (checkpoints, early stopping) only means something for an iterative backend.
    if (snapshot.training && !this.capabilities.iterative) throw new ConflictError("Training policy requires an iterative model");
    // Early stopping and the per-chunk epoch rule are Meticule policy. They reject a model or split that cannot support them.
    const policy = effectivePolicy(snapshot);
    if (!this.capabilities.devices.includes(policy.device)) throw new ConflictError(`${this.id} does not support device ${policy.device}`);
    if (!this.capabilities.precisions.includes(policy.precision)) throw new ConflictError(`${this.id} does not support ${policy.precision} precision`);
    // A model without its own device list inherits the backend's; a model that narrows it must stay inside it.
    if (!(model.capabilities.devices ?? this.capabilities.devices).includes(policy.device)) throw new ConflictError(`${model.id} does not support device ${policy.device}`);
    if (policy.precision === "mixed" && !model.capabilities.mixedPrecision) throw new ConflictError(`${model.id} does not support mixed precision`);
    const { cv, metrics } = snapshot.evaluation;
    const stratified = (snapshot.split.strategy === "random" && snapshot.split.stratify) || cv?.strategy === "stratified_kfold";
    if (stratified && model.taskType === "regression") throw new ConflictError("Regression cannot stratify by target");
    assertMetricsSupported(model, metrics ?? defaultMetrics(model.taskType, labelled), labelled);
  }

  /**
   * Validates, then runs one bounded command in a fresh runner process. Every response goes through `parse`. The request
   * speaks the contract the snapshot needs; a v2 request also names the algorithm's catalog task, which the runner checks.
   */
  protected call(command: string, snapshot: Snapshot, rows: Row[], extra: object, signal: AbortSignal): Promise<unknown> {
    this.validate(snapshot);
    if (signal.aborted) throw new MlBackendFailure("cancelled", `${this.id} ${command} was cancelled`);
    const task = getModel(this.id, snapshot.algorithm).taskType, contract = contractFor(snapshot, task);
    return callRunner({ label: this.id, python: this.python, script: this.script, timeoutMs: this.timeoutMs, signal, contract,
      request: { ...this.requestExtras(), command, snapshot, rows, ...extra, ...(contract === ML_BACKEND_CONTRACT_V2 ? { task } : {}) } });
  }

  /** Backend-specific request fields (e.g. a model root). Protocol fields always win over them. */
  protected requestExtras(): Record<string, unknown> { return {}; }

  /** The platform boundary for runner output: a response that does not match its schema is an invalid_response failure. */
  protected parse<T extends z.ZodType>(schema: T, value: unknown): z.output<T> {
    const result = schema.safeParse(value);
    // The Zod issues are not included: they could echo runner output, which may carry dataset content.
    if (!result.success) throw new MlBackendFailure("invalid_response", `${this.id} runner response did not match the contract`);
    return result.data;
  }

  /** The one artifact format this backend writes and reads. Declared once, in its capabilities. */
  protected get artifactFormat(): string { return this.capabilities.artifactFormats[0]!; }

  async prepare(snapshot: Snapshot, rows: Row[], signal: AbortSignal): Promise<PreparedData> {
    return this.parse(RunnerResponse.Prepared, await this.call("prepare", snapshot, rows, {}, signal));
  }
  async train(snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<TrainedModel> {
    const output = this.parse(z.object({ artifact: z.string().min(1), format: z.literal(this.artifactFormat), environment: RunnerResponse.Environment, resolvedHyperparameters: z.record(z.string(), z.unknown()), performance: RunnerResponse.Performance.optional() }),
      await this.call("train", snapshot, rows, { prepared, expectedEnvironment: prepared.environment }, signal));
    return { bytes: Buffer.from(output.artifact, "base64"), format: output.format, environment: output.environment, resolvedHyperparameters: output.resolvedHyperparameters, performance: output.performance };
  }
  async evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal) { return (await this.evaluateDetailed(snapshot, rows, prepared, model, signal)).metrics; }
  async evaluateDetailed(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<Evaluation> {
    return this.parse(z.object({ metrics: z.array(Metric.omit({ id: true })), performance: RunnerResponse.Performance.optional() }),
      await this.call("evaluate", snapshot, rows, { prepared, artifact: model.bytes.toString("base64"), expectedEnvironment: model.environment }, signal));
  }
  /** Predictions must match the task's output contract: element schema, count (per row or horizon), embedding dimension. */
  async predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]> {
    const model = getModel(this.id, snapshot.algorithm);
    const output = this.parse(z.object({ predictions: z.unknown() }), await this.call("predict", snapshot, rows, { artifact: artifact.toString("base64"), expectedEnvironment: environment }, signal));
    return validatePredictions(model.taskType, output.predictions, predictionCount(model, snapshot, rows.length), environment);
  }
}
