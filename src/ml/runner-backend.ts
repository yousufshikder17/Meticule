import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { BackendCapabilities, Evaluation, MlBackend, PreparedData, TrainedModel } from "./backend.js";
import { MlBackendFailure } from "./contract.js";
import { Metric, validateSnapshot, type Row, type Snapshot } from "./domain.js";
import { assertMetricsSupported, DEFAULT_METRICS } from "./metrics.js";
import { getModel, validateHyperparameters } from "./model-catalog.js";
import { validatePipeline } from "./preprocessing.js";
import { callRunner } from "./runner-process.js";
import { effectivePolicy } from "./training-policy.js";

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
    const stratified = snapshot.split.stratify || cv?.strategy === "stratified_kfold";
    if (stratified && model.taskType === "regression") throw new ConflictError("Regression cannot stratify by target");
    assertMetricsSupported(model, metrics ?? DEFAULT_METRICS[model.taskType]);
  }

  /** Validates, then runs one bounded command in a fresh runner process. Every response goes through `parse`. */
  protected call(command: string, snapshot: Snapshot, rows: Row[], extra: object, signal: AbortSignal): Promise<unknown> {
    this.validate(snapshot);
    if (signal.aborted) throw new MlBackendFailure("cancelled", `${this.id} ${command} was cancelled`);
    return callRunner({ label: this.id, python: this.python, script: this.script, timeoutMs: this.timeoutMs, signal, request: { command, snapshot, rows, ...extra } });
  }

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
  async predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]> {
    return this.parse(z.object({ predictions: z.array(z.json()) }), await this.call("predict", snapshot, rows, { artifact: artifact.toString("base64"), expectedEnvironment: environment }, signal)).predictions;
  }
}
