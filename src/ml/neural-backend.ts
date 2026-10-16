import { z } from "zod";
import type { BackendCapabilities, ChunkResult, Device, PreparedData, TrainedModel } from "./backend.js";
import { TABULAR_COLUMN_TYPES, type Row, type Snapshot } from "./domain.js";
import { RunnerBackend, RunnerResponse } from "./runner-backend.js";

// Operations the NumPy preprocessing shared by the neural runners (ml/neural_common.py) interprets. Others fail explicitly.
export const NEURAL_OPERATIONS = ["impute_median", "impute_mean", "standard_scale", "min_max_scale", "impute_most_frequent", "one_hot_encode", "ordinal_encode"] as const;

export interface NeuralRunnerOptions {
  /** Registry id, e.g. "pytorch". Models are registered under this backend. */
  id: string;
  artifactFormat: string;
  checkpointFormat: string;
  /** Devices the runner really executes on. Declared by the backend, never inferred from the models it registers. */
  devices: readonly Device[];
  /** Whether the runner really trains with mixed precision (CUDA only). */
  mixedPrecision: boolean;
  /** Interpreter with the backend's pinned requirements installed. Separate from the sklearn interpreter. */
  python: string;
  script: string;
  timeoutMs: number;
}

/**
 * A neural backend whose training runs in a Python runner, as bounded chunks. The core sees only the MlBackend contract
 * and catalog capabilities; the framework never leaks out of its runner. Concrete backends are thin subclasses that fix
 * the id, formats, devices and script. Validation and the runner boundary are inherited from RunnerBackend.
 */
export class NeuralRunnerBackend extends RunnerBackend {
  private readonly checkpointFormat: string;

  constructor(options: NeuralRunnerOptions) {
    if (options.mixedPrecision && !options.devices.includes("cuda")) throw new Error(`${options.id} declares mixed precision without CUDA`);
    const capabilities: BackendCapabilities = {
      artifactFormats: [options.artifactFormat],
      checkpointFormats: [options.checkpointFormat],
      devices: options.devices,
      precisions: options.mixedPrecision ? ["fp32", "mixed"] : ["fp32"],
      preprocessing: NEURAL_OPERATIONS,
      iterative: true,
      tasks: ["classification", "regression"],
      inputs: TABULAR_COLUMN_TYPES,
    };
    super({ id: options.id, capabilities, python: options.python, script: options.script, timeoutMs: options.timeoutMs });
    this.checkpointFormat = options.checkpointFormat;
  }

  async trainChunk(snapshot: Snapshot, rows: Row[], prepared: PreparedData, state: Buffer | null, epochs: number, signal: AbortSignal): Promise<ChunkResult> {
    const output = this.parse(
      z.object({
        checkpoint: z.string().min(1), checkpointFormat: z.literal(this.checkpointFormat), checkpointSchema: z.literal(1), completedEpoch: z.int().positive(),
        epochs: z.array(z.object({ epoch: z.int().min(1), trainLoss: z.number().finite(), validationLoss: z.number().finite().nullable(), elapsedSeconds: z.number().finite().nonnegative() })),
        environment: RunnerResponse.Environment,
      }),
      await this.call("train_chunk", snapshot, rows, { prepared, state: state ? state.toString("base64") : null, epochs }, signal));
    return { checkpoint: Buffer.from(output.checkpoint, "base64"), checkpointFormat: output.checkpointFormat, checkpointSchema: output.checkpointSchema, completedEpoch: output.completedEpoch, epochs: output.epochs, environment: output.environment };
  }
  async exportModel(snapshot: Snapshot, checkpoint: Buffer, restoreBest: boolean, signal: AbortSignal): Promise<TrainedModel> {
    const output = this.parse(
      z.object({ artifact: z.string().min(1), format: z.literal(this.artifactFormat), environment: RunnerResponse.Environment, resolvedHyperparameters: z.record(z.string(), z.unknown()), performance: RunnerResponse.Performance }),
      await this.call("export", snapshot, [], { checkpoint: checkpoint.toString("base64"), restoreBest }, signal));
    return { bytes: Buffer.from(output.artifact, "base64"), format: output.format, environment: output.environment, resolvedHyperparameters: output.resolvedHyperparameters, performance: output.performance };
  }
}
