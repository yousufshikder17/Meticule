import { z } from "zod";
import { MAX_TEXT_CHARS, type ColumnType, type SplitStrategy, type Snapshot } from "./domain.js";
import { timestampMs } from "./media.js";
import { ConflictError } from "../domain/errors.js";
import { MlBackendFailure } from "./contract.js";
import type { ModelDefinition } from "./model-catalog.js";

/**
 * The task registry (spec sec. 4): the single source of what a task means. Backends declare tasks, the catalog assigns one
 * per model, and validation, prediction and metrics read the rules here, never a backend or model name.
 */
export const TASK_TYPES = ["classification", "regression", "forecasting", "anomaly_detection", "embedding", "transcription"] as const;
export type TaskType = (typeof TASK_TYPES)[number];

export interface TaskDefinition {
  /** Whether a dataset must declare a target. An optional target, when present, must still have one of `targetTypes`. */
  readonly target: "required" | "optional";
  readonly targetTypes: readonly ColumnType[];
  readonly splits: readonly SplitStrategy[];
  /** Cross-validation is allowed, and then only with a random split. */
  readonly crossValidation: boolean;
  /** Prediction rows carry the target column (forecasting history). */
  readonly predictionIncludesTarget: boolean;
  /** One prediction per input row, or `horizon` predictions per request. */
  readonly outputs: "row" | "horizon";
  /** One prediction as the runner must return it. */
  readonly output: z.ZodType;
  /** Rows one prediction request may carry before the media caps (maxPredictionRows). */
  readonly maxPredictionRows: number;
}

const label = z.union([z.string().max(MAX_TEXT_CHARS), z.number().finite(), z.boolean()]);
const both = ["random", "chronological"] as const;
export const TASKS: Readonly<Record<TaskType, TaskDefinition>> = {
  classification: { target: "required", targetTypes: ["number", "string", "boolean"], splits: both, crossValidation: true, predictionIncludesTarget: false, outputs: "row", output: label, maxPredictionRows: 1000 },
  regression: { target: "required", targetTypes: ["number"], splits: both, crossValidation: true, predictionIncludesTarget: false, outputs: "row", output: z.number().finite(), maxPredictionRows: 1000 },
  forecasting: {
    target: "required", targetTypes: ["number"], splits: ["chronological"], crossValidation: false, predictionIncludesTarget: true, outputs: "horizon",
    output: z.strictObject({ step: z.int().min(1), timestamp: z.string().refine(s => timestampMs(s) !== undefined, "Invalid timestamp"), value: z.number().finite() }),
    maxPredictionRows: 5000, // history rows
  },
  anomaly_detection: { target: "optional", targetTypes: ["boolean"], splits: both, crossValidation: false, predictionIncludesTarget: false, outputs: "row", output: z.strictObject({ score: z.number().finite(), anomaly: z.boolean() }), maxPredictionRows: 1000 },
  embedding: { target: "optional", targetTypes: ["number", "string", "boolean"], splits: both, crossValidation: false, predictionIncludesTarget: false, outputs: "row", output: z.array(z.number().finite()).min(1).max(4096), maxPredictionRows: 256 },
  // Speech to text: the target is the reference transcript (non-nullable text); one transcript per clip.
  transcription: { target: "required", targetTypes: ["text"], splits: ["random"], crossValidation: false, predictionIncludesTarget: false, outputs: "row", output: z.string().max(MAX_TEXT_CHARS), maxPredictionRows: 8 },
};

/**
 * Rows one prediction request may carry: the task's cap, lowered when the rows carry media (C5): image 32, audio 16, and
 * text for an embedding model 64. `types` are the column types of the prediction rows.
 */
export function maxPredictionRows(task: TaskType, types: readonly ColumnType[]): number {
  return Math.min(TASKS[task].maxPredictionRows, types.includes("image") ? 32 : Infinity, types.includes("audio") ? 16 : Infinity, task === "embedding" && types.includes("text") ? 64 : Infinity);
}

/** Artifact metadata must be an integer spelling, never a coerced number (E1). */
export function embeddingDimension(environment: Record<string, string>): number {
  const raw = environment.embedding_dimension;
  if (typeof raw !== "string" || !/^[1-9][0-9]{0,3}$/.test(raw) || Number(raw) > 4096) throw new MlBackendFailure("invalid_response", "Invalid embedding dimension");
  return Number(raw);
}

/** Predictions a request must return: one per row, or the forecasting horizon read like the epoch count (E3). */
export function predictionCount(model: ModelDefinition, snapshot: Snapshot, rows: number): number {
  if (TASKS[model.taskType].outputs === "row") return rows;
  const horizon = { ...model.defaults, ...snapshot.hyperparameters }.horizon;
  if (typeof horizon !== "number" || !Number.isInteger(horizon) || horizon < 1 || horizon > 100) throw new ConflictError("Forecasting requires a horizon from 1 to 100");
  return horizon;
}

/** Shared by the runner and service boundaries, including backends that do not extend RunnerBackend. */
export function validatePredictions(task: TaskType, raw: unknown, count: number, environment: Record<string, string>): unknown[] {
  const result = z.array(TASKS[task].output).length(count).safeParse(raw);
  if (!result.success) throw new MlBackendFailure("invalid_response", "Predictions did not match the task contract");
  if (task === "embedding") {
    const dimension = embeddingDimension(environment);
    if ((result.data as number[][]).some(vector => vector.length !== dimension)) throw new MlBackendFailure("invalid_response", "Prediction embedding dimension mismatch");
  }
  if (task === "forecasting" && (result.data as { step: number }[]).some((item, i) => item.step !== i + 1)) throw new MlBackendFailure("invalid_response", "Prediction forecast steps mismatch");
  return result.data;
}
