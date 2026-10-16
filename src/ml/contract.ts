/**
 * Meticule ML Backend Contracts. Internal only: not a public SDK, and no third-party code is loaded through it.
 *
 * Every runner request carries a version and every successful runner response echoes it. Either side refuses a
 * mismatch, so versions coexist or fail explicitly instead of being misread. v2 is v1 plus the Phase 4 vocabulary
 * (tasks beyond classification/regression, column types beyond number/string/boolean, the chronological split,
 * datasets without a target); envelope, commands, failure categories and artifacts are the same.
 */
import { ConflictError } from "../domain/errors.js";
import { isTabularType, type Snapshot } from "./domain.js";
import type { TaskType } from "./tasks.js";

export const ML_BACKEND_CONTRACT = "meticule-ml-backend/v1";
export const ML_BACKEND_CONTRACT_V2 = "meticule-ml-backend/v2";
export const ML_BACKEND_CONTRACTS = [ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2] as const;
export type MlBackendContract = (typeof ML_BACKEND_CONTRACTS)[number];

/**
 * The contract a request needs (spec sec. 1): v1 only for a classification or regression task on a random split of a
 * dataset that declares a target and whose columns are all number, string or boolean; v2 otherwise. `task` is the
 * algorithm's catalog task. ml/ml_runtime_common.py uses_extensions mirrors this exactly.
 */
export function contractFor(snapshot: Snapshot, task: TaskType): MlBackendContract {
  const { schema } = snapshot.dataset;
  return (task === "classification" || task === "regression") && snapshot.split.strategy === "random" && schema.target !== undefined && schema.columns.every(c => isTabularType(c.type))
    ? ML_BACKEND_CONTRACT : ML_BACKEND_CONTRACT_V2;
}

/**
 * Bounded, machine-readable failure categories. Backend and runner failures are mapped into these before they leave
 * the ML subsystem, so framework exceptions, stderr and stack traces never cross the platform boundary.
 */
export const ML_FAILURE_CATEGORIES = [
  "invalid_request",
  "unsupported_capability",
  "incompatible_artifact",
  "incompatible_checkpoint",
  "invalid_checkpoint",
  "backend_unavailable",
  "device_unavailable",
  "runtime_incompatible",
  "training_failed",
  "evaluation_failed",
  "prediction_failed",
  "invalid_response",
  "output_limit_exceeded",
  "cancelled",
  "timed_out",
] as const;
export type MlFailureCategory = (typeof ML_FAILURE_CATEGORIES)[number];

/**
 * Fixed refusal codes the runners may print on stderr. Only these codes are read; any other text is discarded.
 * The Python runners declare the same codes in ml/ml_runtime_common.py.
 */
const RUNNER_CODES: Record<string, MlFailureCategory> = {
  unsupported_contract: "runtime_incompatible",
  incompatible_artifact: "incompatible_artifact",
  incompatible_checkpoint: "incompatible_checkpoint",
  invalid_checkpoint: "invalid_checkpoint",
  cuda_unavailable: "device_unavailable",
  cuda_unsupported: "unsupported_capability",
  mixed_requires_cuda: "unsupported_capability",
  unsupported_device_or_precision: "unsupported_capability",
  unsupported_task: "unsupported_capability",
  undefined_metric: "invalid_request",
  invalid_time_series: "invalid_request",
  invalid_media: "invalid_request",
};

/** Every refusal code a runner may print. The Python runners must emit exactly this set (see the contract unit test). */
export const RUNNER_REFUSAL_CODES = Object.keys(RUNNER_CODES).sort();

/** A backend or runner failure already classified into the taxonomy. Its message is safe to store and show. */
export class MlBackendFailure extends Error {
  constructor(readonly category: MlFailureCategory, message: string, readonly runnerCode?: string) {
    super(message);
    this.name = "MlBackendFailure";
  }
}

/** Category for a runner's stderr code, or undefined when the code is not one the contract defines. */
export function categoryForRunnerCode(code: unknown): MlFailureCategory | undefined {
  return typeof code === "string" && Object.hasOwn(RUNNER_CODES, code) ? RUNNER_CODES[code] : undefined;
}

/**
 * Category recorded for any failure. A refused request is invalid_request; anything unclassified keeps the phase
 * default, so a framework or runner message never reaches a stored failure.
 */
export function failureCategory(error: unknown, fallback: MlFailureCategory): MlFailureCategory {
  if (error instanceof MlBackendFailure) return error.category;
  if (error instanceof ConflictError) return "invalid_request";
  return fallback;
}
