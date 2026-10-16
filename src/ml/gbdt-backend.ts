import { resolve } from "node:path";
import { ConflictError } from "../domain/errors.js";
import type { BackendCapabilities } from "./backend.js";
import { TABULAR_COLUMN_TYPES, type Snapshot } from "./domain.js";
import "./gbdt-models.js"; // registers the xgboost and lightgbm definitions
import { NEURAL_OPERATIONS } from "./neural-backend.js";
import { RunnerBackend } from "./runner-backend.js";

/**
 * Gradient-boosted trees in their native framework (XGBoost or LightGBM), one fit per attempt on CPU. Preprocessing is the
 * NumPy implementation the neural runners share (ml/neural_common.py), with missing numbers passed through as NaN.
 * Early stopping is the framework's own, monitored on the validation partition; Meticule's training policy is refused
 * (not iterative), and recovery after a crash retries the attempt from scratch.
 */
class GradientBoostingBackend extends RunnerBackend {
  constructor(id: "xgboost" | "lightgbm", artifactFormat: string, python: string, timeoutMs: number, script: string) {
    // LightGBM also forecasts (lightgbm_forecaster), reading only the split's time column; XGBoost stays tabular.
    const forecasting = id === "lightgbm";
    const capabilities: BackendCapabilities = { artifactFormats: [artifactFormat], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: NEURAL_OPERATIONS, iterative: false,
      tasks: forecasting ? ["classification", "regression", "forecasting"] : ["classification", "regression"], inputs: forecasting ? [...TABULAR_COLUMN_TYPES, "timestamp"] : TABULAR_COLUMN_TYPES };
    super({ id, capabilities, python, script, timeoutMs });
  }
  override validate(snapshot: Snapshot): void {
    super.validate(snapshot);
    // A cross-validation fold has no separate monitoring partition, so early stopping there is refused, not silently disabled.
    if (snapshot.evaluation.cv && Number(snapshot.hyperparameters.early_stopping_rounds ?? 0) > 0) throw new ConflictError("Early stopping is not supported with cross-validation");
  }
}

export class XgboostBackend extends GradientBoostingBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/xgboost_runner.py")) { super("xgboost", "xgboost-json-v1", python, timeoutMs, script); }
}
export class LightgbmBackend extends GradientBoostingBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/lightgbm_runner.py")) { super("lightgbm", "lightgbm-text-v1", python, timeoutMs, script); }
}
