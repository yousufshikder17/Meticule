import { FORECAST_INPUTS, forecastSettings, lagFeatures } from "./forecasting-models.js";
import { defineModel, registerModels, type ModelDefinition, type ParameterSpec } from "./model-catalog.js";

// Bounds shared by both gradient-boosting backends (mirrored in ml/gbdt_common.py and the runners, which enforce them again).
const shared: Record<string, ParameterSpec> = {
  n_estimators: { kind: "int", min: 1, max: 2000, default: 100 },
  learning_rate: { kind: "float", min: 0, max: 1, openMin: true, default: 0.1 },
  subsample: { kind: "float", min: 0, max: 1, openMin: true, default: 1 },
  colsample_bytree: { kind: "float", min: 0, max: 1, openMin: true, default: 1 },
  reg_alpha: { kind: "float", min: 0, max: 1000, default: 0 },
  // Framework-native early stopping on the validation partition (0 = off). Not Meticule's training policy.
  early_stopping_rounds: { kind: "int", min: 0, max: 200, default: 0 },
};
const xgboost: Record<string, ParameterSpec> = { ...shared, max_depth: { kind: "int", min: 1, max: 16, default: 6 }, min_child_weight: { kind: "float", min: 0, max: 1000, default: 1 }, reg_lambda: { kind: "float", min: 0, max: 1000, default: 1 } };
const lightgbm: Record<string, ParameterSpec> = { ...shared, max_depth: { kind: "int", min: 1, max: 32, nullable: true, default: null }, num_leaves: { kind: "int", min: 2, max: 1024, default: 31 }, min_child_samples: { kind: "int", min: 1, max: 1000, default: 20 }, reg_lambda: { kind: "float", min: 0, max: 1000, default: 0 } };

// Trees: insensitive to monotone feature scaling, seeded, one fit per attempt on CPU.
const capabilities = (scores: "probability" | "none") => ({ scores, scaleSensitive: false, seeded: true, iterative: false, devices: ["cpu"], mixedPrecision: false } as const);

export const XGBOOST_MODELS: readonly ModelDefinition[] = [
  defineModel({ backend: "xgboost", id: "xgboost_classifier", taskType: "classification", displayName: "XGBoost classifier", parameters: xgboost, capabilities: capabilities("probability") }),
  defineModel({ backend: "xgboost", id: "xgboost_regressor", taskType: "regression", displayName: "XGBoost regressor", parameters: xgboost, capabilities: capabilities("none") }),
];
export const LIGHTGBM_MODELS: readonly ModelDefinition[] = [
  defineModel({ backend: "lightgbm", id: "lightgbm_classifier", taskType: "classification", displayName: "LightGBM classifier", parameters: lightgbm, capabilities: capabilities("probability") }),
  defineModel({ backend: "lightgbm", id: "lightgbm_regressor", taskType: "regression", displayName: "LightGBM regressor", parameters: lightgbm, capabilities: capabilities("none") }),
  // A LightGBM regressor over lag and rolling-mean features, forecasting recursively (ml/forecasting_common.py). No native
  // early stopping: the chronological validation block belongs to evaluation.
  defineModel({ backend: "lightgbm", id: "lightgbm_forecaster", taskType: "forecasting", displayName: "LightGBM forecaster over lags and rolling means", inputs: FORECAST_INPUTS,
    parameters: { ...forecastSettings, ...lagFeatures, ...Object.fromEntries(Object.entries(lightgbm).filter(([name]) => name !== "early_stopping_rounds")) }, capabilities: capabilities("none") }),
];

registerModels([...XGBOOST_MODELS, ...LIGHTGBM_MODELS]);
