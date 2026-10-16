import { defineModel, registerModels, type ModelDefinition, type ModelInputs, type ParameterSpec } from "./model-catalog.js";

/**
 * Univariate forecasters (ml/forecasting_common.py; bounds enforced again by the runners). The only feature is the
 * chronological split's time column; the numeric target is the series. `horizon` (1..100) is how many steps every prediction
 * returns (E3); `seasonal_period` sets the seasonal-naive forecast and the MASE scale.
 */
export const FORECAST_INPUTS: ModelInputs = { types: ["timestamp"], min: 1, max: 1 };
export const forecastSettings: Record<string, ParameterSpec> = {
  horizon: { kind: "int", min: 1, max: 100, default: 1 },
  seasonal_period: { kind: "int", min: 1, max: 366, default: 1 },
};
export const lagFeatures: Record<string, ParameterSpec> = {
  lags: { kind: "int", min: 1, max: 64, default: 7 },
  windows: { kind: "intList", minItems: 0, maxItems: 8, min: 2, max: 366, default: [] },
};
const capabilities = (seeded: boolean) => ({ scores: "none", scaleSensitive: false, seeded, iterative: false, devices: ["cpu"], mixedPrecision: false } as const);

export const FORECASTING_MODELS: readonly ModelDefinition[] = [
  defineModel({ backend: "forecasting", id: "seasonal_naive_forecaster", taskType: "forecasting", displayName: "Seasonal naive forecaster", inputs: FORECAST_INPUTS, parameters: forecastSettings, capabilities: capabilities(false) }),
  defineModel({ backend: "forecasting", id: "ridge_forecaster", taskType: "forecasting", displayName: "Ridge forecaster over lags and rolling means", inputs: FORECAST_INPUTS,
    parameters: { ...forecastSettings, ...lagFeatures, alpha: { kind: "float", min: 0, max: 1e6, default: 1 } }, capabilities: capabilities(false) }),
];

registerModels(FORECASTING_MODELS);
