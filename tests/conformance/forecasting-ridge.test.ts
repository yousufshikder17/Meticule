import { forecastTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real forecasting runner (ridge over lags). Requires ML_TEST_PYTHON; CI's main test job provides it.
describeTaskContract(forecastTarget("forecasting", "ridge_forecaster", { lags: 4, windows: [6], horizon: 6 }));
