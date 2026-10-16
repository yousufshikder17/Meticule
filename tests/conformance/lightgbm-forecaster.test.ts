import { forecastTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real LightGBM forecaster. Requires ML_TEST_LIGHTGBM_PYTHON; the gradient-boosting CI job provides it.
describeTaskContract(forecastTarget("lightgbm", "lightgbm_forecaster", { lags: 4, windows: [6], horizon: 6, n_estimators: 40, min_child_samples: 5 }));
