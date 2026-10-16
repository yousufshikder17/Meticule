import { forecastTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real forecasting runner (seasonal naive). Requires ML_TEST_PYTHON; CI's main test job provides it.
describeTaskContract(forecastTarget("forecasting", "seasonal_naive_forecaster", { seasonal_period: 24, horizon: 6 }));
