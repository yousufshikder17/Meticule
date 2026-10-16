import { resolve } from "node:path";
import { XgboostBackend } from "../../src/ml/gbdt-backend.js";
import { describeContract } from "./contract-suite.js";

// Real XGBoost runner on CPU. Requires ML_TEST_XGBOOST_PYTHON; the gradient-boosting CI job provides it.
const quick = { n_estimators: 30 };
describeContract({
  id: "xgboost",
  create: timeoutMs => new XgboostBackend(process.env.ML_TEST_XGBOOST_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_XGBOOST_PYTHON ?? "",
  script: resolve("ml/xgboost_runner.py"),
  classification: { algorithm: "xgboost_classifier", hyperparameters: quick },
  regression: { algorithm: "xgboost_regressor", hyperparameters: quick },
  requiredInCi: process.env.ML_GBDT_JOB === "true",
});
