import { resolve } from "node:path";
import { LightgbmBackend } from "../../src/ml/gbdt-backend.js";
import { describeContract } from "./contract-suite.js";

// Real LightGBM runner on CPU. Requires ML_TEST_LIGHTGBM_PYTHON; the gradient-boosting CI job provides it.
// min_child_samples 2: the suite's preprocessing checks train on ~54 rows with ~16-18 rows per category, below LightGBM's
// default leaf minimum of 20, which would leave those features unused and the checks unable to see them.
const quick = { n_estimators: 30, min_child_samples: 2 };
describeContract({
  id: "lightgbm",
  create: timeoutMs => new LightgbmBackend(process.env.ML_TEST_LIGHTGBM_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_LIGHTGBM_PYTHON ?? "",
  script: resolve("ml/lightgbm_runner.py"),
  classification: { algorithm: "lightgbm_classifier", hyperparameters: quick },
  regression: { algorithm: "lightgbm_regressor", hyperparameters: quick },
  requiredInCi: process.env.ML_GBDT_JOB === "true",
});
