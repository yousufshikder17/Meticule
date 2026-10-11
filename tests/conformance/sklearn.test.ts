import { resolve } from "node:path";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { describeContract } from "./contract-suite.js";

// Real scikit-learn runner. Requires ML_TEST_PYTHON; CI always provides it (see .github/workflows/ci.yml).
describeContract({
  id: "sklearn",
  create: timeoutMs => new SklearnBackend(process.env.ML_TEST_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_PYTHON ?? "",
  script: resolve("ml/sklearn_runner.py"),
  classification: { algorithm: "logistic_regression", hyperparameters: {} },
  regression: { algorithm: "ridge", hyperparameters: { alpha: 1 } },
  requiredInCi: process.env.CI === "true",
});
