import { resolve } from "node:path";
import { JaxBackend } from "../../src/ml/jax-backend.js";
import { describeContract } from "./contract-suite.js";

// Real JAX runner on CPU. Requires ML_TEST_JAX_PYTHON; CI always provides it.
const quick = { epochs: 4, hiddenLayers: [8], batchSize: 16, learningRate: 0.01 };
describeContract({
  id: "jax",
  create: timeoutMs => new JaxBackend(process.env.ML_TEST_JAX_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_JAX_PYTHON ?? "",
  script: resolve("ml/jax_runner.py"),
  classification: { algorithm: "mlp_classifier", hyperparameters: quick },
  regression: { algorithm: "mlp_regressor", hyperparameters: quick },
  requiredInCi: process.env.ML_JAX_JOB === "true",
});
