import { resolve } from "node:path";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";
import { describeContract } from "./contract-suite.js";

// Real TensorFlow runner on CPU. Requires ML_TEST_TENSORFLOW_PYTHON; the TensorFlow CI job provides it.
const quick = { epochs: 4, hiddenLayers: [8], batchSize: 16, learningRate: 0.01 };
describeContract({
  id: "tensorflow",
  create: timeoutMs => new TensorflowBackend(process.env.ML_TEST_TENSORFLOW_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_TENSORFLOW_PYTHON ?? "",
  script: resolve("ml/tensorflow_runner.py"),
  classification: { algorithm: "mlp_classifier", hyperparameters: quick },
  regression: { algorithm: "mlp_regressor", hyperparameters: quick },
  requiredInCi: process.env.ML_TENSORFLOW_JOB === "true",
});
