import { resolve } from "node:path";
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { describeContract } from "./contract-suite.js";

// Real PyTorch runner on CPU. Requires ML_TEST_PYTORCH_PYTHON; CI always provides it.
const quick = { epochs: 4, hiddenLayers: [8], batchSize: 16, learningRate: 0.01 };
describeContract({
  id: "pytorch",
  create: timeoutMs => new PytorchBackend(process.env.ML_TEST_PYTORCH_PYTHON ?? "", timeoutMs),
  python: process.env.ML_TEST_PYTORCH_PYTHON ?? "",
  script: resolve("ml/pytorch_runner.py"),
  classification: { algorithm: "mlp_classifier", hyperparameters: quick },
  regression: { algorithm: "mlp_regressor", hyperparameters: quick },
  requiredInCi: process.env.CI === "true",
});
