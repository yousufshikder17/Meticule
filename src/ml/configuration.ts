import { z } from "zod";
import type { MlBackend } from "./backend.js";
import { PytorchBackend } from "./pytorch-backend.js";
import { SklearnBackend } from "./sklearn-backend.js";
import { TensorflowBackend } from "./tensorflow-backend.js";

export function mlConfiguration(env: NodeJS.ProcessEnv = process.env) {
  const python = process.platform === "win32" ? "python" : "python3";
  const config = z.object({
    ML_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_PYTHON: z.string().min(1).default(python),
    ML_TIMEOUT_MS: z.coerce.number().int().min(1000).max(600_000).default(120_000),
    // PyTorch and TensorFlow are optional. Each has its own interpreter because each has its own requirements file.
    ML_PYTORCH_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_PYTORCH_PYTHON: z.string().min(1).default(python),
    ML_TENSORFLOW_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_TENSORFLOW_PYTHON: z.string().min(1).default(python),
  }).parse(env);
  const backends = new Map<string, MlBackend>();
  if (config.ML_ENABLED === "true") {
    const backend = new SklearnBackend(config.ML_PYTHON, config.ML_TIMEOUT_MS); backends.set(backend.id, backend);
    if (config.ML_PYTORCH_ENABLED === "true") { const torch = new PytorchBackend(config.ML_PYTORCH_PYTHON, config.ML_TIMEOUT_MS); backends.set(torch.id, torch); }
    if (config.ML_TENSORFLOW_ENABLED === "true") { const tensorflow = new TensorflowBackend(config.ML_TENSORFLOW_PYTHON, config.ML_TIMEOUT_MS); backends.set(tensorflow.id, tensorflow); }
  }
  return { enabled: config.ML_ENABLED === "true", backends };
}
