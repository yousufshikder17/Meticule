import { statSync } from "node:fs";
import { isAbsolute } from "node:path";
import { z } from "zod";
import { AudioBackend } from "./audio-backend.js";
import type { MlBackend } from "./backend.js";
import { LightgbmBackend, XgboostBackend } from "./gbdt-backend.js";
import { JaxBackend } from "./jax-backend.js";
import { PytorchBackend } from "./pytorch-backend.js";
import { SklearnBackend } from "./sklearn-backend.js";
import { TensorflowBackend } from "./tensorflow-backend.js";
import { TransformersBackend } from "./transformers-backend.js";

export function mlConfiguration(env: NodeJS.ProcessEnv = process.env) {
  const python = process.platform === "win32" ? "python" : "python3";
  const config = z.object({
    ML_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_PYTHON: z.string().min(1).default(python),
    ML_TIMEOUT_MS: z.coerce.number().int().min(1000).max(600_000).default(120_000),
    // Every framework backend is optional, off by default and gated by ML_ENABLED. Each has its own interpreter (its own requirements file).
    ML_PYTORCH_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_PYTORCH_PYTHON: z.string().min(1).default(python),
    ML_TENSORFLOW_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_TENSORFLOW_PYTHON: z.string().min(1).default(python),
    ML_JAX_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_JAX_PYTHON: z.string().min(1).default(python),
    ML_XGBOOST_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_XGBOOST_PYTHON: z.string().min(1).default(python),
    ML_LIGHTGBM_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_LIGHTGBM_PYTHON: z.string().min(1).default(python),
    // Pretrained models are provisioned by operators under one platform-wide root, shared by every tenant.
    ML_TRANSFORMERS_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_TRANSFORMERS_PYTHON: z.string().min(1).default(python),
    ML_TRANSFORMERS_MODEL_ROOT: z.string().default(""),
    ML_AUDIO_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_AUDIO_PYTHON: z.string().min(1).default(python),
  }).parse(env);
  const backends = new Map<string, MlBackend>();
  if (config.ML_ENABLED === "true") {
    const backend = new SklearnBackend(config.ML_PYTHON, config.ML_TIMEOUT_MS); backends.set(backend.id, backend);
    if (config.ML_PYTORCH_ENABLED === "true") { const torch = new PytorchBackend(config.ML_PYTORCH_PYTHON, config.ML_TIMEOUT_MS); backends.set(torch.id, torch); }
    if (config.ML_TENSORFLOW_ENABLED === "true") { const tensorflow = new TensorflowBackend(config.ML_TENSORFLOW_PYTHON, config.ML_TIMEOUT_MS); backends.set(tensorflow.id, tensorflow); }
    if (config.ML_JAX_ENABLED === "true") { const jax = new JaxBackend(config.ML_JAX_PYTHON, config.ML_TIMEOUT_MS); backends.set(jax.id, jax); }
    if (config.ML_XGBOOST_ENABLED === "true") { const xgboost = new XgboostBackend(config.ML_XGBOOST_PYTHON, config.ML_TIMEOUT_MS); backends.set(xgboost.id, xgboost); }
    if (config.ML_LIGHTGBM_ENABLED === "true") { const lightgbm = new LightgbmBackend(config.ML_LIGHTGBM_PYTHON, config.ML_TIMEOUT_MS); backends.set(lightgbm.id, lightgbm); }
    if (config.ML_TRANSFORMERS_ENABLED === "true") {
      const root = config.ML_TRANSFORMERS_MODEL_ROOT;
      if (!isAbsolute(root) || !isDirectory(root)) throw new Error("ML_TRANSFORMERS_MODEL_ROOT must be an absolute path to an existing directory");
      const transformers = new TransformersBackend(config.ML_TRANSFORMERS_PYTHON, root, config.ML_TIMEOUT_MS); backends.set(transformers.id, transformers);
    }
    if (config.ML_AUDIO_ENABLED === "true") { const audio = new AudioBackend(config.ML_AUDIO_PYTHON, config.ML_TIMEOUT_MS); backends.set(audio.id, audio); }
  }
  return { enabled: config.ML_ENABLED === "true", backends };
}

function isDirectory(path: string): boolean {
  try { return statSync(path).isDirectory(); } catch { return false; }
}
