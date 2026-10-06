import { z } from "zod";
import type { MlBackend } from "./backend.js";
import { SklearnBackend } from "./sklearn-backend.js";

export function mlConfiguration(env: NodeJS.ProcessEnv = process.env) {
  const config = z.object({
    ML_ENABLED: z.enum(["true", "false"]).default("false"),
    ML_PYTHON: z.string().min(1).default(process.platform === "win32" ? "python" : "python3"),
    ML_TIMEOUT_MS: z.coerce.number().int().min(1000).max(600_000).default(120_000),
  }).parse(env);
  const backends = new Map<string, MlBackend>();
  if (config.ML_ENABLED === "true") { const backend = new SklearnBackend(config.ML_PYTHON, config.ML_TIMEOUT_MS); backends.set(backend.id, backend); }
  return { enabled: config.ML_ENABLED === "true", backends };
}
