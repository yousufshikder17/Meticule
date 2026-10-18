import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { listModels } from "../../src/ml/model-catalog.js";

// Every scikit-learn classifier and regressor in the catalog, under two preprocessing chains, exported and run through ONNX
// Runtime within the parity tolerance; plus the refusals. Runs with the MLOps interpreter (its scikit-learn pins match).
const python = process.env.ML_TEST_MLOPS_PYTHON;
it.runIf(Boolean(python))("exports every catalog scikit-learn classifier and regressor to ONNX within tolerance", () => {
  const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_onnx.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, ok]) => ok !== true).map(([name]) => name)).toEqual([]);
  const models = listModels("sklearn").filter(m => m.taskType === "classification" || m.taskType === "regression");
  expect(models).toHaveLength(15);
  expect(Object.keys(out.results)).toHaveLength(models.length * 2 + 4); // a dropped check must fail, not pass quietly
}, 600_000);
