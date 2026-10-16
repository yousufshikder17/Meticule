import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";

// One interpreter hosts both frameworks (ml/requirements-gbdt.txt); the two variables may name the same venv.
const gbdtPython = process.env.ML_TEST_XGBOOST_PYTHON && process.env.ML_TEST_LIGHTGBM_PYTHON === process.env.ML_TEST_XGBOOST_PYTHON ? process.env.ML_TEST_XGBOOST_PYTHON : undefined;

// Every named behaviour is asserted by name, so a silent regression shows up as a specific failure.
it.runIf(Boolean(gbdtPython))("real XGBoost and LightGBM runners: learning, determinism, validation-only early stopping, best iteration, NaN, safe envelopes and refusals", () => {
  const out = JSON.parse(execFileSync(gbdtPython!, ["-I", resolve("tests/python/check_gbdt.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(2 * 22 + 2); // a dropped check must fail the suite, not pass quietly
}, 900_000);

// The dedicated gradient-boosting CI job sets ML_GBDT_JOB. Only that job must provide the pinned runtimes.
it.runIf(process.env.ML_GBDT_JOB === "true")("CI provides the pinned XGBoost and LightGBM runtimes", () => {
  expect(gbdtPython, "ML_TEST_XGBOOST_PYTHON and ML_TEST_LIGHTGBM_PYTHON must name the gradient-boosting venv").toBeTruthy();
  expect(execFileSync(gbdtPython!, ["-c", "import xgboost, lightgbm, pandas; print(xgboost.__version__, lightgbm.__version__, pandas.__version__)"], { encoding: "utf8" }).trim()).toBe("3.4.1 4.7.0 3.0.6");
});
