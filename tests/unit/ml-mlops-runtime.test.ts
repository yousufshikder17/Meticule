import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

// The Phase 5 runners (ONNX, Optuna, reports, MLflow) share one interpreter. Its scikit-learn stack must equal the training
// interpreter's pins, because ONNX export unpickles scikit-learn artifacts under the same compatibility check.
const python = process.env.ML_TEST_MLOPS_PYTHON;
const pins = (file: string) => Object.fromEntries(readFileSync(file, "utf8").split("\n").map(l => l.trim()).filter(l => /^[A-Za-z0-9_.-]+==/.test(l)).map(l => l.split("==") as [string, string]));
const MODULES: Record<string, string> = { "scikit-learn": "sklearn", numpy: "numpy", scipy: "scipy", joblib: "joblib", threadpoolctl: "threadpoolctl", onnx: "onnx", onnxruntime: "onnxruntime", skl2onnx: "skl2onnx", optuna: "optuna", matplotlib: "matplotlib", "mlflow-skinny": "mlflow" };

it.runIf(Boolean(python))("the MLOps interpreter has exactly the pinned versions", () => {
  const expected = { ...pins("ml/requirements.txt"), ...pins("ml/requirements-mlops.txt") };
  const names = Object.keys(MODULES);
  const script = `import importlib, json; print(json.dumps({n: importlib.import_module(m).__version__ for n, m in ${JSON.stringify(Object.entries(MODULES))}}))`;
  const actual = JSON.parse(execFileSync(python!, ["-c", script], { encoding: "utf8" }).trim().split("\n").pop()!) as Record<string, string>;
  expect(Object.fromEntries(names.map(n => [n, actual[n]]))).toEqual(Object.fromEntries(names.map(n => [n, expected[n]])));
}, 120_000);

// The dedicated MLOps CI job sets ML_MLOPS_JOB; there the real-runner tests must not skip quietly.
it.runIf(process.env.ML_MLOPS_JOB === "true")("CI provides the MLOps and scikit-learn interpreters", () => {
  expect(python, "ML_TEST_MLOPS_PYTHON must name the MLOps venv").toBeTruthy();
  expect(process.env.ML_TEST_PYTHON, "ML_TEST_PYTHON must name a scikit-learn venv").toBeTruthy();
});
