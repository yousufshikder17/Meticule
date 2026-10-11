import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";

const tensorflowPython = process.env.ML_TEST_TENSORFLOW_PYTHON;

// Every named TensorFlow behaviour is asserted by name, so a silent regression shows up as a specific failure.
it.runIf(Boolean(tensorflowPython))("real TensorFlow runner: learning, reproducibility, chunked and dropout-aware resume, refusals, folds and CPU-only devices", () => {
  const out = JSON.parse(execFileSync(tensorflowPython!, ["-I", resolve("tests/python/check_tensorflow.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  const failed = Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name);
  expect(failed).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(28); // a dropped check must fail the suite, not pass quietly
}, 900_000);

// The dedicated TensorFlow CI job sets ML_TENSORFLOW_JOB. Only that job must provide the pinned runtime, so the main job can skip it.
it.runIf(process.env.ML_TENSORFLOW_JOB === "true")("CI provides the pinned TensorFlow runtime", () => {
  expect(tensorflowPython, "ML_TEST_TENSORFLOW_PYTHON must be set in the TensorFlow job so real TensorFlow tests run").toBeTruthy();
  const versions = execFileSync(tensorflowPython!, ["-c", "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"], { encoding: "utf8", env: { ...process.env, TF_CPP_MIN_LOG_LEVEL: "3" } }).trim();
  expect(versions).toBe("2.20.0 3.15.1");
}, 120_000);
