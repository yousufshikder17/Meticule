import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";

const jaxPython = process.env.ML_TEST_JAX_PYTHON;

// Every named JAX behaviour is asserted by name, so a silent regression shows up as a specific failure.
it.runIf(Boolean(jaxPython))("real JAX runner: learning, autodiff, determinism, chunking, optimizer state, safe archives and device refusals", () => {
  const out = JSON.parse(execFileSync(jaxPython!, ["-I", resolve("tests/python/check_jax.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(33); // a dropped check must fail the suite, not pass quietly
}, 900_000);

// The dedicated JAX CI job sets ML_JAX_JOB. Only that job must provide the pinned CPU runtime, so the main job can skip it.
it.runIf(process.env.ML_JAX_JOB === "true")("CI provides the pinned CPU JAX runtime", () => {
  expect(jaxPython, "ML_TEST_JAX_PYTHON must be set in the JAX job so real JAX tests run").toBeTruthy();
  const out = execFileSync(jaxPython!, ["-c", "import os; os.environ['JAX_PLATFORMS']='cpu'; import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.default_backend())"], { encoding: "utf8" }).trim();
  expect(out).toBe("0.11.2 0.11.2 cpu");
});
