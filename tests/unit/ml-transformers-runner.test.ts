import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";

const hfPython = process.env.ML_TEST_TRANSFORMERS_PYTHON;

// Every named behaviour is asserted by name. Fixtures are built offline inside the check (no downloads).
it.runIf(Boolean(hfPython))("real transformers runner: provisioning refusals, label binding, embeddings, heads, media bounds, digest re-verification, no network", () => {
  const out = JSON.parse(execFileSync(hfPython!, ["-I", resolve("tests/python/check_transformers.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean>; details: { skipped: string[] } };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  // One reparse-point variant runs on every OS: a symlink, or a directory junction where Windows refuses symlinks.
  expect(out.details.skipped.length).toBeLessThanOrEqual(1);
  expect(Object.keys(out.results)).toHaveLength(52); // a dropped check must fail the suite, not pass quietly
}, 900_000);

// The dedicated transformers CI job sets ML_TRANSFORMERS_JOB. Only that job must provide the pinned CPU runtime.
it.runIf(process.env.ML_TRANSFORMERS_JOB === "true")("CI provides the pinned CPU transformers runtime", () => {
  expect(hfPython, "ML_TEST_TRANSFORMERS_PYTHON must be set in the transformers job").toBeTruthy();
  const out = execFileSync(hfPython!, ["-c", "import torch, transformers, tokenizers, safetensors, PIL; print(torch.__version__.split('+')[0], transformers.__version__, tokenizers.__version__, safetensors.__version__, PIL.__version__, torch.cuda.is_available())"], { encoding: "utf8" }).trim();
  expect(out).toBe("2.9.1 5.17.0 0.23.2 0.8.0 12.3.0 False");
});
