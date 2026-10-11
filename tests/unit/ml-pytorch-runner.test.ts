import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, it } from "vitest";

const torchPython = process.env.ML_TEST_PYTORCH_PYTHON;

// Every named PyTorch behaviour is asserted by name, so a silent regression shows up as a specific failure.
it.runIf(Boolean(torchPython))("real PyTorch runner: learning, fitting scope, folds, reproducibility, artifacts, compatibility and bounds", () => {
  const out = JSON.parse(execFileSync(torchPython!, ["-I", resolve("tests/python/check_pytorch.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  const failed = Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name);
  expect(failed).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(34); // a dropped check must fail the suite, not pass quietly
}, 900_000);

it.runIf(Boolean(torchPython))("real PyTorch checkpoints: chunked equals uninterrupted, safe restore, tamper and corruption refused, best-epoch export", () => {
  const out = JSON.parse(execFileSync(torchPython!, ["-I", resolve("tests/python/check_checkpoints.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(19);
}, 900_000);

it.runIf(Boolean(torchPython))("real PyTorch devices: explicit CPU, refused or real CUDA, mixed precision gated and device-portable checkpoints", () => {
  const cuda = execFileSync(torchPython!, ["-c", "import torch; print(torch.cuda.is_available())"], { encoding: "utf8" }).trim() === "True";
  const out = JSON.parse(execFileSync(torchPython!, ["-I", resolve("tests/python/check_devices.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  // The count states which branch ran: CUDA-only checks exist only where CUDA is present, and the CPU branch asserts the refusal.
  expect(Object.keys(out.results)).toHaveLength(cuda ? 19 : 9);
}, 900_000);

// CI must never pass by skipping the real PyTorch backend: the environment has to be provided and pinned.
it.runIf(process.env.CI === "true")("CI provides the pinned CPU PyTorch runtime", () => {
  expect(torchPython, "ML_TEST_PYTORCH_PYTHON must be set in CI so real PyTorch tests run").toBeTruthy();
  const version = execFileSync(torchPython!, ["-c", "import torch; print(torch.__version__.split('+')[0])"], { encoding: "utf8" }).trim();
  expect(version).toBe("2.9.1");
  const cuda = execFileSync(torchPython!, ["-c", "import torch; print(torch.cuda.is_available())"], { encoding: "utf8" }).trim();
  expect(cuda).toBe("False");
});
