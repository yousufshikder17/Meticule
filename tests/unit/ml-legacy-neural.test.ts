import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";

const torchPython = process.env.ML_TEST_PYTORCH_PYTHON;
const guard = resolve("tests/python/legacy/phase3_neural/guard.py");

// Spec amendment B8: artifacts and checkpoints written by the pre-Phase-4 neural runner (frozen copies from public 1b4118f) must
// keep predicting and resuming in the current runner, so neural_common's persisted preprocessing state and fingerprint()
// cannot drift unnoticed. Two processes, because both versions import a module named neural_common.
it.runIf(Boolean(torchPython))("the current PyTorch runner predicts with and resumes from a frozen Phase 3 runner's artifact and checkpoint", () => {
  const directory = mkdtempSync(join(tmpdir(), "meticule-legacy-neural-"));
  try {
    const options = { encoding: "utf8" as const, maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" } };
    const record = join(directory, "frozen.json");
    execFileSync(torchPython!, ["-I", "-B", guard, "frozen", record], options);
    const out = JSON.parse(execFileSync(torchPython!, ["-I", "-B", guard, "current", record], options).trim().split("\n").pop()!) as { results: Record<string, boolean> };
    const expected = ["frozen_copies_are_phase3", ...["classifier", "regressor"].flatMap(model => [`${model}_prepare_matches_frozen`,
      `${model}_frozen_artifact_predicts_identically`, `${model}_frozen_checkpoint_resumes_identically`])];
    expect(Object.keys(out.results).sort()).toEqual([...expected].sort());
    for (const name of expected) expect(out.results[name], name).toBe(true);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}, 600_000);
