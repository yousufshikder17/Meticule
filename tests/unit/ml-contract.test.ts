import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";
import { ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2, ML_BACKEND_CONTRACTS, RUNNER_REFUSAL_CODES } from "../../src/ml/contract.js";

// The TypeScript contract and the Python runners are two halves of one protocol. These checks read the Python sources, so a
// version or refusal code that changes on one side only fails here, instead of being recorded and never compared. Every
// ml/*.py file is read, so a new runner module cannot add an unclassified refusal code unnoticed.
const runnerFiles = readdirSync(resolve("ml")).filter(name => name.endsWith(".py")).sort();
const runnerSources = runnerFiles.map(name => readFileSync(join(resolve("ml"), name), "utf8"));

it("the Python runners declare the same contract versions as the TypeScript client", () => {
  const common = readFileSync(resolve("ml/ml_runtime_common.py"), "utf8");
  expect(common.match(/^ML_BACKEND_CONTRACT = "([^"]+)"/m)?.[1]).toBe(ML_BACKEND_CONTRACT);
  expect(common.match(/^ML_BACKEND_CONTRACT_V2 = "([^"]+)"/m)?.[1]).toBe(ML_BACKEND_CONTRACT_V2);
  expect(common).toMatch(/^SUPPORTED_CONTRACTS = \(ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2\)$/m);
  expect(ML_BACKEND_CONTRACTS).toEqual([ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2]);
});

it("every refusal code a Python runner can print is one the TypeScript contract classifies, and none is unused", () => {
  expect(runnerFiles).toEqual(expect.arrayContaining(["media_common.py", "sklearn_runner.py", "ml_runtime_common.py", "neural_common.py", "pytorch_runner.py", "tensorflow_runner.py"]));
  const emitted = new Set<string>();
  for (const source of runnerSources) {
    for (const match of source.matchAll(/RunnerError\(\s*"([a-z_]+)"/g)) emitted.add(match[1]!);
    for (const match of source.matchAll(/"code": "([a-z_]+)"/g)) emitted.add(match[1]!);
  }
  expect([...emitted].sort()).toEqual(RUNNER_REFUSAL_CODES);
});
