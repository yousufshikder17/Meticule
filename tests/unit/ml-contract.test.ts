import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { ML_BACKEND_CONTRACT, RUNNER_REFUSAL_CODES } from "../../src/ml/contract.js";

// The TypeScript contract and the Python runners are two halves of one protocol. These checks read every runner source in
// ml/, so a version or refusal code that changes on one side only fails here, instead of being recorded and never compared.
const runnerSources = readdirSync(resolve("ml")).filter(name => name.endsWith(".py")).map(name => readFileSync(resolve("ml", name), "utf8"));

it("the Python runners declare the same contract version as the TypeScript client", () => {
  const declared = readFileSync(resolve("ml/ml_runtime_common.py"), "utf8").match(/^ML_BACKEND_CONTRACT = "([^"]+)"/m)?.[1];
  expect(declared).toBe(ML_BACKEND_CONTRACT);
});

it("every refusal code a Python runner can print is one the TypeScript contract classifies, and none is unused", () => {
  const emitted = new Set<string>();
  for (const source of runnerSources) {
    for (const match of source.matchAll(/RunnerError\(\s*"([a-z_]+)"/g)) emitted.add(match[1]!);
    for (const match of source.matchAll(/"code": "([a-z_]+)"/g)) emitted.add(match[1]!);
  }
  expect([...emitted].sort()).toEqual(RUNNER_REFUSAL_CODES);
});
