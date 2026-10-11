import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, expect, it } from "vitest";
import { ML_BACKEND_CONTRACT } from "../../src/ml/contract.js";
import { callRunner } from "../../src/ml/runner-process.js";

// Leakage check: a runner must never inherit platform secrets. The planted value below stands in for a JWT secret or a
// database URL; if it reaches the child, this test fails. Runs only when a Python interpreter is configured.
const python = process.env.ML_TEST_PYTHON;
const secretName = "PLATFORM_TEST_SECRET";
let directory: string; let script: string;

beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "meticule-runner-env-"));
  script = join(directory, "echo_environment.py");
  // Echoes the contract and the variable names it can see. It never prints values.
  await writeFile(script, [
    "import json, os, sys",
    "request = json.load(sys.stdin)",
    `print(json.dumps({"contract": "${ML_BACKEND_CONTRACT}", "echoed": request["command"], "names": sorted(os.environ)}))`,
  ].join("\n"));
});
afterAll(async () => { await rm(directory, { recursive: true, force: true }); });

it.runIf(Boolean(python))("runs the interpreter without inheriting process secrets, and echoes the contract", async () => {
  process.env[secretName] = "must-not-leak";
  try {
    const response = await callRunner({ label: "test", python: python!, script, timeoutMs: 30_000, request: { command: "probe" }, signal: new AbortController().signal }) as { names: string[]; echoed: string; contract: string };
    expect(response.contract).toBe(ML_BACKEND_CONTRACT);
    expect(response.echoed).toBe("probe");
    expect(response.names).not.toContain(secretName);
    expect(response.names).not.toContain("PYTHONPATH");
    expect(response.names).not.toContain("PYTHONHOME");
  } finally {
    delete process.env[secretName];
  }
}, 60_000);
