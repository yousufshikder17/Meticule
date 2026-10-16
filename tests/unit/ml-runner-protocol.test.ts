import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, expect, it } from "vitest";
import { ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2, MlBackendFailure } from "../../src/ml/contract.js";
import { callRunner } from "../../src/ml/runner-process.js";

// The runner client is a process boundary, so these checks drive it with small Node programs in place of a Python runner.
// That keeps them independent of any ML interpreter while still exercising every classification the client performs.
const scripts: Record<string, string> = {
  // Echoes the envelope's contract so the test can see the client sent it.
  echo: `let d=""; for await (const c of process.stdin) d+=c; const r=JSON.parse(d); process.stdout.write(JSON.stringify({contract: r.contract, command: r.command}));`,
  // A refusal on the last stderr line, after far more library noise than the old 4 KB cap.
  noisyRefusal: `process.stderr.write("library warning ".repeat(2000)+"\\n"); process.stderr.write(JSON.stringify({error:"RunnerError",code:"cuda_unavailable",message:"x"})+"\\n"); process.exit(1);`,
  // A code the contract does not define is never reported, and the phase default applies.
  unknownCode: `process.stderr.write(JSON.stringify({error:"RunnerError",code:"surprise_code",message:"x"})); process.exit(1);`,
  wrongContract: `process.stdout.write(JSON.stringify({contract:"meticule-ml-backend/v0"}));`,
  v1Only: `process.stdout.write(JSON.stringify({contract:"meticule-ml-backend/v1"}));`,
  v2Only: `process.stdout.write(JSON.stringify({contract:"meticule-ml-backend/v2"}));`,
  environment: `process.stdout.write(JSON.stringify({contract:"meticule-ml-backend/v1",env:Object.fromEntries(Object.entries(process.env).filter(([k])=>k.startsWith("HF_")||k.startsWith("TRANSFORMERS_")||k==="TOKENIZERS_PARALLELISM"))}));`,
  notJson: `process.stdout.write("not json");`,
  oversized: `const chunk="x".repeat(1024*1024); for (let i=0;i<65;i++) process.stdout.write(chunk);`,
  hangs: `setTimeout(()=>{}, 60000);`,
};

let directory: string;
beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "meticule-runner-protocol-"));
  for (const [name, body] of Object.entries(scripts)) await writeFile(join(directory, `${name}.mjs`), body);
});
afterAll(async () => { await rm(directory, { recursive: true, force: true }); });

const run = (name: string, timeoutMs = 30_000, command = "prepare") => callRunner({
  label: "protocol-test", python: process.execPath, script: join(directory, `${name}.mjs`), timeoutMs, request: { command }, signal: new AbortController().signal,
});
async function failureOf(promise: Promise<unknown>): Promise<MlBackendFailure> {
  try { await promise; } catch (error) { if (error instanceof MlBackendFailure) return error; throw error; }
  throw new Error("expected a classified failure");
}

it("sends the contract with every request", async () => {
  expect(await run("echo")).toEqual({ contract: ML_BACKEND_CONTRACT, command: "prepare" });
}, 60_000);

it("requires the response to echo exactly the contract that was sent", async () => {
  const v2 = (name: string) => callRunner({ label: "protocol-test", python: process.execPath, script: join(directory, `${name}.mjs`), timeoutMs: 30_000, request: { command: "prepare" }, signal: new AbortController().signal, contract: ML_BACKEND_CONTRACT_V2 });
  expect(await v2("echo")).toEqual({ contract: ML_BACKEND_CONTRACT_V2, command: "prepare" });
  expect((await failureOf(v2("v1Only"))).category).toBe("runtime_incompatible");
  // A v1 request (the default) never accepts a v2 answer either.
  expect((await failureOf(run("v2Only"))).category).toBe("runtime_incompatible");
}, 60_000);

it("fixes offline settings for every runner without inheriting tokens or cache paths", async () => {
  const names = ["HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"];
  const before = names.map(name => process.env[name]);
  try {
    for (const name of names) process.env[name] = "must-not-inherit";
    expect(await run("environment")).toEqual({ contract: ML_BACKEND_CONTRACT, env: { HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1", HF_HUB_DISABLE_TELEMETRY: "1", TOKENIZERS_PARALLELISM: "false" } });
  } finally {
    names.forEach((name, i) => { if (before[i] === undefined) delete process.env[name]; else process.env[name] = before[i]; });
  }
}, 60_000);

it("keeps a refusal code that follows a large amount of stderr noise", async () => {
  const failure = await failureOf(run("noisyRefusal"));
  expect(failure.category).toBe("device_unavailable");
  expect(failure.runnerCode).toBe("cuda_unavailable");
}, 60_000);

it("never reports a stderr code the contract does not define", async () => {
  const failure = await failureOf(run("unknownCode"));
  expect(failure.category).toBe("training_failed");
  expect(failure.runnerCode).toBeUndefined();
  expect(failure.message).not.toContain("surprise_code");
}, 60_000);

it("refuses a response from another contract", async () => {
  expect((await failureOf(run("wrongContract"))).category).toBe("runtime_incompatible");
}, 60_000);

it("refuses a response that is not JSON as an invalid response", async () => {
  expect((await failureOf(run("notJson"))).category).toBe("invalid_response");
}, 60_000);

it("kills a runner whose output exceeds the limit", async () => {
  expect((await failureOf(run("oversized", 120_000))).category).toBe("output_limit_exceeded");
}, 180_000);

it("kills a runner that exceeds its time budget", async () => {
  expect((await failureOf(run("hangs", 500))).category).toBe("timed_out");
}, 60_000);

it("reports a runner that cannot start as unavailable, without the interpreter path", async () => {
  const failure = await failureOf(callRunner({ label: "protocol-test", python: join(directory, "no-such-interpreter"), script: join(directory, "echo.mjs"), timeoutMs: 30_000, request: { command: "prepare" }, signal: new AbortController().signal }));
  expect(failure.category).toBe("backend_unavailable");
  expect(failure.message).not.toContain("no-such-interpreter");
}, 60_000);
