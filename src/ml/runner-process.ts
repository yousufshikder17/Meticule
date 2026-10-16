import { spawn } from "node:child_process";
import { ML_BACKEND_CONTRACT, MlBackendFailure, categoryForRunnerCode, type MlBackendContract, type MlFailureCategory } from "./contract.js";

const MAX_OUTPUT_BYTES = 64 * 1024 * 1024;
// A runner exit that carries no fixed code is classified by the operation that failed, never by a framework message.
const FALLBACK_CATEGORY: Record<string, MlFailureCategory> = { evaluate: "evaluation_failed", predict: "prediction_failed" };

export interface RunnerCall {
  /** Backend identity used in error messages only; never part of the protocol. */
  label: string;
  python: string;
  script: string;
  timeoutMs: number;
  /** Protocol command plus its JSON payload, written to the runner's stdin as one object. */
  request: Record<string, unknown>;
  signal: AbortSignal;
  /** Contract the request speaks; v1 unless the caller selects one (RunnerBackend uses contractFor). */
  contract?: MlBackendContract;
}

/**
 * Runs one bounded computation in a fresh Python process and returns its JSON response.
 * The child is killed on timeout or abort (lease loss, cancellation); it never outlives the call.
 * Every request carries the contract version, and every response must echo exactly that version (A2).
 */
export function callRunner({ label, python, script, timeoutMs, request, signal, contract = ML_BACKEND_CONTRACT }: RunnerCall): Promise<unknown> {
  return new Promise((resolveCall, reject) => {
    const envelope = { ...request, contract };
    const command = String(request.command);
    let timedOut = false; let oversize = false;
    const child = spawn(python, [script], { windowsHide: true, shell: false, signal, env: runnerEnvironment(), stdio: ["pipe", "pipe", "pipe"] });
    let size = 0; const chunks: Buffer[] = []; let errors = "";
    const timeout = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => { size += chunk.length; if (size > MAX_OUTPUT_BYTES) { oversize = true; child.kill(); } else chunks.push(chunk); });
    // Keeps the tail: a refusal is the runner's last line, and it must survive any library noise printed before it.
    child.stderr.on("data", (chunk: Buffer) => { errors = (errors + chunk.toString("utf8")).slice(-4096); });
    child.stdin.on("error", () => undefined);
    // A runner that cannot start is backend_unavailable. The OS error is not forwarded: it names the interpreter path.
    child.once("error", () => { clearTimeout(timeout); reject(signal.aborted ? new MlBackendFailure("cancelled", `${label} ${command} was cancelled`) : new MlBackendFailure("backend_unavailable", `${label} runner could not start`)); });
    child.once("close", code => {
      clearTimeout(timeout);
      if (signal.aborted) { reject(new MlBackendFailure("cancelled", `${label} ${command} was cancelled`)); return; }
      if (timedOut) { reject(new MlBackendFailure("timed_out", `${label} ${command} timed out`)); return; }
      if (oversize) { reject(new MlBackendFailure("output_limit_exceeded", `${label} ${command} produced too much output`)); return; }
      if (code !== 0) {
        // Only a fixed runner code is surfaced. Free-form stderr is never returned to callers.
        const runnerCode = runnerErrorCode(errors);
        const category = categoryForRunnerCode(runnerCode) ?? FALLBACK_CATEGORY[command] ?? "training_failed";
        reject(new MlBackendFailure(category, `${label} ${command} failed (${runnerCode ?? code ?? "terminated"})`, runnerCode)); return;
      }
      let response: unknown;
      try { response = JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { reject(new MlBackendFailure("invalid_response", "Invalid runner response")); return; }
      if (!response || typeof response !== "object" || (response as { contract?: unknown }).contract !== contract) {
        reject(new MlBackendFailure("runtime_incompatible", `${label} runner speaks a different ML backend contract`)); return;
      }
      resolveCall(response);
    });
    child.stdin.end(JSON.stringify(envelope));
  });
}

/**
 * Least-privilege environment for the runner: only what Python and the native libraries need to start. The runner
 * inherits no database URL, JWT secret, provider key or other process secret, so a malicious artifact that reaches the
 * interpreter has nothing of the platform's to read. PYTHONPATH and PYTHONHOME are never inherited, so the interpreter
 * imports only the runner directory and its own installation.
 */
const INHERITED = ["PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL", "CUDA_PATH", "CUDA_VISIBLE_DEVICES"];
function runnerEnvironment(): NodeJS.ProcessEnv {
  // Hugging Face libraries are pinned offline with no telemetry; no token or cache path is ever inherited.
  const env: NodeJS.ProcessEnv = { OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1", MKL_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1", PYTHONIOENCODING: "utf-8",
    HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1", HF_HUB_DISABLE_TELEMETRY: "1", TOKENIZERS_PARALLELISM: "false" };
  for (const name of INHERITED) if (process.env[name] !== undefined) env[name] = process.env[name];
  return env;
}

/**
 * The runner's last stderr line is a JSON refusal when it failed on purpose. Returns its code only when the code is
 * one the contract defines, so arbitrary text printed by a library can never become a reported code.
 */
function runnerErrorCode(stderr: string): string | undefined {
  const line = stderr.trim().split("\n").pop() ?? "";
  if (!line.startsWith("{")) return undefined;
  try {
    const code = (JSON.parse(line) as { code?: unknown }).code;
    return categoryForRunnerCode(code) ? code as string : undefined;
  } catch { return undefined; }
}
