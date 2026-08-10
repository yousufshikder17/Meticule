import { describe, expect, it } from "vitest";
import { closeServer } from "../../src/deployment/graceful-shutdown.js";
import { runSchedulerCycle } from "../../src/worker/scheduler-cycle.js";
import { loadConfig } from "../../src/config.js";
import { readFileSync } from "node:fs";

describe("deployment hardening", () => {
  it("stops scheduler phases between bounded tasks while preserving progress", async () => {
    let stopping = false; const calls: string[] = [];
    const progressed = await runSchedulerCycle([
      async () => { calls.push("first"); stopping = true; return true; },
      async () => { calls.push("must-not-run"); return true; },
    ], () => stopping);
    expect(progressed).toBe(true); expect(calls).toEqual(["first"]);
  });

  it("waits for ordinary HTTP drain and force-closes at the configured deadline", async () => {
    expect(await closeServer({ close: (callback) => callback() }, 100)).toBe("drained");
    let forced = false; expect(await closeServer({ close: () => undefined, closeAllConnections: () => { forced = true; } }, 5)).toBe("forced"); expect(forced).toBe(true);
  });

  it("validates explicit database pool and shutdown bounds", () => {
    const config = loadConfig({ DATABASE_URL: "postgres://example.invalid/db", DB_POOL_MAX: "7", DB_POOL_IDLE_TIMEOUT_MS: "4000", DB_CONNECT_TIMEOUT_MS: "2000", SHUTDOWN_GRACE_MS: "9000" });
    expect(config).toMatchObject({ DB_POOL_MAX: 7, DB_POOL_IDLE_TIMEOUT_MS: 4000, DB_CONNECT_TIMEOUT_MS: 2000, SHUTDOWN_GRACE_MS: 9000 });
  });

  it("supplies identical provider configuration surfaces to API and worker containers", () => {
    const compose = readFileSync("compose.yaml", "utf8");
    for (const key of ["OLLAMA_ENABLED", "OLLAMA_MODELS", "OPENAI_COMPAT_ENABLED", "ANTHROPIC_ENABLED", "GEMINI_ENABLED"]) {
      expect(compose.match(new RegExp(`^\\s+${key}:`, "gm"))).toHaveLength(2);
    }
  });
});
