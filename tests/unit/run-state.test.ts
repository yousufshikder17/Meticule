import { describe, expect, it } from "vitest";
import { RUN_STATES, assertTransition, canTransition, isTerminal } from "../../src/domain/run-state.js";

describe("canonical run state policy", () => {
  it("allows every documented forward/recovery edge", () => {
    expect(canTransition("queued", "claimed")).toBe(true);
    expect(canTransition("claimed", "running")).toBe(true);
    expect(canTransition("running", "waiting_for_approval")).toBe(true);
    expect(canTransition("waiting_for_approval", "queued")).toBe(true);
    expect(canTransition("running", "paused")).toBe(true);
    expect(canTransition("paused", "queued")).toBe(true);
    expect(canTransition("running", "completed")).toBe(true);
    expect(canTransition("cancelling", "cancelled")).toBe(true);
  });

  it("rejects invalid shortcuts and all terminal transitions", () => {
    expect(() => assertTransition("queued", "completed")).toThrow("queued -> completed");
    expect(() => assertTransition("waiting_for_approval", "running")).toThrow();
    for (const terminal of ["completed", "failed", "cancelled"] as const) {
      expect(isTerminal(terminal)).toBe(true);
      for (const target of RUN_STATES) expect(canTransition(terminal, target)).toBe(false);
    }
  });
});
