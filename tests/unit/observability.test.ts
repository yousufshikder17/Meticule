import { describe, expect, it } from "vitest";
import { redact } from "../../src/observability/redaction.js";
import { JsonLogger } from "../../src/observability/logger.js";

describe("observability redaction", () => {
  it("redacts sensitive keys and bearer credentials recursively", () => {
    expect(redact({ apiKey: "private", nested: { password: "private", note: "Bearer abc.def.ghi" }, safe: "visible" })).toEqual({
      apiKey: "[REDACTED]", nested: { password: "[REDACTED]", note: "Bearer [REDACTED]" }, safe: "visible",
    });
  });

  it("emits structured JSON without raw error secrets", () => {
    const records: string[] = []; const sink = { log: (value: string) => records.push(value), warn: (value: string) => records.push(value), error: (value: string) => records.push(value) };
    new JsonLogger(sink).log("error", "provider.failed", { correlationId: "request-1", authorization: "Bearer private", error: new Error("Bearer private-token") });
    const parsed = JSON.parse(records[0]!) as Record<string, unknown>;
    expect(parsed.event).toBe("provider.failed"); expect(parsed.authorization).toBe("[REDACTED]"); expect(JSON.stringify(parsed)).not.toContain("private-token");
  });
});
