import { describe, expect, it } from "vitest";
import { loadAuthConfig } from "../../src/config.js";

describe("authentication configuration", () => {
  it("defaults to JWT and requires adequate secret material", () => {
    expect(() => loadAuthConfig({})).toThrow();
    expect(() => loadAuthConfig({ JWT_SECRET: "short" })).toThrow();
    expect(loadAuthConfig({ JWT_SECRET: "a-local-secret-that-is-at-least-32-characters" }).AUTH_MODE).toBe("jwt");
  });

  it("allows development headers only through an explicit non-production mode", () => {
    expect(loadAuthConfig({ AUTH_MODE: "development_headers", NODE_ENV: "development" })).toEqual({ AUTH_MODE: "development_headers" });
    expect(() => loadAuthConfig({ AUTH_MODE: "development_headers", NODE_ENV: "production" })).toThrow(/prohibited/);
  });
});
