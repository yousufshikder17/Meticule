import { describe, expect, it } from "vitest";
import { EnvironmentConnectorCredentialResolver } from "../../src/connectors/credential-resolver.js";
import { ConnectorNetworkPolicy } from "../../src/connectors/network-policy.js";
import { McpHttpTransport } from "../../src/connectors/mcp-transport.js";
import { compileUntrustedSchema, MCP_PROTOCOL_VERSION } from "../../src/connectors/mcp-schemas.js";

const publicLookup = async () => [{ address: "93.184.216.34", family: 4 }];

describe("controlled connector components", () => {
  it("enforces exact origins, rejects redirects/userinfo, and blocks private resolution", async () => {
    const policy = new ConnectorNetworkPolicy(["https://mcp.example.invalid"], false, publicLookup);
    expect((await policy.assertAllowed("https://mcp.example.invalid/v1/mcp")).pathname).toBe("/v1/mcp");
    await expect(policy.assertAllowed("https://other.example.invalid/mcp")).rejects.toThrow(/allowlisted/);
    await expect(policy.assertAllowed("https://user:password@mcp.example.invalid/mcp")).rejects.toThrow(/userinfo/);
    const privatePolicy = new ConnectorNetworkPolicy(["http://localhost:4000"], false, async () => [{ address: "127.0.0.1", family: 4 }]);
    await expect(privatePolicy.assertAllowed("http://localhost:4000/mcp")).rejects.toThrow(/private/);
    const reservedPolicy = new ConnectorNetworkPolicy(["https://reserved.example.invalid"], false, async () => [{ address: "203.0.113.10", family: 4 }]);
    await expect(reservedPolicy.assertAllowed("https://reserved.example.invalid/mcp")).rejects.toThrow(/private|special/);
  });

  it("resolves only named environment credentials", () => {
    const resolver = new EnvironmentConnectorCredentialResolver({ CONNECTOR_SECRET_DEMO: "synthetic-test-secret" });
    expect(resolver.resolve(null)).toBeNull(); expect(resolver.resolve("CONNECTOR_SECRET_DEMO")).toBe("synthetic-test-secret");
    expect(() => resolver.resolve("UNSAFE_NAME")).toThrow(/invalid/); expect(() => resolver.resolve("CONNECTOR_SECRET_MISSING")).toThrow(/unavailable/);
  });

  it("bounds and compiles untrusted JSON schemas without dereferencing references", () => {
    const validate = compileUntrustedSchema({ type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false }, true);
    expect(validate({ city: "Toronto" })).toBe(true); expect(validate({ city: 42 })).toBe(false);
    expect(() => compileUntrustedSchema({ type: "object", properties: { value: { $ref: "https://example.invalid/schema" } } }, true)).toThrow(/reference/);
    expect(() => compileUntrustedSchema({ type: "string" }, true)).toThrow(/object type/);
  });

  it("uses stateless MCP discovery headers and validates discovered tools", async () => {
    const requests: { method: string; headers: Headers; body: Record<string, unknown> }[] = [];
    const fetcher: typeof fetch = async (_url, init) => {
      const body = JSON.parse(String(init?.body)) as { id: string; method: string };
      requests.push({ method: body.method, headers: new Headers(init?.headers), body });
      const result = body.method === "server/discover"
        ? { supportedVersions: [MCP_PROTOCOL_VERSION], capabilities: { tools: {} }, _meta: { "io.modelcontextprotocol/serverInfo": { name: "synthetic-server", version: "1" } } }
        : { tools: [{ name: "lookup", description: "untrusted synthetic description", inputSchema: { type: "object", properties: { key: { type: "string" } }, required: ["key"] } }] };
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }), { status: 200, headers: { "content-type": "application/json" } });
    };
    const transport = new McpHttpTransport(new ConnectorNetworkPolicy(["https://mcp.example.invalid"], false, publicLookup), 1000, 100_000, fetcher);
    const result = await transport.discover({ endpointUrl: "https://mcp.example.invalid/mcp", credential: "synthetic-test-secret" });
    expect(result.tools.map((tool) => tool.name)).toEqual(["lookup"]); expect(result.serverInfo).toMatchObject({ name: "synthetic-server" });
    expect(requests.map((request) => request.method)).toEqual(["server/discover", "tools/list"]);
    expect(requests[1]!.headers.get("mcp-protocol-version")).toBe(MCP_PROTOCOL_VERSION); expect(requests[1]!.headers.get("mcp-method")).toBe("tools/list");
    expect(JSON.stringify(requests)).not.toContain("synthetic-test-secret");
  });

  it("parses SSE tool results and classifies post-dispatch failures as uncertain", async () => {
    const policy = new ConnectorNetworkPolicy(["https://mcp.example.invalid"], false, publicLookup);
    const sseFetch: typeof fetch = async (_url, init) => { const body = JSON.parse(String(init?.body)) as { id: string }; return new Response(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { content: [{ type: "text", text: "synthetic result" }], isError: false } })}\n\n`, { headers: { "content-type": "text/event-stream" } }); };
    const result = await new McpHttpTransport(policy, 1000, 100_000, sseFetch).callTool({ endpointUrl: "https://mcp.example.invalid/mcp", credential: null }, "lookup", { key: "value" }, new AbortController().signal);
    expect(result.content[0]).toMatchObject({ type: "text", text: "synthetic result" });
    const failedFetch: typeof fetch = async () => new Response("unavailable", { status: 500 });
    const error = await new McpHttpTransport(policy, 1000, 100_000, failedFetch).callTool({ endpointUrl: "https://mcp.example.invalid/mcp", credential: null }, "lookup", {}, new AbortController().signal).catch((value: unknown) => value as { outcomeUnknown: boolean; message: string });
    expect(error.outcomeUnknown).toBe(true); expect(error.message).not.toContain("unavailable");
  });
});
