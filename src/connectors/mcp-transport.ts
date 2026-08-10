import { z } from "zod";
import { ConnectorNetworkPolicy } from "./network-policy.js";
import { CallToolResultSchema, DiscoverResultSchema, JsonRpcResponseSchema, ListToolsResultSchema, MCP_PROTOCOL_VERSION, McpToolSchema } from "./mcp-schemas.js";

type McpTool = z.infer<typeof McpToolSchema>;
export class McpTransportError extends Error {
  constructor(public readonly code: string, message: string, public readonly outcomeUnknown = false, options?: ErrorOptions) { super(message, options); this.name = "McpTransportError"; }
}
export interface McpEndpoint { endpointUrl: string; credential: string | null }
export interface McpDiscovery { supportedVersions: string[]; capabilities: Record<string, unknown>; serverInfo: unknown; instructions: string | null; tools: McpTool[] }
export interface McpToolResult { content: Record<string, unknown>[]; structuredContent?: unknown; isError: boolean }

export class McpHttpTransport {
  constructor(private readonly network: ConnectorNetworkPolicy, private readonly timeoutMs = 60_000, private readonly maximumResponseBytes = 1_000_000, private readonly fetcher: typeof fetch = fetch) {}

  async validateEndpoint(endpointUrl: string): Promise<void> { await this.network.assertAllowed(endpointUrl); }

  async discover(endpoint: McpEndpoint, signal = new AbortController().signal): Promise<McpDiscovery> {
    const discovered = DiscoverResultSchema.parse(await this.request(endpoint, "server/discover", {}, signal, false));
    if (!discovered.supportedVersions.includes(MCP_PROTOCOL_VERSION)) throw new McpTransportError("unsupported_protocol", `MCP server does not support ${MCP_PROTOCOL_VERSION}`);
    const tools: McpTool[] = []; let cursor: string | undefined;
    for (let page = 0; page < 10; page += 1) {
      const result = ListToolsResultSchema.parse(await this.request(endpoint, "tools/list", cursor ? { cursor } : {}, signal, false));
      tools.push(...result.tools); if (tools.length > 100) throw new McpTransportError("too_many_tools", "MCP server exposed more than 100 tools");
      cursor = result.nextCursor; if (!cursor) break; if (page === 9) throw new McpTransportError("too_many_pages", "MCP tools/list pagination exceeded 10 pages");
    }
    const meta = discovered._meta ?? {};
    return { supportedVersions: discovered.supportedVersions, capabilities: discovered.capabilities, serverInfo: discovered.serverInfo ?? meta["io.modelcontextprotocol/serverInfo"] ?? null, instructions: discovered.instructions ?? null, tools };
  }

  async callTool(endpoint: McpEndpoint, toolName: string, args: Record<string, unknown>, signal: AbortSignal): Promise<McpToolResult> {
    return CallToolResultSchema.parse(await this.request(endpoint, "tools/call", { name: toolName, arguments: args }, signal, true, toolName));
  }

  private async request(endpoint: McpEndpoint, method: string, params: Record<string, unknown>, signal: AbortSignal, sideEffecting: boolean, name?: string): Promise<unknown> {
    if (signal.aborted) throw new McpTransportError("cancelled", "MCP request was cancelled before transport");
    const url = await this.network.assertAllowed(endpoint.endpointUrl);
    if (signal.aborted) throw new McpTransportError("cancelled", "MCP request was cancelled before transport");
    const id = crypto.randomUUID();
    const body = { jsonrpc: "2.0", id, method, params: { ...params, _meta: { "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION, "io.modelcontextprotocol/clientInfo": { name: "durable-agent-platform", version: "0.1.0" }, "io.modelcontextprotocol/clientCapabilities": {} } } };
    const headers: Record<string, string> = { "content-type": "application/json", accept: "application/json, text/event-stream", "MCP-Protocol-Version": MCP_PROTOCOL_VERSION, "Mcp-Method": method };
    if (name) headers["Mcp-Name"] = name;
    if (endpoint.credential) headers.authorization = `Bearer ${endpoint.credential}`;
    const controller = new AbortController(); const abort = () => controller.abort(signal.reason); signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(new Error("MCP transport timeout")), this.timeoutMs); let sent = false;
    try {
      sent = true;
      const response = await this.fetcher(url, { method: "POST", headers, body: JSON.stringify(body), redirect: "manual", signal: controller.signal });
      if (response.status >= 300 && response.status < 400) throw new McpTransportError("redirect_rejected", "MCP redirects are prohibited", sideEffecting);
      if (!response.ok) throw new McpTransportError("http_error", `MCP request failed with HTTP ${response.status}`, sideEffecting && response.status >= 500);
      const text = await this.readBounded(response);
      const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
      const candidate = contentType.includes("text/event-stream") ? this.parseSse(text, id) : this.parseJson(text);
      const rpc = JsonRpcResponseSchema.parse(candidate); if (String(rpc.id) !== id) throw new McpTransportError("mismatched_response", "MCP response ID does not match request", sideEffecting);
      if (rpc.error) throw new McpTransportError("protocol_error", `MCP protocol error ${rpc.error.code}: ${rpc.error.message}`, sideEffecting);
      return rpc.result;
    } catch (error) {
      if (error instanceof McpTransportError) {
        if (sideEffecting && sent && !error.outcomeUnknown) throw new McpTransportError(error.code, error.message, true, { cause: error });
        throw error;
      }
      const aborted = controller.signal.aborted;
      throw new McpTransportError(aborted ? (signal.aborted ? "cancelled" : "timeout") : "transport_failure", aborted ? (signal.aborted ? "MCP request was cancelled" : "MCP request timed out") : "MCP transport failed", sideEffecting && sent, { cause: error });
    } finally { clearTimeout(timer); signal.removeEventListener("abort", abort); }
  }

  private async readBounded(response: Response): Promise<string> {
    if (!response.body) return ""; const reader = response.body.getReader(); const decoder = new TextDecoder(); let total = 0; let output = "";
    while (true) { const chunk = await reader.read(); if (chunk.done) break; total += chunk.value.byteLength; if (total > this.maximumResponseBytes) { await reader.cancel(); throw new McpTransportError("response_too_large", "MCP response exceeded configured size limit"); } output += decoder.decode(chunk.value, { stream: true }); }
    return output + decoder.decode();
  }
  private parseJson(text: string): unknown { try { return JSON.parse(text); } catch (error) { throw new McpTransportError("malformed_response", "MCP response was not valid JSON", false, { cause: error }); } }
  private parseSse(text: string, id: string): unknown {
    for (const event of text.split(/\r?\n\r?\n/)) {
      const data = event.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
      if (!data) continue; const value = this.parseJson(data);
      if (value && typeof value === "object" && "id" in value && String((value as { id: unknown }).id) === id) return value;
    }
    throw new McpTransportError("malformed_response", "MCP event stream contained no matching response");
  }
}
