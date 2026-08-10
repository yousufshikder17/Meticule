import Ajv2020Module from "ajv/dist/2020.js";
import { z } from "zod";

const Ajv2020 = Ajv2020Module as unknown as new (options?: object) => { compile(schema: object): ((data: unknown) => boolean) & { errors?: unknown } };
export const MCP_PROTOCOL_VERSION = "2026-07-28";

const JsonSchemaSchema = z.record(z.string(), z.unknown());
export const McpToolSchema = z.object({
  name: z.string().trim().min(1).max(200).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/),
  title: z.string().max(500).optional(), description: z.string().max(10_000).optional(),
  inputSchema: JsonSchemaSchema,
  outputSchema: JsonSchemaSchema.optional(),
  annotations: z.object({ title: z.string().max(500).optional(), readOnlyHint: z.boolean().optional(), destructiveHint: z.boolean().optional(), idempotentHint: z.boolean().optional(), openWorldHint: z.boolean().optional() }).loose().optional(),
  execution: z.object({ taskSupport: z.enum(["forbidden", "optional", "required"]).optional() }).loose().optional(),
}).loose();
export const DiscoverResultSchema = z.object({
  supportedVersions: z.array(z.string()).min(1).max(20), capabilities: z.record(z.string(), z.unknown()),
  instructions: z.string().max(10_000).optional(), serverInfo: z.object({ name: z.string(), version: z.string() }).loose().optional(),
  _meta: z.record(z.string(), z.unknown()).optional(),
}).loose();
export const ListToolsResultSchema = z.object({ tools: z.array(McpToolSchema).max(100), nextCursor: z.string().max(10_000).optional(), ttlMs: z.number().nonnegative().optional(), cacheScope: z.string().optional() }).loose();
const ContentBlockSchema = z.object({ type: z.string().min(1).max(100) }).loose();
export const CallToolResultSchema = z.object({ content: z.array(ContentBlockSchema).max(100), structuredContent: z.unknown().optional(), isError: z.boolean().default(false) }).loose();
export const JsonRpcResponseSchema = z.object({ jsonrpc: z.literal("2.0"), id: z.union([z.string(), z.number()]), result: z.unknown().optional(), error: z.object({ code: z.number().int(), message: z.string().max(10_000), data: z.unknown().optional() }).optional() }).refine((value) => (value.result !== undefined) !== (value.error !== undefined), "JSON-RPC response must contain exactly one of result or error");

export class McpSchemaError extends Error {}

export function compileUntrustedSchema(raw: unknown, input: boolean): (value: unknown) => boolean {
  const schema = JsonSchemaSchema.parse(raw); const serialized = JSON.stringify(schema);
  if (serialized.length > 50_000) throw new McpSchemaError("MCP tool schema is too large");
  let nodes = 0;
  const inspect = (value: unknown, depth: number): void => {
    if (depth > 12 || ++nodes > 500) throw new McpSchemaError("MCP tool schema exceeds complexity limits");
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) { for (const entry of value) inspect(entry, depth + 1); return; }
    const record = value as Record<string, unknown>;
    if ("$ref" in record || "$dynamicRef" in record) throw new McpSchemaError("External and reference-based MCP schemas are unsupported");
    for (const child of Object.values(record)) inspect(child, depth + 1);
  };
  inspect(schema, 0);
  if (input && schema.type !== "object") throw new McpSchemaError("MCP tool input schema must have object type");
  try { return new Ajv2020({ strict: false, allErrors: true, validateFormats: false }).compile(schema); }
  catch (error) { throw new McpSchemaError(`MCP tool schema is invalid: ${error instanceof Error ? error.message : String(error)}`); }
}
