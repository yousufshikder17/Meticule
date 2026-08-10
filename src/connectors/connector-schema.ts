import { z } from "zod";

const NameSchema = z.string().trim().min(1).max(200).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const UniqueNamesSchema = z.array(NameSchema).max(100).refine((values) => new Set(values).size === values.length, "Names must be unique");

export const CreateConnectorSchema = z.object({
  name: z.string().trim().min(1).max(200),
  transport: z.literal("mcp_streamable_http").default("mcp_streamable_http"),
  endpointUrl: z.url().max(2_000),
  credentialRef: z.string().regex(/^CONNECTOR_SECRET_[A-Z0-9_]+$/).nullable().default(null),
  allowedTools: UniqueNamesSchema.min(1),
  requiredRoles: UniqueNamesSchema.max(20).default([]),
  rateLimitPerMinute: z.int().min(1).max(10_000).default(60),
});

export const RevokeConnectorSchema = z.object({ reason: z.string().trim().min(1).max(2_000) });
export const McpCallInputSchema = z.object({
  connectorId: z.uuid(), toolName: NameSchema,
  arguments: z.record(z.string().max(200), z.unknown()).refine((value) => Object.keys(value).length <= 100, "MCP arguments contain too many keys")
    .refine((value) => JSON.stringify(value).length <= 100_000, "MCP arguments are too large"),
});

const McpContentBlockSchema = z.object({ type: z.string().min(1).max(100) }).loose();
export const McpCallOutputSchema = z.object({
  content: z.array(McpContentBlockSchema).max(100),
  structuredContent: z.unknown().optional(),
  isError: z.boolean().default(false),
  connectorId: z.uuid(), toolName: NameSchema,
});

export type CreateConnectorInput = z.infer<typeof CreateConnectorSchema>;
export type McpCallInput = z.infer<typeof McpCallInputSchema>;
