import { z } from "zod";
import type pg from "pg";
import { EnvironmentConnectorCredentialResolver } from "./credential-resolver.js";
import { ConnectorNetworkPolicy } from "./network-policy.js";
import { McpHttpTransport } from "./mcp-transport.js";
import { ConnectorService } from "./connector-service.js";

const SettingsSchema = z.object({
  CONNECTOR_EGRESS_ALLOWLIST: z.string().min(1),
  CONNECTOR_ALLOW_PRIVATE_NETWORK: z.enum(["true", "false"]).default("false"),
  CONNECTOR_TIMEOUT_MS: z.coerce.number().int().min(100).max(600_000).default(60_000),
  CONNECTOR_MAX_RESPONSE_BYTES: z.coerce.number().int().min(1_024).max(10_000_000).default(1_000_000),
});

export function createConnectorService(pool: pg.Pool, env: NodeJS.ProcessEnv = process.env): ConnectorService | null {
  if (!env.CONNECTOR_EGRESS_ALLOWLIST?.trim()) return null;
  const settings = SettingsSchema.parse(env); const origins = settings.CONNECTOR_EGRESS_ALLOWLIST.split(",").map((entry) => entry.trim()).filter(Boolean);
  const policy = new ConnectorNetworkPolicy(origins, settings.CONNECTOR_ALLOW_PRIVATE_NETWORK === "true");
  return new ConnectorService(pool, new McpHttpTransport(policy, settings.CONNECTOR_TIMEOUT_MS, settings.CONNECTOR_MAX_RESPONSE_BYTES), new EnvironmentConnectorCredentialResolver(env));
}
