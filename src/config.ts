import { z } from "zod";

const ConfigSchema = z.object({
  DATABASE_URL: z.string().min(1),
  PORT: z.coerce.number().int().positive().default(3000),
  WORKER_ID: z.string().min(1).default(`worker-${process.pid}`),
  LEASE_SECONDS: z.coerce.number().int().min(5).max(300).default(30),
  WORKER_POLL_MS: z.coerce.number().int().min(100).max(60_000).default(1000),
  WORKER_ROLES: z.string().default(""),
  DB_POOL_MAX: z.coerce.number().int().min(1).max(200).default(20),
  DB_POOL_IDLE_TIMEOUT_MS: z.coerce.number().int().min(1_000).max(600_000).default(30_000),
  DB_CONNECT_TIMEOUT_MS: z.coerce.number().int().min(1_000).max(120_000).default(10_000),
  SHUTDOWN_GRACE_MS: z.coerce.number().int().min(1_000).max(300_000).default(30_000),
});

const AuthConfigSchema = z.discriminatedUnion("AUTH_MODE", [
  z.object({ AUTH_MODE: z.literal("jwt").default("jwt"), JWT_SECRET: z.string().min(32), JWT_ISSUER: z.string().min(1).default("durable-agent-platform"), JWT_AUDIENCE: z.string().min(1).default("durable-agent-api") }),
  z.object({ AUTH_MODE: z.literal("development_headers") }),
]);

export function loadConfig(env: NodeJS.ProcessEnv = process.env) {
  return ConfigSchema.parse(env);
}

export function loadAuthConfig(env: NodeJS.ProcessEnv = process.env) {
  const config = AuthConfigSchema.parse({ AUTH_MODE: env.AUTH_MODE ?? "jwt", JWT_SECRET: env.JWT_SECRET, JWT_ISSUER: env.JWT_ISSUER, JWT_AUDIENCE: env.JWT_AUDIENCE });
  if (config.AUTH_MODE === "development_headers" && env.NODE_ENV === "production") throw new Error("Development header authentication is prohibited in production");
  return config;
}
