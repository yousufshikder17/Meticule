import { serve } from "@hono/node-server";
import { loadAuthConfig, loadConfig } from "../config.js";
import { getPool, closePool } from "../db/pool.js";
import { createApp } from "./app.js";
import { DevelopmentHeaderAuthenticator, JwtAuthenticator } from "../auth/authentication.js";
import { createRetrievalService } from "../retrieval/retrieval-configuration.js";
import { createConnectorService } from "../connectors/connector-configuration.js";
import { createProviderRegistry } from "../models/provider-configuration.js";
import { logger } from "../observability/logger.js";
import { migrationReadiness } from "../db/migration-state.js";
import { closeServer } from "../deployment/graceful-shutdown.js";

const config = loadConfig();
const pool = getPool();
const auth = loadAuthConfig();
const authenticator = auth.AUTH_MODE === "development_headers" ? new DevelopmentHeaderAuthenticator() : new JwtAuthenticator(pool, auth.JWT_SECRET, auth.JWT_ISSUER, auth.JWT_AUDIENCE);
const retrieval = createRetrievalService(pool);
const connectors = createConnectorService(pool);
const providers = createProviderRegistry();
let draining = false; let shutdownPromise: Promise<void> | null = null;
const readiness = { isDraining: () => draining, checkMigrations: () => migrationReadiness(pool) };
const server = serve({ fetch: createApp(pool, authenticator, retrieval, connectors, providers, logger, readiness).fetch, port: config.PORT }, ({ port }) => {
  logger.log("info", "api.started", { port });
});

async function shutdown(): Promise<void> {
  if (shutdownPromise) return shutdownPromise; draining = true;
  shutdownPromise = (async () => { const result = await closeServer(server, config.SHUTDOWN_GRACE_MS); logger.log(result === "forced" ? "warn" : "info", "api.stopped", { shutdown: result }); await closePool(); })();
  return shutdownPromise;
}
process.once("SIGINT", () => { void shutdown(); });
process.once("SIGTERM", () => { void shutdown(); });
