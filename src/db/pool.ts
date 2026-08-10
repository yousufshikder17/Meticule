import pg from "pg";
import { loadConfig } from "../config.js";

const { Pool } = pg;
let singleton: pg.Pool | undefined;

export function getPool(): pg.Pool {
  if (!singleton) { const config = loadConfig(); singleton = new Pool({ connectionString: config.DATABASE_URL, max: config.DB_POOL_MAX, idleTimeoutMillis: config.DB_POOL_IDLE_TIMEOUT_MS, connectionTimeoutMillis: config.DB_CONNECT_TIMEOUT_MS }); }
  return singleton;
}

export async function closePool(): Promise<void> {
  if (singleton) await singleton.end();
  singleton = undefined;
}
