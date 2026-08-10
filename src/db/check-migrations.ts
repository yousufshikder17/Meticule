import { getPool, closePool } from "./pool.js";
import { migrationReadiness } from "./migration-state.js";

const state = await migrationReadiness(getPool());
if (!state.ready) { console.error(JSON.stringify({ error: "migrations_pending", missing: state.missing })); process.exitCode = 1; }
else process.stdout.write(JSON.stringify({ status: "ok", applied: state.applied, expected: state.expected }) + "\n");
await closePool();
