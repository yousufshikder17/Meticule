import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { getPool, closePool } from "./pool.js";

async function migrate(): Promise<void> {
  const directory = resolve(process.cwd(), "migrations");
  const files = (await readdir(directory)).filter((file) => file.endsWith(".sql")).sort();
  const pool = getPool();
  const client = await pool.connect();
  try {
  for (const file of files) {
    // The baseline migration sets an empty search_path on its session.
    await client.query("SET search_path TO public");
    const version = file.replace(/\.sql$/, "");
    const exists = await client.query<{ exists: boolean }>(
      "SELECT to_regclass('public.schema_migrations') IS NOT NULL AS exists",
    );
    if (exists.rows[0]?.exists) {
      const applied = await client.query("SELECT 1 FROM schema_migrations WHERE version = $1", [version]);
      if (applied.rowCount) continue;
    }
    await client.query(await readFile(resolve(directory, file), "utf8"));
    process.stdout.write(`applied ${file}\n`);
  }
  } finally { client.release(); }
}

migrate().finally(closePool).catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
