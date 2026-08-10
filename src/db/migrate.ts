import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { getPool, closePool } from "./pool.js";

async function migrate(): Promise<void> {
  const directory = resolve(process.cwd(), "migrations");
  const files = (await readdir(directory)).filter((file) => file.endsWith(".sql")).sort();
  const pool = getPool();
  for (const file of files) {
    const version = file.replace(/\.sql$/, "");
    const exists = await pool.query<{ exists: boolean }>(
      "SELECT to_regclass('public.schema_migrations') IS NOT NULL AS exists",
    );
    if (exists.rows[0]?.exists) {
      const applied = await pool.query("SELECT 1 FROM schema_migrations WHERE version = $1", [version]);
      if (applied.rowCount) continue;
    }
    await pool.query(await readFile(resolve(directory, file), "utf8"));
    process.stdout.write(`applied ${file}\n`);
  }
}

migrate().finally(closePool).catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
