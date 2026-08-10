import { readdir } from "node:fs/promises";
import { resolve } from "node:path";
import type pg from "pg";

export async function expectedMigrationVersions(directory = resolve(process.cwd(), "migrations")): Promise<string[]> {
  return (await readdir(directory)).filter((file) => file.endsWith(".sql")).sort().map((file) => file.replace(/\.sql$/, ""));
}

export async function migrationReadiness(database: Pick<pg.Pool, "query">, directory?: string): Promise<{ ready: boolean; expected: number; applied: number; missing: string[] }> {
  const expected = await expectedMigrationVersions(directory);
  const exists = await database.query<{ exists: boolean }>("SELECT to_regclass('public.schema_migrations') IS NOT NULL AS exists");
  if (!exists.rows[0]?.exists) return { ready: false, expected: expected.length, applied: 0, missing: expected };
  const result = await database.query<{ version: string }>("SELECT version FROM schema_migrations ORDER BY version"); const applied = new Set(result.rows.map((row) => row.version));
  const missing = expected.filter((version) => !applied.has(version)); return { ready: missing.length === 0, expected: expected.length, applied: applied.size, missing };
}
