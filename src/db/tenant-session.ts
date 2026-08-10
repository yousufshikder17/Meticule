import type pg from "pg";

let savepointSequence = 0;
const emptyResult = (): pg.QueryResult => ({ command: "", rowCount: null, oid: 0, rows: [], fields: [] });

function scopedPool(client: pg.PoolClient): pg.Pool {
  return {
    query: client.query.bind(client),
    connect: async () => {
      const savepoint = `api_nested_${++savepointSequence}`;
      let active = false;
      return {
        query: async (query: unknown, values?: unknown[]) => {
          if (typeof query === "string") {
            const command = query.trim().toUpperCase();
            if (command === "BEGIN") { await client.query(`SAVEPOINT ${savepoint}`); active = true; return emptyResult(); }
            if (command === "COMMIT") { if (active) await client.query(`RELEASE SAVEPOINT ${savepoint}`); active = false; return emptyResult(); }
            if (command === "ROLLBACK") { if (active) { await client.query(`ROLLBACK TO SAVEPOINT ${savepoint}`); await client.query(`RELEASE SAVEPOINT ${savepoint}`); } active = false; return emptyResult(); }
          }
          return client.query(query as string, values);
        },
        release: () => undefined,
      } as unknown as pg.PoolClient;
    },
  } as unknown as pg.Pool;
}

export async function withTenantSession<T>(pool: pg.Pool, tenantId: string, work: (database: pg.Pool) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("SET LOCAL ROLE durable_agent_api");
    await client.query("SELECT set_config('app.tenant_id',$1,true)", [tenantId]);
    const result = await work(scopedPool(client));
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK").catch(() => undefined);
    throw error;
  } finally {
    client.release();
  }
}
