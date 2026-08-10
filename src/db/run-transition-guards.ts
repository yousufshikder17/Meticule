import type pg from "pg";

export async function hasUnresolvedToolReconciliation(client: pg.PoolClient, tenantId: string, runId: string): Promise<boolean> {
  const result = await client.query(
    `SELECT EXISTS (
       SELECT 1 FROM tool_executions
       WHERE tenant_id=$1 AND run_id=$2 AND status='unknown' AND reconciliation_status<>'resolved'
     ) AS unresolved`,
    [tenantId, runId],
  );
  return result.rows[0]?.unresolved === true;
}
