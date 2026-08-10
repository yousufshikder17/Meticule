import type pg from "pg";
import type { Principal } from "../db/types.js";
import { hasUnresolvedToolReconciliation } from "../db/run-transition-guards.js";
import { ToolRegistry } from "./types.js";

interface ClaimedReconciliation { id: string; tenantId: string; runId: string; stepId: string; toolName: string; idempotencyKey: string; arguments: unknown; requesterId: string }

export class ReconciliationService {
  constructor(private readonly pool: pg.Pool, private readonly tools: ToolRegistry) {}

  async reconcileNext(workerId: string, leaseSeconds = 30): Promise<boolean> {
    const claimed = await this.claim(workerId, leaseSeconds);
    if (!claimed) return false;
    const tool = this.tools.get(claimed.toolName);
    if (!tool.reconcile) {
      await this.resolve(workerId, claimed, { status: "manual", reason: "Tool has no reconciliation contract" });
      return true;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), tool.timeoutMs);
    try {
      const principal: Principal = { tenantId: claimed.tenantId, userId: claimed.requesterId, roles: tool.authorization.requiredRoles };
      const result = await tool.reconcile(claimed.arguments, { pool: this.pool, principal, runId: claimed.runId, idempotencyKey: claimed.idempotencyKey, signal: controller.signal });
      if (result.status === "succeeded") tool.outputSchema.parse(result.output);
      await this.resolve(workerId, claimed, result);
      return true;
    } catch (error) {
      await this.resolve(workerId, claimed, { status: "manual", reason: error instanceof Error ? error.message : String(error) });
      throw error;
    } finally { clearTimeout(timer); }
  }

  private async claim(workerId: string, leaseSeconds: number): Promise<ClaimedReconciliation | null> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const selected = await client.query(
        `SELECT te.*,r.created_by FROM tool_executions te JOIN runs r ON r.tenant_id=te.tenant_id AND r.id=te.run_id
         WHERE te.status='unknown' AND (te.reconciliation_status='pending' OR
           (te.reconciliation_status='running' AND te.reconciliation_expires_at<=now()))
         ORDER BY te.updated_at,te.id FOR UPDATE OF te SKIP LOCKED LIMIT 1`,
      );
      if (!selected.rowCount) { await client.query("COMMIT"); return null; }
      const row = selected.rows[0];
      await client.query(
        `UPDATE tool_executions SET reconciliation_status='running',reconciliation_owner=$1,
         reconciliation_expires_at=now()+($2*interval '1 second'),updated_at=now() WHERE id=$3`,
        [workerId, leaseSeconds, row.id],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,'tool.reconciliation_claimed',$4)", [row.tenant_id, row.run_id, workerId, JSON.stringify({ executionId: row.id, tool: row.tool_name })]);
      await client.query("COMMIT");
      return { id: row.id, tenantId: row.tenant_id, runId: row.run_id, stepId: row.step_id, toolName: row.tool_name, idempotencyKey: row.idempotency_key, arguments: row.validated_arguments, requesterId: row.created_by };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async resolve(workerId: string, claimed: ClaimedReconciliation, result: { status: "succeeded"; output: unknown } | { status: "failed"; error: unknown } | { status: "pending" } | { status: "manual"; reason: string }): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const locked = await client.query(
        `SELECT te.reconciliation_owner,te.reconciliation_expires_at,r.status AS run_status,r.cancellation_requested_at
         FROM tool_executions te JOIN runs r ON r.tenant_id=te.tenant_id AND r.id=te.run_id
         WHERE te.tenant_id=$1 AND te.id=$2 FOR UPDATE OF te,r`, [claimed.tenantId, claimed.id],
      );
      const row = locked.rows[0];
      if (!row || row.reconciliation_owner !== workerId || new Date(row.reconciliation_expires_at) <= new Date()) throw new Error("Reconciliation lease was lost");
      if (result.status === "pending") {
        await client.query("UPDATE tool_executions SET reconciliation_status='pending',reconciliation_owner=NULL,reconciliation_expires_at=NULL,updated_at=now() WHERE id=$1", [claimed.id]);
      } else if (result.status === "manual") {
        await client.query("UPDATE tool_executions SET reconciliation_status='manual',reconciliation_owner=NULL,reconciliation_expires_at=NULL,error_details=$1,updated_at=now() WHERE id=$2", [JSON.stringify({ code: "manual_reconciliation", reason: result.reason }), claimed.id]);
      } else {
        const status = result.status === "succeeded" ? "succeeded" : "failed";
        const output = result.status === "succeeded" ? result.output : null;
        const error = result.status === "failed" ? result.error : null;
        await client.query("UPDATE tool_executions SET status=$1,output=$2,error_details=$3,reconciliation_status='resolved',reconciliation_owner=NULL,reconciliation_expires_at=NULL,reconciled_at=now(),updated_at=now() WHERE id=$4", [status, output === null ? null : JSON.stringify(output), error === null ? null : JSON.stringify(error), claimed.id]);
        await client.query("UPDATE steps SET status=$1,output=$2,error_details=$3,finished_at=now() WHERE id=$4", [status, output === null ? null : JSON.stringify(output), error === null ? null : JSON.stringify(error), claimed.stepId]);
        if (!row.cancellation_requested_at && row.run_status === "paused" && !await hasUnresolvedToolReconciliation(client, claimed.tenantId, claimed.runId)) {
          await client.query("UPDATE runs SET status='queued',version=version+1,updated_at=now(),error_details=NULL WHERE tenant_id=$1 AND id=$2", [claimed.tenantId, claimed.runId]);
        }
      }
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,$4,$5)", [claimed.tenantId, claimed.runId, workerId, `tool.reconciliation_${result.status}`, JSON.stringify({ executionId: claimed.id, tool: claimed.toolName })]);
      await client.query("COMMIT");
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }
}
