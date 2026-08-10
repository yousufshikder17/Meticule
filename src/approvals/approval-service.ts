import type pg from "pg";
import type { Principal } from "../db/types.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { ToolAuthorizationError, ToolExecutionConflictError, ToolRegistry, type ToolDefinition } from "../tools/types.js";
import type { ExecuteToolCommand } from "../tools/executor.js";

type ApprovalDecision = "approved" | "rejected";
type Row = Record<string, unknown>;

interface ApprovalPolicyResolution {
  required: boolean;
  requiredRole: string;
  separationOfDuties: boolean;
  riskExplanation: string;
}

export class ApprovalService {
  constructor(private readonly pool: pg.Pool, private readonly tools: ToolRegistry) {}

  async proposeIfRequired(command: ExecuteToolCommand): Promise<boolean> {
    const tool = this.tools.get(command.toolName);
    const input = tool.inputSchema.parse(command.arguments);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query(
        `SELECT r.*,r.agent_configuration_snapshot->'allowedTools' AS allowed_tools,r.agent_configuration_snapshot->'approvalPolicy' AS approval_policy FROM runs r
         WHERE r.tenant_id=$1 AND r.id=$2 FOR UPDATE OF r`,
        [command.principal.tenantId, command.runId],
      );
      if (!result.rowCount) throw new ToolAuthorizationError("Run not found in tenant");
      const run = result.rows[0];
      if (!(run.allowed_tools as string[]).includes(tool.name)) throw new ToolAuthorizationError(`Tool is not allowed by agent: ${tool.name}`);
      for (const role of tool.authorization.requiredRoles) if (!command.principal.roles.includes(role)) throw new ToolAuthorizationError(`Missing role: ${role}`);
      if (run.cancellation_requested_at) throw new ToolExecutionConflictError("Cancellation was requested");
      if (run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at) <= new Date()) throw new ToolExecutionConflictError("Worker does not own a running lease");
      await tool.authorize?.(input, { pool: this.pool, principal: command.principal, runId: command.runId, idempotencyKey: command.idempotencyKey, signal: new AbortController().signal });
      const policy = this.resolvePolicy(tool, (run.approval_policy ?? {}) as Record<string, unknown>);
      if (!policy.required) { await client.query("COMMIT"); return false; }
      const argumentHash = canonicalJsonHash(input);
      const operationHash = canonicalJsonHash({ tenantId: command.principal.tenantId, runId: command.runId, toolName: tool.name, arguments: input, idempotencyKey: command.idempotencyKey });
      const duplicate = await client.query("SELECT id,tool_name,canonical_argument_hash,run_id FROM approvals WHERE tenant_id=$1 AND idempotency_key=$2", [command.principal.tenantId, command.idempotencyKey]);
      if (duplicate.rowCount) {
        throw new ConflictError("Approval idempotency key identifies a different action");
      }
      const sequence = Number(run.current_step) + 1;
      const step = await client.query(
        `INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input)
         VALUES($1,$2,$3,'tool','pending',$4,0,$5) RETURNING id`,
        [command.principal.tenantId, command.runId, sequence, command.idempotencyKey, JSON.stringify({ tool: tool.name, arguments: input, approvalRequired: true })],
      );
      await client.query(
        `INSERT INTO tool_executions(tenant_id,run_id,step_id,tool_name,idempotency_key,status,validated_arguments,operation_hash,retry_safety)
         VALUES($1,$2,$3,$4,$5,'pending',$6,$7,$8)`,
        [command.principal.tenantId, command.runId, step.rows[0].id, tool.name, command.idempotencyKey, JSON.stringify(input), operationHash, tool.retrySafety],
      );
      const approval = await client.query(
        `INSERT INTO approvals(tenant_id,run_id,step_id,tool_name,validated_arguments,canonical_argument_hash,idempotency_key,
          risk_explanation,requester_id,required_approver_role,requested_roles,tool_risk,separation_of_duties)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING id`,
        [command.principal.tenantId, command.runId, step.rows[0].id, tool.name, JSON.stringify(input), argumentHash,
         command.idempotencyKey, policy.riskExplanation, run.created_by, policy.requiredRole,
         JSON.stringify([policy.requiredRole]), tool.riskLevel, policy.separationOfDuties],
      );
      await client.query(
        `UPDATE runs SET status='waiting_for_approval',current_step=$1,lease_owner=NULL,lease_expires_at=NULL,
         version=version+1,updated_at=now() WHERE tenant_id=$2 AND id=$3`,
        [sequence, command.principal.tenantId, command.runId],
      );
      await client.query("DELETE FROM worker_leases WHERE run_id=$1", [command.runId]);
      const details = JSON.stringify({ approvalId: approval.rows[0].id, stepId: step.rows[0].id, tool: tool.name, risk: tool.riskLevel, argumentHash, idempotencyKey: command.idempotencyKey });
      await client.query(
        `INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES
         ($1,$2,'worker',$3,'approval.requested',$4),
         ($1,$2,'worker',$3,'run.transitioned',$5)`,
        [command.principal.tenantId, command.runId, command.workerId, details, JSON.stringify({ from: "running", to: "waiting_for_approval" })],
      );
      await client.query("COMMIT");
      return true;
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async get(principal: Principal, approvalId: string): Promise<Row> {
    const result = await this.pool.query("SELECT * FROM approvals WHERE tenant_id=$1 AND id=$2", [principal.tenantId, approvalId]);
    if (!result.rowCount) throw new NotFoundError("Approval not found");
    const approval = result.rows[0];
    if (approval.requester_id !== principal.userId && !principal.roles.includes(approval.required_approver_role as string)) throw new AuthorizationError("Approval inspection requires requester or approver access");
    return approval as Row;
  }

  async listForRun(principal: Principal, runId: string): Promise<Row[]> {
    const run = await this.pool.query("SELECT 1 FROM runs WHERE tenant_id=$1 AND id=$2", [principal.tenantId, runId]);
    if (!run.rowCount) throw new NotFoundError("Run not found");
    return (await this.pool.query(
      `SELECT * FROM approvals WHERE tenant_id=$1 AND run_id=$2
       AND (requester_id=$3 OR required_approver_role=ANY($4::text[])) ORDER BY created_at`,
      [principal.tenantId, runId, principal.userId, principal.roles],
    )).rows as Row[];
  }

  async decide(principal: Principal, approvalId: string, decision: ApprovalDecision, comment?: string): Promise<Row> {
    const client = await this.pool.connect();
    let denied: string | null = null;
    let output: Row | null = null;
    try {
      await client.query("BEGIN");
      const result = await client.query(
        `SELECT ap.*,r.status AS run_status,r.cancellation_requested_at FROM approvals ap
         JOIN runs r ON r.tenant_id=ap.tenant_id AND r.id=ap.run_id
         WHERE ap.tenant_id=$1 AND ap.id=$2 FOR UPDATE OF ap,r`,
        [principal.tenantId, approvalId],
      );
      if (!result.rowCount) throw new NotFoundError("Approval not found");
      const approval = result.rows[0];
      if (!principal.roles.includes(approval.required_approver_role as string)) denied = "required_role_missing";
      else if (approval.separation_of_duties && approval.requester_id === principal.userId) denied = "requester_self_approval";
      if (denied) {
        await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'user',$3,'approval.decision_denied',$4)", [principal.tenantId, approval.run_id, principal.userId, JSON.stringify({ approvalId, reason: denied })]);
        await client.query("COMMIT");
      } else if (approval.decision !== "pending") {
        if (approval.decision !== decision) throw new ConflictError(`Approval was already ${approval.decision}`);
        output = approval as Row;
        await client.query("COMMIT");
      } else {
        if (approval.cancellation_requested_at || approval.run_status !== "waiting_for_approval") throw new ConflictError("Run is no longer waiting for this approval");
        const updated = await client.query(
          "UPDATE approvals SET decision=$1,approver_id=$2,decided_at=now(),comment=$3 WHERE tenant_id=$4 AND id=$5 AND decision='pending' RETURNING *",
          [decision, principal.userId, comment ?? null, principal.tenantId, approvalId],
        );
        if (decision === "rejected") {
          const rejection = { approved: false, reason: "rejected", approvalId, comment: comment ?? null };
          await client.query("UPDATE tool_executions SET status='cancelled',error_details=$1,updated_at=now() WHERE tenant_id=$2 AND step_id=$3 AND status='pending'", [JSON.stringify(rejection), principal.tenantId, approval.step_id]);
          await client.query("UPDATE steps SET status='failed',output=$1,error_details=$1,finished_at=now() WHERE tenant_id=$2 AND id=$3 AND status='pending'", [JSON.stringify(rejection), principal.tenantId, approval.step_id]);
        }
        await client.query("UPDATE runs SET status='queued',version=version+1,updated_at=now() WHERE tenant_id=$1 AND id=$2 AND status='waiting_for_approval'", [principal.tenantId, approval.run_id]);
        await client.query(
          `INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES
           ($1,$2,'user',$3,$4,$5),($1,$2,'system','approval-service','run.requeued',$6)`,
          [principal.tenantId, approval.run_id, principal.userId, decision === "approved" ? "approval.granted" : "approval.rejected",
           JSON.stringify({ approvalId, commentProvided: Boolean(comment) }), JSON.stringify({ approvalId, decision })],
        );
        output = updated.rows[0] as Row;
        await client.query("COMMIT");
      }
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
    if (denied) throw new AuthorizationError(denied === "requester_self_approval" ? "Requester cannot approve this action" : "Required approver role is missing");
    return output as Row;
  }

  private resolvePolicy(tool: ToolDefinition, policy: Record<string, unknown>): ApprovalPolicyResolution {
    const configured = policy[tool.name];
    const required = tool.approvalRequirement === "always" || (tool.approvalRequirement === "policy" && Boolean(configured));
    const object = configured && typeof configured === "object" && !Array.isArray(configured) ? configured as Record<string, unknown> : {};
    return {
      required,
      requiredRole: typeof object.requiredApproverRole === "string" ? object.requiredApproverRole : "approval_reviewer",
      separationOfDuties: typeof object.separationOfDuties === "boolean" ? object.separationOfDuties : true,
      riskExplanation: typeof object.riskExplanation === "string" ? object.riskExplanation : `${tool.riskLevel}-risk tool action requires human approval`,
    };
  }
}
