import type pg from "pg";
import type { Principal } from "../db/types.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { ToolAuthorizationError, ToolExecutionConflictError, ToolRegistry, ToolTimeoutError, UnknownToolOutcomeError, type ToolDefinition } from "./types.js";

export interface ExecuteToolCommand {
  principal: Principal; runId: string; workerId: string; toolName: string; arguments: unknown; idempotencyKey: string;
}

interface PreparedExecution { reused: boolean; output: unknown; executionId: string; stepId: string; startedAt: number }

export class ToolExecutor {
  constructor(private readonly pool: pg.Pool, private readonly registry: ToolRegistry) {}

  async execute(command: ExecuteToolCommand): Promise<unknown> {
    const tool = this.registry.get(command.toolName);
    const validatedInput = tool.inputSchema.parse(command.arguments);
    await this.authorize(command, tool, validatedInput);
    const prepared = await this.prepare(command, tool.approvalRequirement, validatedInput);
    if (prepared.reused) return prepared.output;
    await this.markRunningAfterCancellationCheck(command, prepared.executionId, prepared.stepId);
    return this.invoke(tool, validatedInput, command, prepared, false);
  }

  async resumeApproved(command: { principal: Principal; runId: string; workerId: string }): Promise<{ resumed: boolean; output?: unknown }> {
    const client = await this.pool.connect();
    let prepared: (PreparedExecution & { tool: ToolDefinition; input: unknown; idempotencyKey: string }) | null = null;
    try {
      await client.query("BEGIN");
      const pending = await client.query(
        `SELECT ap.*,te.id AS execution_id,te.status AS execution_status,te.tool_name AS execution_tool,
                te.idempotency_key AS execution_key,te.validated_arguments AS execution_arguments,
                r.status AS run_status,r.lease_owner,r.lease_expires_at,r.cancellation_requested_at,r.agent_configuration_snapshot->'allowedTools' AS allowed_tools
         FROM approvals ap
         JOIN tool_executions te ON te.tenant_id=ap.tenant_id AND te.step_id=ap.step_id
         JOIN runs r ON r.tenant_id=ap.tenant_id AND r.id=ap.run_id
         WHERE ap.tenant_id=$1 AND ap.run_id=$2 AND ap.decision='approved'
           AND (ap.consumed_at IS NULL OR te.status IN ('pending','running'))
         ORDER BY ap.created_at LIMIT 1 FOR UPDATE OF ap,te,r`,
        [command.principal.tenantId, command.runId],
      );
      if (!pending.rowCount) { await client.query("COMMIT"); return { resumed: false }; }
      const row = pending.rows[0];
      if (row.consumed_at || row.execution_status !== "pending") throw new ToolExecutionConflictError("Approved execution was already consumed and requires recovery");
      if (row.cancellation_requested_at) throw new ToolExecutionConflictError("Cancellation was requested");
      if (row.run_status !== "running" || row.lease_owner !== command.workerId || new Date(row.lease_expires_at) <= new Date()) throw new ToolExecutionConflictError("Worker does not own a running lease");
      if (row.execution_tool !== row.tool_name || row.execution_key !== row.idempotency_key) throw new ToolExecutionConflictError("Frozen approval identity does not match tool execution");
      const tool = this.registry.get(row.tool_name as string);
      if (!(row.allowed_tools as string[]).includes(tool.name)) throw new ToolAuthorizationError(`Tool is not allowed by agent: ${tool.name}`);
      for (const role of tool.authorization.requiredRoles) if (!command.principal.roles.includes(role)) throw new ToolAuthorizationError(`Missing role: ${role}`);
      const input = tool.inputSchema.parse(row.validated_arguments);
      if (canonicalJsonHash(input) !== row.canonical_argument_hash || canonicalJsonHash(row.execution_arguments) !== row.canonical_argument_hash) throw new ToolExecutionConflictError("Frozen approval argument hash mismatch");
      await tool.authorize?.(input, { pool: this.pool, principal: command.principal, runId: command.runId, idempotencyKey: row.idempotency_key, signal: new AbortController().signal });
      const consumed = await client.query("UPDATE approvals SET consumed_at=now(),consumed_by_worker=$1 WHERE id=$2 AND consumed_at IS NULL RETURNING consumed_at", [command.workerId, row.id]);
      if (!consumed.rowCount) throw new ToolExecutionConflictError("Approval was already consumed");
      await client.query("UPDATE tool_executions SET status='running',updated_at=now() WHERE id=$1 AND status='pending'", [row.execution_id]);
      await client.query("UPDATE steps SET status='running',started_at=now(),attempt_count=attempt_count+1 WHERE id=$1 AND status='pending'", [row.step_id]);
      await client.query(
        `INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES
         ($1,$2,'worker',$3,'approval.consumed',$4),
         ($1,$2,'worker',$3,'approval.execution_started',$4)`,
        [command.principal.tenantId, command.runId, command.workerId, JSON.stringify({ approvalId: row.id, tool: tool.name, idempotencyKey: row.idempotency_key })],
      );
      await client.query("COMMIT");
      prepared = { reused: false, output: null, executionId: row.execution_id, stepId: row.step_id, startedAt: Date.now(), tool, input, idempotencyKey: row.idempotency_key };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
    const output = await this.invoke(prepared.tool, prepared.input, { ...command, toolName: prepared.tool.name, arguments: prepared.input, idempotencyKey: prepared.idempotencyKey }, prepared, true);
    return { resumed: true, output };
  }

  private async authorize(command: ExecuteToolCommand, tool: ToolDefinition, input: unknown): Promise<void> {
    const result = await this.pool.query(
      `SELECT r.status,r.lease_owner,r.lease_expires_at,r.cancellation_requested_at,r.agent_configuration_snapshot->'allowedTools' AS allowed_tools
       FROM runs r
       WHERE r.tenant_id=$1 AND r.id=$2`, [command.principal.tenantId, command.runId],
    );
    if (!result.rowCount) throw new ToolAuthorizationError("Run not found in tenant");
    const run = result.rows[0];
    if (!(run.allowed_tools as string[]).includes(command.toolName)) throw new ToolAuthorizationError(`Tool is not allowed by agent: ${command.toolName}`);
    for (const role of tool.authorization.requiredRoles) if (!command.principal.roles.includes(role)) throw new ToolAuthorizationError(`Missing role: ${role}`);
    if (run.cancellation_requested_at) throw new ToolExecutionConflictError("Cancellation was requested");
    if (run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at) <= new Date()) throw new ToolExecutionConflictError("Worker does not own a running lease");
    await tool.authorize?.(input, { pool: this.pool, principal: command.principal, runId: command.runId, idempotencyKey: command.idempotencyKey, signal: new AbortController().signal });
  }

  private async prepare(command: ExecuteToolCommand, approval: "never" | "policy" | "always", input: unknown): Promise<PreparedExecution> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const operationHash = canonicalJsonHash({ tenantId: command.principal.tenantId, runId: command.runId, toolName: command.toolName, arguments: input, idempotencyKey: command.idempotencyKey });
      const duplicate = await client.query("SELECT id,run_id,step_id,tool_name,status,output,validated_arguments,operation_hash FROM tool_executions WHERE tenant_id=$1 AND idempotency_key=$2 FOR UPDATE", [command.principal.tenantId, command.idempotencyKey]);
      if (duplicate.rowCount) {
        const existing = duplicate.rows[0];
        if (existing.run_id !== command.runId || existing.tool_name !== command.toolName || canonicalJsonHash(existing.validated_arguments) !== canonicalJsonHash(input)) throw new ToolExecutionConflictError("Idempotency key identifies a different operation");
        await client.query("COMMIT");
        if (duplicate.rows[0].status === "succeeded") return { reused: true, output: duplicate.rows[0].output, executionId: duplicate.rows[0].id, stepId: duplicate.rows[0].step_id, startedAt: Date.now() };
        throw new ToolExecutionConflictError(`Idempotency key is already ${duplicate.rows[0].status}`);
      }
      const runResult = await client.query(
        `SELECT r.*,r.agent_configuration_snapshot->'allowedTools' AS allowed_tools,r.agent_configuration_snapshot->'approvalPolicy' AS approval_policy FROM runs r
         WHERE r.tenant_id=$1 AND r.id=$2 FOR UPDATE OF r`, [command.principal.tenantId, command.runId],
      );
      if (!runResult.rowCount) throw new ToolAuthorizationError("Run not found in tenant");
      const run = runResult.rows[0];
      if (run.cancellation_requested_at) throw new ToolExecutionConflictError("Cancellation was requested");
      if (run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at) <= new Date()) throw new ToolExecutionConflictError("Worker does not own a running lease");
      if (!(run.allowed_tools as string[]).includes(command.toolName)) throw new ToolAuthorizationError(`Tool is not allowed by agent: ${command.toolName}`);
      const policy = run.approval_policy as Record<string, unknown>;
      if (approval === "always" || (approval === "policy" && Boolean(policy[command.toolName]))) throw new ToolExecutionConflictError("Tool requires approval proposal flow");
      const sequence = Number(run.current_step) + 1;
      const step = await client.query(
        `INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input)
         VALUES($1,$2,$3,'tool','pending',$4,1,$5) RETURNING id`,
        [command.principal.tenantId, command.runId, sequence, command.idempotencyKey, JSON.stringify({ tool: command.toolName, arguments: input })],
      );
      const execution = await client.query(
        `INSERT INTO tool_executions(tenant_id,run_id,step_id,tool_name,idempotency_key,status,validated_arguments,operation_hash,retry_safety)
         VALUES($1,$2,$3,$4,$5,'pending',$6,$7,$8) RETURNING id`,
        [command.principal.tenantId, command.runId, step.rows[0].id, command.toolName, command.idempotencyKey, JSON.stringify(input), operationHash, this.registry.get(command.toolName).retrySafety],
      );
      await client.query("UPDATE runs SET current_step=$1,version=version+1,updated_at=now() WHERE id=$2", [sequence, command.runId]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,'tool.execution_started',$4)", [command.principal.tenantId, command.runId, command.workerId, JSON.stringify({ tool: command.toolName, idempotencyKey: command.idempotencyKey })]);
      await client.query("COMMIT");
      return { reused: false, output: null, executionId: execution.rows[0].id, stepId: step.rows[0].id, startedAt: Date.now() };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async markRunningAfterCancellationCheck(command: ExecuteToolCommand, executionId: string, stepId: string): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const run = await client.query("SELECT cancellation_requested_at,status,lease_owner,lease_expires_at FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [command.principal.tenantId, command.runId]);
      const row = run.rows[0];
      if (!row || row.cancellation_requested_at || row.status !== "running" || row.lease_owner !== command.workerId || new Date(row.lease_expires_at) <= new Date()) {
        await client.query("UPDATE tool_executions SET status='cancelled',updated_at=now() WHERE id=$1 AND status='pending'", [executionId]);
        await client.query("UPDATE steps SET status='cancelled',finished_at=now() WHERE id=$1 AND status='pending'", [stepId]);
        await client.query("COMMIT");
        throw new ToolExecutionConflictError("Cancellation or lease loss detected before tool invocation");
      }
      await client.query("UPDATE tool_executions SET status='running',updated_at=now() WHERE id=$1 AND status='pending'", [executionId]);
      await client.query("UPDATE steps SET status='running',started_at=now() WHERE id=$1 AND status='pending'", [stepId]);
      await client.query("COMMIT");
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async invoke(tool: ToolDefinition, input: unknown, command: ExecuteToolCommand, prepared: PreparedExecution, approved: boolean): Promise<unknown> {
    for (let attempt = 1; attempt <= tool.retryPolicy.maxAttempts; attempt += 1) {
      await this.assertRunCanContinue(command);
      const attemptId = await this.beginAttempt(command, prepared.executionId, attempt);
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), tool.timeoutMs);
      try {
        const output = tool.outputSchema.parse(await Promise.race([
          tool.execute(input, { pool: this.pool, principal: command.principal, runId: command.runId, idempotencyKey: command.idempotencyKey, signal: controller.signal }),
          new Promise<never>((_, reject) => controller.signal.addEventListener("abort", () => reject(new ToolTimeoutError(`Tool timed out after ${tool.timeoutMs}ms`)), { once: true })),
        ]));
        await this.finishAttempt(attemptId, "succeeded", null, null);
        await this.finish(command, prepared.executionId, prepared.stepId, "succeeded", output, null, prepared.startedAt, approved);
        return output;
      } catch (error) {
        const code = this.errorCode(error);
        await this.finishAttempt(attemptId, "failed", code, { message: error instanceof Error ? error.message : String(error) });
        const safeRetry = ["pure", "externally_idempotent"].includes(tool.retrySafety)
          && attempt < tool.retryPolicy.maxAttempts && tool.retryPolicy.retryableErrors.includes(code);
        if (safeRetry) {
          try { await this.waitForRetry(command, attempt); continue; }
          catch (waitError) {
            await this.finish(command, prepared.executionId, prepared.stepId, "cancelled", null, { code: "cancelled_between_attempts", message: waitError instanceof Error ? waitError.message : String(waitError) }, prepared.startedAt, approved);
            throw waitError;
          }
        }
        const unknown = error instanceof ToolTimeoutError && tool.idempotency === "keyed_side_effect" && tool.retrySafety !== "externally_idempotent";
        const status = unknown ? "unknown" : "failed";
        await this.finish(command, prepared.executionId, prepared.stepId, status, null, { code, message: error instanceof Error ? error.message : String(error) }, prepared.startedAt, approved, unknown ? (tool.reconcile ? "pending" : "manual") : "not_required");
        if (unknown) throw new UnknownToolOutcomeError("Tool outcome is unknown and must be reconciled before continuation");
        throw error;
      } finally { clearTimeout(timer); }
    }
    throw new ToolExecutionConflictError("Tool retry policy exhausted unexpectedly");
  }

  private async assertRunCanContinue(command: ExecuteToolCommand): Promise<void> {
    const result = await this.pool.query("SELECT status,lease_owner,lease_expires_at,cancellation_requested_at FROM runs WHERE tenant_id=$1 AND id=$2", [command.principal.tenantId, command.runId]);
    const run = result.rows[0];
    if (!run || run.cancellation_requested_at || run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at) <= new Date()) throw new ToolExecutionConflictError("Cancellation or lease loss detected before tool attempt");
  }

  private async beginAttempt(command: ExecuteToolCommand, executionId: string, attempt: number): Promise<string> {
    const result = await this.pool.query(
      `INSERT INTO tool_execution_attempts(tenant_id,run_id,tool_execution_id,attempt_number,status)
       VALUES($1,$2,$3,$4,'running') RETURNING id`,
      [command.principal.tenantId, command.runId, executionId, attempt],
    );
    await this.pool.query("UPDATE tool_executions SET attempt_count=GREATEST(attempt_count,$1),updated_at=now() WHERE id=$2", [attempt, executionId]);
    return result.rows[0].id as string;
  }

  private async finishAttempt(attemptId: string, status: "succeeded" | "failed", code: string | null, details: unknown): Promise<void> {
    await this.pool.query("UPDATE tool_execution_attempts SET status=$1,completed_at=now(),error_code=$2,error_details=$3 WHERE id=$4", [status, code, details === null ? null : JSON.stringify(details), attemptId]);
  }

  private errorCode(error: unknown): string {
    if (error instanceof ToolTimeoutError) return "timeout";
    if (error && typeof error === "object" && "code" in error && typeof (error as { code?: unknown }).code === "string") return (error as { code: string }).code;
    return "execution_error";
  }

  private async waitForRetry(command: ExecuteToolCommand, attempt: number): Promise<void> {
    await this.assertRunCanContinue(command);
    const delay = Math.min(2_000, 100 * (2 ** (attempt - 1))) + Math.floor(Math.random() * 25);
    await new Promise((resolve) => setTimeout(resolve, delay));
    await this.assertRunCanContinue(command);
  }

  private async finish(command: ExecuteToolCommand, executionId: string, stepId: string, status: "succeeded" | "failed" | "unknown" | "cancelled", output: unknown, error: unknown, startedAt: number, approved: boolean, reconciliationStatus = "not_required"): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query("UPDATE tool_executions SET status=$1,output=$2,error_details=$3,duration_ms=$4,reconciliation_status=$5,updated_at=now() WHERE id=$6 AND status='running'", [status, output === null ? null : JSON.stringify(output), error === null ? null : JSON.stringify(error), Date.now()-startedAt, reconciliationStatus, executionId]);
      await client.query("UPDATE steps SET status=$1,output=$2,error_details=$3,finished_at=now() WHERE id=$4 AND status='running'", [status, output === null ? null : JSON.stringify(output), error === null ? null : JSON.stringify(error), stepId]);
      if (approved) await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,$4,$5)", [command.principal.tenantId, command.runId, command.workerId, `approval.execution_${status}`, JSON.stringify({ tool: command.toolName, idempotencyKey: command.idempotencyKey })]);
      await client.query("COMMIT");
    } catch (failure) { await client.query("ROLLBACK").catch(() => undefined); throw failure; }
    finally { client.release(); }
  }
}
