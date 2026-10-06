import type pg from "pg";
import { assertTransition, type RunState } from "../domain/run-state.js";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { AgentConfigurationSchema, ModelConfigurationSchema, type AgentConfiguration, type AgentConfigurationInput, type Run } from "../domain/schemas.js";
import type { AgentRecord, Principal } from "./types.js";
import { hasUnresolvedToolReconciliation } from "./run-transition-guards.js";

type Row = Record<string, unknown>;

export function agentConfigurationFromDatabaseRow(row: Row): AgentConfiguration {
  return AgentConfigurationSchema.parse({
    name: row.name as string, systemInstructions: row.system_instructions as string,
    model: ModelConfigurationSchema.parse(row.model_config), allowedTools: row.allowed_tools as string[],
    maximumSteps: row.maximum_steps as number, tokenBudget: Number(row.token_budget),
    costBudgetMicrousd: Number(row.cost_budget_microusd),
    approvalPolicy: row.approval_policy as AgentConfiguration["approvalPolicy"],
    outputSchema: row.output_schema as Record<string, unknown> | null,
    composition: row.composition_config as AgentConfiguration["composition"],
    contextPolicy: row.context_policy as AgentConfiguration["contextPolicy"],
    memoryPolicy: row.memory_policy as AgentConfiguration["memoryPolicy"],
    retrievalPolicy: row.retrieval_policy as AgentConfiguration["retrievalPolicy"],
    connectorPolicy: row.connector_policy as AgentConfiguration["connectorPolicy"],
    skillPolicy: row.skill_policy as AgentConfiguration["skillPolicy"],
    orchestrationPolicy: row.orchestration_policy as AgentConfiguration["orchestrationPolicy"],
  });
}

function agentFrom(row: Row): AgentRecord {
  return {
    id: row.id as string, tenantId: row.tenant_id as string, createdBy: row.created_by as string,
    ...agentConfigurationFromDatabaseRow(row),
    version: row.version as number, createdAt: row.created_at as Date, updatedAt: row.updated_at as Date,
  };
}

function runFrom(row: Row): Run {
  return {
    id: row.id as string, tenantId: row.tenant_id as string, kind: row.kind as Run["kind"], agentId: row.agent_id as string | null,
    createdBy: row.created_by as string, goal: row.goal as string, status: row.status as RunState,
    currentStep: row.current_step as number, version: row.version as number,
    leaseOwner: row.lease_owner as string | null, leaseExpiresAt: row.lease_expires_at as Date | null,
    cancellationRequestedAt: row.cancellation_requested_at as Date | null,
    agentVersion: row.agent_version === null ? null : Number(row.agent_version), agentConfigurationSnapshot: row.agent_configuration_snapshot === null ? null : AgentConfigurationSchema.parse(row.agent_configuration_snapshot),
    parentRunId: row.parent_run_id as string | null, rootRunId: row.root_run_id as string, delegationDepth: Number(row.delegation_depth), delegationRole: row.delegation_role as string | null, contextScope: row.context_scope as Run["contextScope"],
    tokenBudgetLimit: Number(row.token_budget_limit), costBudgetLimitMicrousd: Number(row.cost_budget_limit_microusd), reservedChildTokens: Number(row.reserved_child_tokens), reservedChildCostMicrousd: Number(row.reserved_child_cost_microusd),
    inputTokens: Number(row.input_tokens), outputTokens: Number(row.output_tokens),
    costMicrousd: Number(row.cost_microusd), finalOutput: row.final_output ?? null,
    errorDetails: row.error_details ?? null, createdAt: row.created_at as Date, updatedAt: row.updated_at as Date,
  };
}

async function transaction<T>(pool: pg.Pool, work: (client: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try { await client.query("BEGIN"); const result = await work(client); await client.query("COMMIT"); return result; }
  catch (error) { await client.query("ROLLBACK"); throw error; }
  finally { client.release(); }
}

async function audit(client: pg.PoolClient, tenantId: string, runId: string | null, actorType: string, actorId: string, eventType: string, details: unknown = {}): Promise<void> {
  await client.query(
    "INSERT INTO audit_events(tenant_id, run_id, actor_type, actor_id, event_type, details) VALUES($1,$2,$3,$4,$5,$6)",
    [tenantId, runId, actorType, actorId, eventType, JSON.stringify(details)],
  );
}

export class AgentRepository {
  constructor(private readonly pool: pg.Pool) {}

  async create(principal: Principal, rawInput: AgentConfigurationInput): Promise<AgentRecord> {
    const input = AgentConfigurationSchema.parse(rawInput);
    const result = await this.pool.query(
      `INSERT INTO agents(tenant_id,created_by,name,system_instructions,model_config,allowed_tools,maximum_steps,token_budget,cost_budget_microusd,approval_policy,output_schema,composition_config,context_policy,memory_policy,retrieval_policy,connector_policy,skill_policy,orchestration_policy)
       VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18) RETURNING *`,
      [principal.tenantId, principal.userId, input.name, input.systemInstructions, input.model,
       JSON.stringify(input.allowedTools), input.maximumSteps, input.tokenBudget, input.costBudgetMicrousd,
       JSON.stringify(input.approvalPolicy), input.outputSchema ? JSON.stringify(input.outputSchema) : null, JSON.stringify(input.composition), JSON.stringify(input.contextPolicy), JSON.stringify(input.memoryPolicy), JSON.stringify(input.retrievalPolicy), JSON.stringify(input.connectorPolicy), JSON.stringify(input.skillPolicy), JSON.stringify(input.orchestrationPolicy)],
    );
    return agentFrom(result.rows[0] as Row);
  }

  async get(tenantId: string, id: string): Promise<AgentRecord> {
    const result = await this.pool.query("SELECT * FROM agents WHERE tenant_id=$1 AND id=$2", [tenantId, id]);
    if (!result.rowCount) throw new NotFoundError("Agent not found");
    return agentFrom(result.rows[0] as Row);
  }

  async patch(principal: Principal, id: string, expectedVersion: number, patch: Partial<AgentConfiguration>): Promise<AgentRecord> {
    const current = await this.get(principal.tenantId, id);
    const merged = AgentConfigurationSchema.parse({ ...current, ...patch });
    const result = await this.pool.query(
      `UPDATE agents SET name=$1,system_instructions=$2,model_config=$3,allowed_tools=$4,maximum_steps=$5,
       token_budget=$6,cost_budget_microusd=$7,approval_policy=$8,output_schema=$9,composition_config=$10,context_policy=$11,memory_policy=$12,retrieval_policy=$13,connector_policy=$14,skill_policy=$15,orchestration_policy=$16,version=version+1,updated_at=now()
       WHERE tenant_id=$17 AND id=$18 AND version=$19 RETURNING *`,
      [merged.name, merged.systemInstructions, merged.model, JSON.stringify(merged.allowedTools), merged.maximumSteps,
       merged.tokenBudget, merged.costBudgetMicrousd, JSON.stringify(merged.approvalPolicy),
       merged.outputSchema ? JSON.stringify(merged.outputSchema) : null, JSON.stringify(merged.composition), JSON.stringify(merged.contextPolicy), JSON.stringify(merged.memoryPolicy), JSON.stringify(merged.retrievalPolicy), JSON.stringify(merged.connectorPolicy), JSON.stringify(merged.skillPolicy), JSON.stringify(merged.orchestrationPolicy), principal.tenantId, id, expectedVersion],
    );
    if (!result.rowCount) throw new ConflictError("Agent version conflict");
    return agentFrom(result.rows[0] as Row);
  }
}

export class RunRepository {
  constructor(private readonly pool: pg.Pool) {}

  async create(principal: Principal, agentId: string, goal: string): Promise<Run> {
    return transaction(this.pool, async (client) => {
        const agent = await client.query("SELECT * FROM agents WHERE tenant_id=$1 AND id=$2", [principal.tenantId, agentId]);
        if (!agent.rowCount) throw new NotFoundError("Agent not found");
        const id = crypto.randomUUID();
        const snapshot = agentConfigurationFromDatabaseRow(agent.rows[0] as Row);
        const result = await client.query(
          "INSERT INTO runs(id,tenant_id,agent_id,created_by,goal,root_run_id,token_budget_limit,cost_budget_limit_microusd,agent_version,agent_configuration_snapshot) VALUES($1,$2,$3,$4,$5,$1,$6,$7,$8,$9) RETURNING *",
          [id, principal.tenantId, agentId, principal.userId, goal, agent.rows[0].token_budget, agent.rows[0].cost_budget_microusd, agent.rows[0].version, JSON.stringify(snapshot)],
      );
      const run = runFrom(result.rows[0] as Row);
      await audit(client, principal.tenantId, run.id, "user", principal.userId, "run.created", { status: "queued" });
      return run;
    });
  }

  async get(tenantId: string, id: string): Promise<Run> {
    const result = await this.pool.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2", [tenantId, id]);
    if (!result.rowCount) throw new NotFoundError("Run not found");
    return runFrom(result.rows[0] as Row);
  }

  async transition(principal: Principal, id: string, expectedVersion: number, to: RunState): Promise<Run> {
    return transaction(this.pool, async (client) => {
      const locked = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, id]);
      if (!locked.rowCount) throw new NotFoundError("Run not found");
      const before = runFrom(locked.rows[0] as Row);
      if (before.version !== expectedVersion) throw new ConflictError("Run version conflict");
      if (before.status === "waiting_for_approval" && to === "queued") throw new ConflictError("Approval decision is required before requeueing");
      if (before.status === "paused" && to === "queued") {
        const waiting = await client.query("SELECT 1 FROM orchestration_waits WHERE tenant_id=$1 AND parent_run_id=$2 AND status='waiting'", [principal.tenantId, id]);
        if (waiting.rowCount) throw new ConflictError("Child runs must resolve before requeueing the parent");
        if (await hasUnresolvedToolReconciliation(client, principal.tenantId, id)) {
          throw new ConflictError("Run cannot resume while external tool-effect reconciliation remains unresolved");
        }
      }
      assertTransition(before.status, to);
      if (to === "completed" && before.cancellationRequestedAt) throw new ConflictError("Cancelled run cannot complete");
      const release = ["queued", "waiting_for_approval", "paused", "completed", "failed", "cancelled"].includes(to);
      const result = await client.query(
        `UPDATE runs SET status=$1,version=version+1,updated_at=now(),
         lease_owner=CASE WHEN $2 THEN NULL ELSE lease_owner END,
         lease_expires_at=CASE WHEN $2 THEN NULL ELSE lease_expires_at END,
         completed_at=CASE WHEN $1::run_status IN ('completed','failed','cancelled') THEN now() ELSE completed_at END
         WHERE tenant_id=$3 AND id=$4 RETURNING *`, [to, release, principal.tenantId, id],
      );
      if (release) await client.query("DELETE FROM worker_leases WHERE run_id=$1", [id]);
      await audit(client, principal.tenantId, id, "user", principal.userId, "run.transitioned", { from: before.status, to });
      return runFrom(result.rows[0] as Row);
    });
  }

  async requestCancellation(principal: Principal, id: string): Promise<Run> {
    return transaction(this.pool, async (client) => {
      const locked = await client.query(
        `WITH RECURSIVE descendants AS (
           SELECT id FROM runs WHERE tenant_id=$1 AND id=$2
           UNION ALL SELECT child.id FROM runs child JOIN descendants parent ON child.parent_run_id=parent.id WHERE child.tenant_id=$1
         ) SELECT r.* FROM runs r WHERE r.tenant_id=$1 AND r.id IN (SELECT id FROM descendants) ORDER BY r.id FOR UPDATE`,
        [principal.tenantId, id],
      );
      if (!locked.rowCount) throw new NotFoundError("Run not found");
      const targetRow = locked.rows.find((row) => row.id === id) as Row | undefined;
      if (!targetRow) throw new NotFoundError("Run not found");
      const run = runFrom(targetRow);
      if (["completed", "failed", "cancelled"].includes(run.status)) throw new ConflictError("Terminal run cannot be cancelled");
      let result: Row | null = null;
      for (const raw of locked.rows as Row[]) {
        const descendant = runFrom(raw);
        if (["completed", "failed", "cancelled"].includes(descendant.status)) continue;
        if (descendant.status !== "cancelling") assertTransition(descendant.status, "cancelling");
        const updated = await client.query(
          `UPDATE runs SET cancellation_requested_at=COALESCE(cancellation_requested_at,now()),status='cancelling',
           version=version+1,updated_at=now() WHERE tenant_id=$1 AND id=$2 RETURNING *`, [principal.tenantId, descendant.id],
        );
        if (descendant.id === id) result = updated.rows[0] as Row;
        await audit(client, principal.tenantId, descendant.id, "user", principal.userId, descendant.id === id ? "run.cancellation_requested" : "orchestration.cancellation_propagated", { from: descendant.status, parentCancellationRunId: id });
        const approvals = await client.query("SELECT id,decision FROM approvals WHERE tenant_id=$1 AND run_id=$2 AND consumed_at IS NULL AND decision IN ('pending','approved')", [principal.tenantId, descendant.id]);
        for (const approval of approvals.rows) await audit(client, principal.tenantId, descendant.id, "user", principal.userId, "approval.cancellation_intervened", { approvalId: approval.id, decision: approval.decision });
      }
      await client.query("UPDATE orchestration_waits SET status='cancelled',resolved_at=now() WHERE tenant_id=$1 AND parent_run_id=ANY($2::uuid[]) AND status='waiting'", [principal.tenantId, locked.rows.map((row) => row.id)]);
      if (!result) throw new ConflictError("Cancellation did not update the requested run");
      return runFrom(result);
    });
  }

  async claimNext(workerId: string, leaseSeconds: number, kind: Run["kind"] = "agent"): Promise<Run | null> {
    return transaction(this.pool, async (client) => {
      const selected = await client.query(
        `SELECT * FROM runs WHERE status='queued' AND cancellation_requested_at IS NULL AND kind=$1
         ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1`, [kind],
      );
      if (!selected.rowCount) return null;
      const before = runFrom(selected.rows[0] as Row); assertTransition(before.status, "claimed");
      const result = await client.query(
        `UPDATE runs SET status='claimed',lease_owner=$1,lease_expires_at=now()+($2*interval '1 second'),
         version=version+1,updated_at=now() WHERE id=$3 RETURNING *`, [workerId, leaseSeconds, before.id],
      );
      await client.query(
        `INSERT INTO worker_leases(run_id,tenant_id,worker_id,acquired_at,heartbeat_at,expires_at,lease_generation)
         VALUES($1,$2,$3,now(),now(),now()+($4*interval '1 second'),1)
         ON CONFLICT(run_id) DO UPDATE SET worker_id=excluded.worker_id,acquired_at=now(),heartbeat_at=now(),
         expires_at=excluded.expires_at,lease_generation=worker_leases.lease_generation+1`,
        [before.id, before.tenantId, workerId, leaseSeconds],
      );
      await audit(client, before.tenantId, before.id, "worker", workerId, "run.claimed");
      return runFrom(result.rows[0] as Row);
    });
  }

  async heartbeat(runId: string, workerId: string, leaseSeconds: number): Promise<boolean> {
    return transaction(this.pool, async (client) => {
      const result = await client.query(
        `UPDATE runs SET lease_expires_at=now()+($1*interval '1 second'),updated_at=now(),version=version+1
         WHERE id=$2 AND lease_owner=$3 AND lease_expires_at>now() AND status IN ('claimed','running','cancelling') RETURNING tenant_id`,
        [leaseSeconds, runId, workerId],
      );
      if (!result.rowCount) return false;
      await client.query("UPDATE worker_leases SET heartbeat_at=now(),expires_at=now()+($1*interval '1 second') WHERE run_id=$2 AND worker_id=$3", [leaseSeconds, runId, workerId]);
      return true;
    });
  }

  async workerTransition(runId: string, workerId: string, to: RunState, details: unknown = {}): Promise<Run> {
    return transaction(this.pool, async (client) => {
      const locked = await client.query("SELECT * FROM runs WHERE id=$1 FOR UPDATE", [runId]);
      if (!locked.rowCount) throw new NotFoundError("Run not found");
      const before = runFrom(locked.rows[0] as Row);
      if (before.leaseOwner !== workerId || !before.leaseExpiresAt || before.leaseExpiresAt <= new Date()) throw new ConflictError("Worker does not own an active lease");
      assertTransition(before.status, to);
      if (to === "completed" && before.cancellationRequestedAt) throw new ConflictError("Cancelled run cannot complete");
      const release = ["queued", "waiting_for_approval", "paused", "completed", "failed", "cancelled"].includes(to);
      const result = await client.query(
        `UPDATE runs SET status=$1,version=version+1,updated_at=now(),
         lease_owner=CASE WHEN $2 THEN NULL ELSE lease_owner END,lease_expires_at=CASE WHEN $2 THEN NULL ELSE lease_expires_at END,
         error_details=CASE WHEN $1::run_status IN ('failed','paused') THEN $3::jsonb ELSE error_details END,
         completed_at=CASE WHEN $1::run_status IN ('completed','failed','cancelled') THEN now() ELSE completed_at END
         WHERE id=$4 RETURNING *`, [to, release, JSON.stringify(details), runId],
      );
      if (release) await client.query("DELETE FROM worker_leases WHERE run_id=$1", [runId]);
      await audit(client, before.tenantId, runId, "worker", workerId, "run.transitioned", { from: before.status, to, ...details as object });
      return runFrom(result.rows[0] as Row);
    });
  }

  async complete(runId: string, workerId: string, finalOutput: unknown): Promise<Run> {
    return transaction(this.pool, async (client) => {
      const locked = await client.query("SELECT * FROM runs WHERE id=$1 FOR UPDATE", [runId]);
      if (!locked.rowCount) throw new NotFoundError("Run not found");
      const before = runFrom(locked.rows[0] as Row);
      if (before.status !== "running" || before.leaseOwner !== workerId || !before.leaseExpiresAt || before.leaseExpiresAt <= new Date()) throw new ConflictError("Worker does not own a running lease");
      if (before.cancellationRequestedAt) throw new ConflictError("Cancelled run cannot complete");
      const children = await client.query(
        `SELECT d.required,child.id,child.status,child.final_output FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
         WHERE d.tenant_id=$1 AND d.parent_run_id=$2 FOR UPDATE OF child`, [before.tenantId, runId],
      );
      if (children.rows.some((child) => !["completed", "failed", "cancelled"].includes(child.status))) throw new ConflictError("Run cannot complete while child runs remain active");
      if (children.rows.some((child) => child.required && (child.status !== "completed" || child.final_output === null || ["null", "{}", "[]", '""'].includes(JSON.stringify(child.final_output))))) throw new ConflictError("Run cannot complete with a failed or empty required child output");
      assertTransition(before.status, "completed");
      const result = await client.query(
        "UPDATE runs SET final_output=$1,status='completed',lease_owner=NULL,lease_expires_at=NULL,version=version+1,updated_at=now(),completed_at=now() WHERE id=$2 RETURNING *",
        [JSON.stringify(finalOutput), runId],
      );
      await client.query("DELETE FROM worker_leases WHERE run_id=$1", [runId]);
      await audit(client, before.tenantId, runId, "worker", workerId, "run.completed", { finalOutputPersisted: true });
      return runFrom(result.rows[0] as Row);
    });
  }

  async recoverExpired(limit = 100): Promise<number> {
    return transaction(this.pool, async (client) => {
      const rows = await client.query(
        `SELECT * FROM runs WHERE lease_expires_at<=now() AND status IN ('claimed','running','cancelling')
         ORDER BY lease_expires_at FOR UPDATE SKIP LOCKED LIMIT $1`, [limit],
      );
      for (const row of rows.rows as Row[]) {
        const run = runFrom(row);
        let to: RunState = run.cancellationRequestedAt ? "cancelled" : "queued";
        let recoveryReason = "lease_expired_before_effect";
        if (run.cancellationRequestedAt && run.status === "cancelling") {
          const tool = await client.query("SELECT * FROM tool_executions WHERE tenant_id=$1 AND run_id=$2 AND status IN ('pending','running') ORDER BY created_at DESC LIMIT 1 FOR UPDATE", [run.tenantId, run.id]);
          const model = await client.query("SELECT * FROM model_attempts WHERE tenant_id=$1 AND run_id=$2 AND status IN ('pending','running') ORDER BY created_at DESC LIMIT 1 FOR UPDATE", [run.tenantId, run.id]);
          if (tool.rowCount && tool.rows[0].status === "running") {
            const execution = tool.rows[0]; recoveryReason = "cancelled_with_tool_outcome_unknown";
            await client.query("UPDATE tool_executions SET status='unknown',reconciliation_status=$1,error_details=$2,updated_at=now() WHERE id=$3", [execution.retry_safety === "reconcilable" ? "pending" : "manual", JSON.stringify({ code: "worker_crash_during_cancelled_invocation" }), execution.id]);
            await client.query("UPDATE steps SET status='unknown',error_details=$1,finished_at=now() WHERE id=$2 AND status='running'", [JSON.stringify({ code: "worker_crash_during_cancelled_invocation" }), execution.step_id]);
          } else if (tool.rowCount) {
            await client.query("UPDATE tool_executions SET status='cancelled',error_details=$1,updated_at=now() WHERE id=$2", [JSON.stringify({ code: "cancelled_before_invocation" }), tool.rows[0].id]);
            await client.query("UPDATE steps SET status='cancelled',error_details=$1,finished_at=now() WHERE id=$2 AND status='pending'", [JSON.stringify({ code: "cancelled_before_invocation" }), tool.rows[0].step_id]);
          } else if (model.rowCount) {
            recoveryReason = "cancelled_with_model_outcome_unknown";
            await client.query("UPDATE model_attempts SET status='unknown',normalized_error_code='unknown_outcome',completed_at=now(),updated_at=now() WHERE id=$1", [model.rows[0].id]);
            await client.query("UPDATE steps SET status='unknown',error_details=$1,finished_at=now() WHERE id=$2 AND status='running'", [JSON.stringify({ code: "worker_crash_during_cancelled_model" }), model.rows[0].step_id]);
          }
        }
        if (!run.cancellationRequestedAt && run.status === "running") {
          const tool = await client.query("SELECT * FROM tool_executions WHERE tenant_id=$1 AND run_id=$2 AND status IN ('pending','running') ORDER BY created_at DESC LIMIT 1 FOR UPDATE", [run.tenantId, run.id]);
          const model = await client.query("SELECT * FROM model_attempts WHERE tenant_id=$1 AND run_id=$2 AND status IN ('pending','running') ORDER BY created_at DESC LIMIT 1 FOR UPDATE", [run.tenantId, run.id]);
          if (tool.rowCount && tool.rows[0].status === "running") {
            const execution = tool.rows[0]; to = "paused"; recoveryReason = "tool_outcome_unknown";
            const reconciliation = execution.retry_safety === "reconcilable" ? "pending" : "manual";
            await client.query("UPDATE tool_executions SET status='unknown',reconciliation_status=$1,error_details=$2,updated_at=now() WHERE id=$3", [reconciliation, JSON.stringify({ code: "worker_crash_during_invocation" }), execution.id]);
            await client.query("UPDATE steps SET status='unknown',error_details=$1,finished_at=now() WHERE id=$2 AND status='running'", [JSON.stringify({ code: "worker_crash_during_invocation" }), execution.step_id]);
          } else if (tool.rowCount) {
            const execution = tool.rows[0]; to = "failed"; recoveryReason = "tool_not_invoked";
            await client.query("UPDATE tool_executions SET status='failed',error_details=$1,updated_at=now() WHERE id=$2", [JSON.stringify({ code: "worker_crash_before_invocation" }), execution.id]);
            await client.query("UPDATE steps SET status='failed',error_details=$1,finished_at=now() WHERE id=$2 AND status='pending'", [JSON.stringify({ code: "worker_crash_before_invocation" }), execution.step_id]);
          } else if (model.rowCount) {
            const attempt = model.rows[0]; to = "failed"; recoveryReason = "model_outcome_unknown";
            await client.query("UPDATE model_attempts SET status='unknown',normalized_error_code='unknown_outcome',completed_at=now(),updated_at=now() WHERE id=$1", [attempt.id]);
            await client.query("UPDATE steps SET status='unknown',error_details=$1,finished_at=now() WHERE id=$2 AND status='running'", [JSON.stringify({ code: "worker_crash_during_model_invocation" }), attempt.step_id]);
          } else {
            const orphanModelStep = await client.query("SELECT id FROM steps WHERE tenant_id=$1 AND run_id=$2 AND kind='model' AND status='running' ORDER BY sequence DESC LIMIT 1 FOR UPDATE", [run.tenantId, run.id]);
            if (orphanModelStep.rowCount) {
              to = "failed"; recoveryReason = "model_attempt_not_persisted";
              await client.query("UPDATE steps SET status='failed',error_details=$1,finished_at=now() WHERE id=$2", [JSON.stringify({ code: "worker_crash_before_model_attempt" }), orphanModelStep.rows[0].id]);
            }
          }
        }
        // ML computation is safe to restart as a new attempt; agent recovery stays unchanged.
        if (!(run.kind === "ml" && run.status === "running" && to === "queued")) assertTransition(run.status, to);
        await client.query(
          `UPDATE runs SET status=$1,lease_owner=NULL,lease_expires_at=NULL,version=version+1,updated_at=now(),
           completed_at=CASE WHEN $1::run_status='cancelled' THEN now() ELSE completed_at END WHERE id=$2`, [to, run.id],
        );
        await client.query("DELETE FROM worker_leases WHERE run_id=$1", [run.id]);
        await audit(client, run.tenantId, run.id, "system", "lease-reaper", "run.lease_recovered", { from: run.status, to, reason: recoveryReason });
      }
      return rows.rowCount ?? 0;
    });
  }

  async finalizeUnleasedCancellations(limit = 100): Promise<number> {
    return transaction(this.pool, async (client) => {
      const rows = await client.query(
        `SELECT * FROM runs WHERE status='cancelling' AND lease_owner IS NULL
         ORDER BY updated_at FOR UPDATE SKIP LOCKED LIMIT $1`, [limit],
      );
      for (const row of rows.rows as Row[]) {
        const run = runFrom(row); assertTransition(run.status, "cancelled");
        await client.query("UPDATE runs SET status='cancelled',version=version+1,updated_at=now(),completed_at=now() WHERE id=$1", [run.id]);
        await audit(client, run.tenantId, run.id, "system", "cancellation-reaper", "run.cancelled");
      }
      return rows.rowCount ?? 0;
    });
  }

  async listSteps(tenantId: string, runId: string): Promise<Row[]> {
    await this.get(tenantId, runId);
    return (await this.pool.query("SELECT * FROM steps WHERE tenant_id=$1 AND run_id=$2 ORDER BY sequence", [tenantId, runId])).rows;
  }
  async listApprovals(tenantId: string, runId: string): Promise<Row[]> {
    await this.get(tenantId, runId);
    return (await this.pool.query("SELECT * FROM approvals WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at", [tenantId, runId])).rows;
  }
  async trace(tenantId: string, runId: string): Promise<{ run: Run; steps: Row[]; modelAttempts: Row[]; toolExecutions: Row[]; toolAttempts: Row[]; connectorInvocations: Row[]; approvals: Row[]; plans: Row[]; summaries: Row[]; checkpoints: Row[]; contextBuilds: Row[]; delegations: Row[]; orchestrationWaits: Row[]; events: Row[] }> {
    const run = await this.get(tenantId, runId);
    const steps = await this.listSteps(tenantId, runId);
    const modelAttempts = await this.pool.query("SELECT * FROM model_attempts WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const toolExecutions = await this.pool.query("SELECT * FROM tool_executions WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const toolAttempts = await this.pool.query("SELECT * FROM tool_execution_attempts WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const connectorInvocations = await this.pool.query("SELECT * FROM connector_invocations WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const approvals = await this.pool.query(`SELECT id,tenant_id,run_id,step_id,tool_name,tool_risk,risk_explanation,requester_id,
      required_approver_role,decision,approver_id,decided_at,created_at,consumed_at FROM approvals WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id`, [tenantId, runId]);
    const plans = await this.pool.query("SELECT * FROM run_plans WHERE tenant_id=$1 AND run_id=$2 ORDER BY version", [tenantId, runId]);
    const summaries = await this.pool.query("SELECT * FROM context_summaries WHERE tenant_id=$1 AND run_id=$2 ORDER BY through_sequence", [tenantId, runId]);
    const checkpoints = await this.pool.query("SELECT * FROM checkpoints WHERE tenant_id=$1 AND run_id=$2 ORDER BY version", [tenantId, runId]);
    const contextBuilds = await this.pool.query("SELECT * FROM context_builds WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const delegations = await this.pool.query("SELECT id,parent_run_id,child_run_id,parent_step_id,target_agent_id,role_name,required,context_scope,token_budget,cost_budget_microusd,created_by,created_at FROM run_delegations WHERE tenant_id=$1 AND parent_run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const orchestrationWaits = await this.pool.query("SELECT * FROM orchestration_waits WHERE tenant_id=$1 AND parent_run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const events = await this.pool.query("SELECT * FROM audit_events WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    return { run, steps, modelAttempts: modelAttempts.rows, toolExecutions: toolExecutions.rows, toolAttempts: toolAttempts.rows, connectorInvocations: connectorInvocations.rows, approvals: approvals.rows, plans: plans.rows, summaries: summaries.rows, checkpoints: checkpoints.rows, contextBuilds: contextBuilds.rows, delegations: delegations.rows, orchestrationWaits: orchestrationWaits.rows, events: events.rows };
  }
}
