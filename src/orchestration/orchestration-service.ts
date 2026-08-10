import type pg from "pg";
import type { Principal } from "../db/types.js";
import { ConflictError, NotFoundError, AuthorizationError } from "../domain/errors.js";
import { OrchestrationPolicySchema } from "../domain/schemas.js";
import type { DelegateRunInput, DelegateRunOutput } from "./orchestration-schema.js";
import { agentConfigurationFromDatabaseRow } from "../db/repositories.js";
import { hasUnresolvedToolReconciliation } from "../db/run-transition-guards.js";

const TERMINAL = ["completed", "failed", "cancelled"];
const EMPTY_OUTPUTS = new Set(["null", "{}", "[]", '""']);

type ChildState = {
  id: string; agentId: string; roleName: string; required: boolean; contextScope: "private" | "shared";
  status: string; finalOutput: unknown; errorDetails: unknown; inputTokens: number; outputTokens: number; costMicrousd: number;
};

export class RequiredChildRunError extends ConflictError {}

function isEmptyOutput(value: unknown): boolean {
  return value === null || EMPTY_OUTPUTS.has(JSON.stringify(value));
}

export class OrchestrationService {
  constructor(private readonly pool: pg.Pool) {}

  async authorizeDelegation(principal: Principal, parentRunId: string, input: DelegateRunInput, idempotencyKey?: string): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      if (idempotencyKey) {
        const duplicate = await client.query(
          `SELECT d.*,r.goal FROM run_delegations d JOIN runs r ON r.tenant_id=d.tenant_id AND r.id=d.child_run_id
           WHERE d.tenant_id=$1 AND d.parent_run_id=$2 AND d.idempotency_key=$3`, [principal.tenantId, parentRunId, idempotencyKey],
        );
        if (duplicate.rowCount) {
          const row = duplicate.rows[0];
          if (row.target_agent_id !== input.targetAgentId || row.goal !== input.goal || row.role_name !== input.roleName || row.required !== input.required || row.context_scope !== input.contextScope || Number(row.token_budget) !== input.tokenBudget || Number(row.cost_budget_microusd) !== input.costBudgetMicrousd) throw new ConflictError("Delegation idempotency key identifies a different operation");
          await client.query("ROLLBACK"); return;
        }
      }
      await this.validateDelegation(client, principal, parentRunId, input, false); await client.query("ROLLBACK");
    }
    catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async delegate(principal: Principal, parentRunId: string, idempotencyKey: string, input: DelegateRunInput): Promise<DelegateRunOutput> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const duplicate = await client.query("SELECT * FROM run_delegations WHERE tenant_id=$1 AND parent_run_id=$2 AND idempotency_key=$3 FOR UPDATE", [principal.tenantId, parentRunId, idempotencyKey]);
      if (duplicate.rowCount) {
        const row = duplicate.rows[0];
        if (row.target_agent_id !== input.targetAgentId || row.role_name !== input.roleName || row.required !== input.required || row.context_scope !== input.contextScope || Number(row.token_budget) !== input.tokenBudget || Number(row.cost_budget_microusd) !== input.costBudgetMicrousd) throw new ConflictError("Delegation idempotency key identifies a different operation");
        const child = await client.query("SELECT goal,status,delegation_depth FROM runs WHERE tenant_id=$1 AND id=$2", [principal.tenantId, row.child_run_id]);
        if (!child.rowCount || child.rows[0].goal !== input.goal) throw new ConflictError("Delegation idempotency key identifies a different goal");
        await client.query("COMMIT");
        return { childRunId: row.child_run_id, status: "queued", depth: Number(child.rows[0].delegation_depth), roleName: row.role_name };
      }
      const validated = await this.validateDelegation(client, principal, parentRunId, input, true);
      const execution = await client.query("SELECT step_id FROM tool_executions WHERE tenant_id=$1 AND run_id=$2 AND idempotency_key=$3 AND tool_name='delegate_run' FOR UPDATE", [principal.tenantId, parentRunId, idempotencyKey]);
      if (!execution.rowCount) throw new ConflictError("Canonical delegation tool execution was not persisted");
      const childRunId = crypto.randomUUID();
      const sharedContext = input.contextScope === "shared" ? await this.sharedSnapshot(client, principal.tenantId, parentRunId) : null;
      await client.query(
        `INSERT INTO runs(id,tenant_id,agent_id,created_by,goal,parent_run_id,root_run_id,delegation_depth,delegation_role,context_scope,token_budget_limit,cost_budget_limit_microusd,agent_version,agent_configuration_snapshot)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`,
        [childRunId, principal.tenantId, input.targetAgentId, principal.userId, input.goal, parentRunId, validated.parent.root_run_id, Number(validated.parent.delegation_depth) + 1, input.roleName, input.contextScope, input.tokenBudget, input.costBudgetMicrousd, validated.target.version, JSON.stringify(agentConfigurationFromDatabaseRow(validated.target))],
      );
      await client.query(
        `INSERT INTO run_delegations(tenant_id,parent_run_id,child_run_id,parent_step_id,target_agent_id,idempotency_key,role_name,required,context_scope,shared_context,token_budget,cost_budget_microusd,created_by)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)`,
        [principal.tenantId, parentRunId, childRunId, execution.rows[0].step_id, input.targetAgentId, idempotencyKey, input.roleName, input.required, input.contextScope, sharedContext === null ? null : JSON.stringify(sharedContext), input.tokenBudget, input.costBudgetMicrousd, principal.userId],
      );
      await client.query("UPDATE runs SET reserved_child_tokens=reserved_child_tokens+$1,reserved_child_cost_microusd=reserved_child_cost_microusd+$2,version=version+1,updated_at=now() WHERE tenant_id=$3 AND id=$4", [input.tokenBudget, input.costBudgetMicrousd, principal.tenantId, parentRunId]);
      await client.query(
        `INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES
         ($1,$2,'agent',$3,'orchestration.child_delegated',$4),
         ($1,$5,'system','orchestrator','orchestration.child_created',$6)`,
        [principal.tenantId, parentRunId, principal.userId, JSON.stringify({ childRunId, targetAgentId: input.targetAgentId, roleName: input.roleName, required: input.required, contextScope: input.contextScope }), childRunId, JSON.stringify({ parentRunId, rootRunId: validated.parent.root_run_id, roleName: input.roleName })],
      );
      await client.query("COMMIT");
      return { childRunId, status: "queued", depth: Number(validated.parent.delegation_depth) + 1, roleName: input.roleName };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async reconcileDelegation(principal: Principal, parentRunId: string, idempotencyKey: string): Promise<DelegateRunOutput | null> {
    const result = await this.pool.query(
      `SELECT d.child_run_id,d.role_name,r.status,r.delegation_depth FROM run_delegations d
       JOIN runs r ON r.tenant_id=d.tenant_id AND r.id=d.child_run_id
       WHERE d.tenant_id=$1 AND d.parent_run_id=$2 AND d.idempotency_key=$3`,
      [principal.tenantId, parentRunId, idempotencyKey],
    );
    if (!result.rowCount) return null;
    return { childRunId: result.rows[0].child_run_id, status: "queued", depth: Number(result.rows[0].delegation_depth), roleName: result.rows[0].role_name };
  }

  async pauseForChildren(principal: Principal, runId: string, workerId: string, reason: string): Promise<"paused" | "ready" | "none"> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const parent = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, runId]);
      if (!parent.rowCount) throw new NotFoundError("Run not found");
      const row = parent.rows[0];
      if (row.status !== "running" || row.lease_owner !== workerId || new Date(row.lease_expires_at) <= new Date()) throw new ConflictError("Worker does not own a running lease");
      const children = await this.childRows(client, principal.tenantId, runId, true);
      if (!children.length) { await client.query("COMMIT"); return "none"; }
      if (children.every((child) => TERMINAL.includes(child.status))) { await client.query("COMMIT"); return "ready"; }
      const snapshot = children.map(({ finalOutput: _output, errorDetails: _error, ...child }) => child);
      await client.query("INSERT INTO orchestration_waits(tenant_id,parent_run_id,status,reason,child_snapshot) VALUES($1,$2,'waiting',$3,$4) ON CONFLICT(tenant_id,parent_run_id) WHERE status='waiting' DO NOTHING", [principal.tenantId, runId, reason, JSON.stringify(snapshot)]);
      await client.query("UPDATE runs SET status='paused',lease_owner=NULL,lease_expires_at=NULL,error_details=$1,version=version+1,updated_at=now() WHERE tenant_id=$2 AND id=$3", [JSON.stringify({ code: "waiting_for_children", reason }), principal.tenantId, runId]);
      await client.query("DELETE FROM worker_leases WHERE run_id=$1", [runId]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,'orchestration.waiting',$4)", [principal.tenantId, runId, workerId, JSON.stringify({ childCount: children.length, reason })]);
      await client.query("COMMIT"); return "paused";
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async advanceNext(workerId: string): Promise<boolean> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const selected = await client.query(
        `SELECT ow.* FROM orchestration_waits ow JOIN runs parent ON parent.tenant_id=ow.tenant_id AND parent.id=ow.parent_run_id
         WHERE ow.status='waiting' AND NOT EXISTS (
           SELECT 1 FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
           WHERE d.tenant_id=ow.tenant_id AND d.parent_run_id=ow.parent_run_id AND child.status NOT IN ('completed','failed','cancelled')
         ) ORDER BY ow.created_at,ow.id FOR UPDATE OF ow,parent SKIP LOCKED LIMIT 1`,
      );
      if (!selected.rowCount) { await client.query("COMMIT"); return false; }
      const wait = selected.rows[0];
      const parent = await client.query("SELECT status,cancellation_requested_at FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [wait.tenant_id, wait.parent_run_id]);
      const cancelled = !parent.rowCount || parent.rows[0].cancellation_requested_at || parent.rows[0].status === "cancelling";
      await client.query("UPDATE orchestration_waits SET status=$1,resolved_at=now() WHERE id=$2", [cancelled ? "cancelled" : "resolved", wait.id]);
      if (!cancelled && parent.rows[0].status === "paused" && !await hasUnresolvedToolReconciliation(client, wait.tenant_id, wait.parent_run_id)) {
        await client.query("UPDATE runs SET status='queued',error_details=NULL,version=version+1,updated_at=now() WHERE id=$1", [wait.parent_run_id]);
      }
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,$4,$5)", [wait.tenant_id, wait.parent_run_id, workerId, cancelled ? "orchestration.wait_cancelled" : "orchestration.children_resolved", JSON.stringify({ waitId: wait.id })]);
      await client.query("COMMIT"); return true;
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async assertReadyForTurn(tenantId: string, runId: string): Promise<void> {
    const wait = await this.pool.query("SELECT status FROM orchestration_waits WHERE tenant_id=$1 AND parent_run_id=$2 ORDER BY created_at DESC,id DESC LIMIT 1", [tenantId, runId]);
    if (!wait.rowCount || wait.rows[0].status !== "resolved") return;
    const children = await this.children(tenantId, runId);
    if (children.some((child) => !TERMINAL.includes(child.status))) throw new ConflictError("Child runs are still active");
    const invalid = children.find((child) => child.required && (child.status !== "completed" || isEmptyOutput(child.finalOutput)));
    if (invalid) throw new RequiredChildRunError(`Required child run ${invalid.id} did not complete with a non-empty result`);
  }

  async assertCanComplete(tenantId: string, runId: string): Promise<void> { await this.assertReadyForTurn(tenantId, runId); }

  async context(tenantId: string, runId: string): Promise<{ parent: unknown | null; sharedContext: unknown | null; children: ChildState[]; selectedChildRunIds: string[] }> {
    const parent = await this.pool.query(
      `SELECT d.parent_run_id,d.role_name,d.context_scope,d.shared_context,parent.goal AS parent_goal
       FROM run_delegations d JOIN runs parent ON parent.tenant_id=d.tenant_id AND parent.id=d.parent_run_id
       WHERE d.tenant_id=$1 AND d.child_run_id=$2`, [tenantId, runId],
    );
    const children = await this.children(tenantId, runId);
    return {
      parent: parent.rowCount ? { runId: parent.rows[0].parent_run_id, roleName: parent.rows[0].role_name, goal: parent.rows[0].parent_goal } : null,
      sharedContext: parent.rowCount && parent.rows[0].context_scope === "shared" ? parent.rows[0].shared_context : null,
      children,
      selectedChildRunIds: children.map((child) => child.id),
    };
  }

  async tree(principal: Principal, runId: string): Promise<{ rootRunId: string; runs: unknown[]; delegations: unknown[] }> {
    const root = await this.pool.query("SELECT root_run_id FROM runs WHERE tenant_id=$1 AND id=$2", [principal.tenantId, runId]);
    if (!root.rowCount) throw new NotFoundError("Run not found");
    const rootId = root.rows[0].root_run_id;
    const [runs, delegations] = await Promise.all([
      this.pool.query("SELECT id,parent_run_id,root_run_id,agent_id,delegation_depth,delegation_role,context_scope,status,input_tokens,output_tokens,cost_microusd,final_output,error_details,created_at,completed_at FROM runs WHERE tenant_id=$1 AND root_run_id=$2 ORDER BY delegation_depth,created_at,id", [principal.tenantId, rootId]),
      this.pool.query("SELECT id,parent_run_id,child_run_id,target_agent_id,role_name,required,context_scope,token_budget,cost_budget_microusd,created_at FROM run_delegations WHERE tenant_id=$1 AND parent_run_id IN (SELECT id FROM runs WHERE tenant_id=$1 AND root_run_id=$2) ORDER BY created_at,id", [principal.tenantId, rootId]),
    ]);
    return { rootRunId: rootId, runs: runs.rows, delegations: delegations.rows };
  }

  private async children(tenantId: string, runId: string): Promise<ChildState[]> {
    const result = await this.pool.query(
      `SELECT child.id,child.agent_id,d.role_name,d.required,d.context_scope,child.status,child.final_output,child.error_details,child.input_tokens,child.output_tokens,child.cost_microusd
       FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
       WHERE d.tenant_id=$1 AND d.parent_run_id=$2 ORDER BY d.created_at,d.id`, [tenantId, runId],
    );
    return result.rows.map((row) => ({ id: row.id, agentId: row.agent_id, roleName: row.role_name, required: row.required, contextScope: row.context_scope, status: row.status, finalOutput: row.final_output, errorDetails: row.error_details, inputTokens: Number(row.input_tokens), outputTokens: Number(row.output_tokens), costMicrousd: Number(row.cost_microusd) }));
  }

  private async childRows(client: pg.PoolClient, tenantId: string, runId: string, lock: boolean): Promise<ChildState[]> {
    const result = await client.query(
      `SELECT child.id,child.agent_id,d.role_name,d.required,d.context_scope,child.status,child.final_output,child.error_details,child.input_tokens,child.output_tokens,child.cost_microusd
       FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
       WHERE d.tenant_id=$1 AND d.parent_run_id=$2 ORDER BY child.id ${lock ? "FOR UPDATE OF child" : ""}`, [tenantId, runId],
    );
    return result.rows.map((row) => ({ id: row.id, agentId: row.agent_id, roleName: row.role_name, required: row.required, contextScope: row.context_scope, status: row.status, finalOutput: row.final_output, errorDetails: row.error_details, inputTokens: Number(row.input_tokens), outputTokens: Number(row.output_tokens), costMicrousd: Number(row.cost_microusd) }));
  }

  private async validateDelegation(client: pg.PoolClient, principal: Principal, parentRunId: string, input: DelegateRunInput, lock: boolean): Promise<{ parent: Record<string, unknown>; target: Record<string, unknown> }> {
    const parent = await client.query(
      `SELECT r.*,r.agent_configuration_snapshot->'orchestrationPolicy' AS orchestration_policy FROM runs r
       WHERE r.tenant_id=$1 AND r.id=$2 ${lock ? "FOR UPDATE OF r" : ""}`, [principal.tenantId, parentRunId],
    );
    if (!parent.rowCount) throw new NotFoundError("Parent run not found");
    const row = parent.rows[0]; const policy = OrchestrationPolicySchema.parse(row.orchestration_policy);
    if (!policy.enabled) throw new AuthorizationError("Agent orchestration is disabled");
    if (!policy.allowedAgentIds.includes(input.targetAgentId)) throw new AuthorizationError("Target agent is not allowed for delegation");
    if (!policy.allowedRoles.includes(input.roleName)) throw new AuthorizationError("Delegation role is not allowed");
    if (input.contextScope === "shared" && !policy.allowSharedContext) throw new AuthorizationError("Shared parent context is not allowed");
    if (Number(row.delegation_depth) + 1 > policy.maximumDepth) throw new ConflictError("Maximum delegation depth would be exceeded");
    if (input.tokenBudget > policy.maximumChildTokenBudget || input.costBudgetMicrousd > policy.maximumChildCostBudgetMicrousd) throw new ConflictError("Child budget exceeds orchestration policy");
    if (Number(row.input_tokens) + Number(row.output_tokens) + Number(row.reserved_child_tokens) + input.tokenBudget > Number(row.token_budget_limit) || Number(row.cost_microusd) + Number(row.reserved_child_cost_microusd) + input.costBudgetMicrousd > Number(row.cost_budget_limit_microusd)) throw new ConflictError("Delegation exceeds the parent run budget");
    const target = await client.query("SELECT * FROM agents WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.targetAgentId]);
    if (!target.rowCount) throw new NotFoundError("Target agent not found");
    if (input.tokenBudget > Number(target.rows[0].token_budget) || input.costBudgetMicrousd > Number(target.rows[0].cost_budget_microusd)) throw new ConflictError("Delegated budget exceeds the target agent budget");
    const counts = await client.query(
      `SELECT count(*)::int AS total,count(*) FILTER (WHERE child.status NOT IN ('completed','failed','cancelled'))::int AS active
       FROM run_delegations d JOIN runs child ON child.tenant_id=d.tenant_id AND child.id=d.child_run_id
       WHERE d.tenant_id=$1 AND d.parent_run_id=$2`, [principal.tenantId, parentRunId],
    );
    if (Number(counts.rows[0].total) >= policy.maximumChildren) throw new ConflictError("Maximum delegated child count reached");
    if (Number(counts.rows[0].active) >= policy.maximumParallel) throw new ConflictError("Maximum parallel child count reached");
    const ancestors = await client.query(
      `WITH RECURSIVE chain AS (
         SELECT id,parent_run_id,agent_id FROM runs WHERE tenant_id=$1 AND id=$2
         UNION ALL SELECT parent.id,parent.parent_run_id,parent.agent_id FROM runs parent JOIN chain child ON child.parent_run_id=parent.id WHERE parent.tenant_id=$1
       ) SELECT agent_id FROM chain`, [principal.tenantId, parentRunId],
    );
    if (ancestors.rows.some((ancestor) => ancestor.agent_id === input.targetAgentId)) throw new ConflictError("Delegation would create an agent ancestry loop");
    return { parent: row, target: target.rows[0] };
  }

  private async sharedSnapshot(client: pg.PoolClient, tenantId: string, parentRunId: string): Promise<unknown> {
    const parent = await client.query("SELECT goal FROM runs WHERE tenant_id=$1 AND id=$2", [tenantId, parentRunId]);
    const steps = await client.query("SELECT sequence,kind,status,output,error_details FROM steps WHERE tenant_id=$1 AND run_id=$2 AND status IN ('succeeded','failed','cancelled','unknown') ORDER BY sequence DESC LIMIT 10", [tenantId, parentRunId]);
    return { parentGoal: String(parent.rows[0].goal).slice(0, 8_000), persistedSteps: steps.rows.reverse() };
  }
}
