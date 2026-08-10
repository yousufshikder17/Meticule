import type pg from "pg";
import type { Principal } from "../db/types.js";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { deriveTaskStates, validatePlanGraph, type Plan } from "./plan-schema.js";

export interface PersistedPlan { id: string; tenantId: string; runId: string; version: number; plan: Plan; createdByStepId: string | null }

export class PlanService {
  constructor(private readonly pool: pg.Pool) {}

  async active(tenantId: string, runId: string): Promise<PersistedPlan | null> {
    const result = await this.pool.query("SELECT * FROM run_plans WHERE tenant_id=$1 AND run_id=$2 AND status='active'", [tenantId, runId]);
    return result.rowCount ? this.fromRow(result.rows[0]) : null;
  }

  async revise(command: { principal: Principal; runId: string; workerId: string; stepId: string; plan: unknown }): Promise<PersistedPlan> {
    let proposed = validatePlanGraph(command.plan);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const runResult = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [command.principal.tenantId, command.runId]);
      if (!runResult.rowCount) throw new NotFoundError("Run not found");
      const run = runResult.rows[0];
      if (run.cancellation_requested_at) throw new ConflictError("Cancellation was requested");
      if (run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at) <= new Date()) throw new ConflictError("Worker does not own a running lease");
      const sourceStep = await client.query("SELECT status,sequence FROM steps WHERE tenant_id=$1 AND run_id=$2 AND id=$3", [command.principal.tenantId, command.runId, command.stepId]);
      if (!sourceStep.rowCount || sourceStep.rows[0].status !== "succeeded") throw new ConflictError("Plan revision requires a persisted successful model step");
      const previousResult = await client.query("SELECT * FROM run_plans WHERE tenant_id=$1 AND run_id=$2 AND status='active' FOR UPDATE", [command.principal.tenantId, command.runId]);
      if (previousResult.rowCount) {
        const previous = this.fromRow(previousResult.rows[0]);
        const byId = new Map(proposed.tasks.map((task) => [task.taskId, task]));
        const previousById = new Map(previous.plan.tasks.map((task) => [task.taskId, task]));
        for (const task of proposed.tasks) {
          const prior = previousById.get(task.taskId);
          task.attemptCount = prior?.attemptCount ?? 0;
          if (["running", "completed", "failed"].includes(task.status) && prior?.status !== task.status) task.attemptCount += 1;
        }
        for (const completed of previous.plan.tasks.filter((task) => task.status === "completed")) {
          const next = byId.get(completed.taskId);
          if (!next || next.objective !== completed.objective || JSON.stringify(next.dependencies) !== JSON.stringify(completed.dependencies)) throw new ConflictError(`Completed task ${completed.taskId} cannot be removed or redefined`);
          Object.assign(next, completed);
        }
      } else {
        for (const task of proposed.tasks) task.attemptCount = ["running", "completed", "failed"].includes(task.status) ? 1 : 0;
      }
      for (const task of proposed.tasks) {
        if (task.status === "completed" || task.status === "failed") {
          if (!task.evidenceStepId) throw new ConflictError(`${task.status} task ${task.taskId} requires evidenceStepId`);
          if (task.status === "completed" && task.result === null) throw new ConflictError(`Completed task ${task.taskId} requires a result`);
          if (task.status === "failed" && task.error === null) throw new ConflictError(`Failed task ${task.taskId} requires an error`);
          const evidence = await client.query("SELECT status,sequence FROM steps WHERE tenant_id=$1 AND run_id=$2 AND id=$3", [command.principal.tenantId, command.runId, task.evidenceStepId]);
          const allowed = task.status === "completed" ? ["succeeded"] : ["failed", "unknown", "cancelled"];
          if (!evidence.rowCount || !allowed.includes(evidence.rows[0].status) || Number(evidence.rows[0].sequence) >= Number(sourceStep.rows[0].sequence)) throw new ConflictError(`Task ${task.taskId} evidence does not support ${task.status}`);
        }
      }
      proposed = deriveTaskStates(proposed);
      const taskById = new Map(proposed.tasks.map((task) => [task.taskId, task]));
      for (const task of proposed.tasks.filter((item) => ["running", "completed"].includes(item.status))) {
        if (!task.dependencies.every((dependency) => taskById.get(dependency)?.status === "completed")) throw new ConflictError(`Task ${task.taskId} cannot be ${task.status} before its dependencies complete`);
      }
      const version = previousResult.rowCount ? Number(previousResult.rows[0].version) + 1 : 1;
      if (previousResult.rowCount) await client.query("UPDATE run_plans SET status='superseded',superseded_at=now() WHERE id=$1", [previousResult.rows[0].id]);
      const inserted = await client.query(
        `INSERT INTO run_plans(tenant_id,run_id,version,status,objective,plan,created_by_step_id)
         VALUES($1,$2,$3,'active',$4,$5,$6) RETURNING *`,
        [command.principal.tenantId, command.runId, version, proposed.objective, JSON.stringify(proposed), command.stepId],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,'plan.revised',$4)", [command.principal.tenantId, command.runId, command.workerId, JSON.stringify({ planId: inserted.rows[0].id, version, sourceStepId: command.stepId })]);
      await client.query("COMMIT");
      return this.fromRow(inserted.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private fromRow(row: Record<string, unknown>): PersistedPlan {
    return { id: row.id as string, tenantId: row.tenant_id as string, runId: row.run_id as string, version: Number(row.version), plan: validatePlanGraph(row.plan), createdByStepId: row.created_by_step_id as string | null };
  }
}
