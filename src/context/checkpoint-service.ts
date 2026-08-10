import { z } from "zod";
import type pg from "pg";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { ConflictError } from "../domain/errors.js";

const CheckpointStateSchema = z.object({
  runVersion: z.int().positive(), currentStep: z.int().nonnegative(),
  activePlan: z.object({ id: z.uuid(), version: z.int().positive() }).nullable(),
  summary: z.object({ id: z.uuid(), throughSequence: z.int().positive() }).nullable(),
  lastTerminalStepSequence: z.int().nonnegative(),
});
export type CheckpointState = z.infer<typeof CheckpointStateSchema>;
export interface RestoredCheckpoint { id: string; version: number; state: CheckpointState }

export class CheckpointService {
  constructor(private readonly pool: pg.Pool) {}

  async create(tenantId: string, runId: string, provenance: Record<string, unknown>): Promise<RestoredCheckpoint> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const run = await client.query("SELECT version,current_step FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [tenantId, runId]);
      if (!run.rowCount) throw new ConflictError("Run not found for checkpoint");
      const plan = await client.query("SELECT id,version FROM run_plans WHERE tenant_id=$1 AND run_id=$2 AND status='active'", [tenantId, runId]);
      const summary = await client.query("SELECT id,through_sequence FROM context_summaries WHERE tenant_id=$1 AND run_id=$2 ORDER BY through_sequence DESC LIMIT 1", [tenantId, runId]);
      const step = await client.query("SELECT COALESCE(max(sequence),0) AS sequence FROM steps WHERE tenant_id=$1 AND run_id=$2 AND status IN ('succeeded','failed','cancelled','unknown')", [tenantId, runId]);
      const prior = await client.query("SELECT COALESCE(max(version),0)+1 AS version FROM checkpoints WHERE tenant_id=$1 AND run_id=$2", [tenantId, runId]);
      const state: CheckpointState = {
        runVersion: Number(run.rows[0].version), currentStep: Number(run.rows[0].current_step),
        activePlan: plan.rowCount ? { id: plan.rows[0].id, version: Number(plan.rows[0].version) } : null,
        summary: summary.rowCount ? { id: summary.rows[0].id, throughSequence: Number(summary.rows[0].through_sequence) } : null,
        lastTerminalStepSequence: Number(step.rows[0].sequence),
      };
      const version = Number(prior.rows[0].version); const checksum = canonicalJsonHash(state);
      const inserted = await client.query("INSERT INTO checkpoints(tenant_id,run_id,state,version,kind,state_checksum,provenance) VALUES($1,$2,$3,$4,'working_state',$5,$6) RETURNING id", [tenantId, runId, JSON.stringify(state), version, checksum, JSON.stringify(provenance)]);
      await client.query("COMMIT");
      return { id: inserted.rows[0].id, version, state };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async restoreLatest(tenantId: string, runId: string): Promise<RestoredCheckpoint | null> {
    const result = await this.pool.query("SELECT id,version,state,state_checksum FROM checkpoints WHERE tenant_id=$1 AND run_id=$2 AND kind='working_state' ORDER BY version DESC LIMIT 1", [tenantId, runId]);
    if (!result.rowCount) return null;
    const state = CheckpointStateSchema.parse(result.rows[0].state);
    if (canonicalJsonHash(state) !== result.rows[0].state_checksum) throw new ConflictError("Checkpoint checksum mismatch");
    return { id: result.rows[0].id, version: Number(result.rows[0].version), state };
  }
}
