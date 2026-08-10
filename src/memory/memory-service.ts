import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { CreateMemorySchema, CorrectMemorySchema, MemoryListQuerySchema, MemoryToolInputSchema, type CreateMemoryInput, type MemoryToolInput } from "./memory-schema.js";

type Row = Record<string, unknown>;
export interface MemoryRecord {
  id: string; tenantId: string; scope: "user" | "agent" | "tenant"; ownerUserId: string | null; agentId: string | null;
  memoryType: "fact" | "note" | "procedure" | "outcome"; content: { text: string; attributes: Record<string, unknown> };
  provenance: Record<string, unknown>; creationReason: string; sourceRunId: string | null; createdBy: string; writeSource: string;
  relevance: { tags: string[]; importance: number }; retentionUntil: Date | null; version: number; correctedFromId: string | null;
  deletedAt: Date | null; createdAt: Date; updatedAt: Date;
}

export class MemoryService {
  constructor(private readonly pool: pg.Pool) {}

  async create(principal: Principal, raw: unknown): Promise<MemoryRecord> {
    const input = CreateMemorySchema.parse(raw);
    this.authorizeCreate(principal, input);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await this.verifyReferences(client, principal.tenantId, input.agentId, input.sourceRunId);
      this.validateRetention(input.retentionUntil);
      const inserted = await client.query(
        `INSERT INTO memories(tenant_id,scope,owner_user_id,agent_id,memory_type,content,content_text,provenance,creation_reason,source_run_id,created_by,write_source,relevance_metadata,retention_until)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'user',$12,$13) RETURNING *`,
        [principal.tenantId, input.scope, input.scope === "user" ? principal.userId : null, input.agentId, input.memoryType, JSON.stringify(input.content), input.content.text,
         JSON.stringify({ ...input.provenance, recordedBy: "authenticated_api" }), input.creationReason, input.sourceRunId, principal.userId, JSON.stringify(input.relevance), input.retentionUntil],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'user',$3,'memory.created',$4)", [principal.tenantId, input.sourceRunId, principal.userId, JSON.stringify({ memoryId: inserted.rows[0].id, scope: input.scope, memoryType: input.memoryType, writeSource: "user" })]);
      await client.query("COMMIT"); return this.fromRow(inserted.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async list(principal: Principal, rawQuery: unknown): Promise<MemoryRecord[]> {
    const query = MemoryListQuerySchema.parse(rawQuery);
    if (query.agentId) await this.verifyAgent(this.pool, principal.tenantId, query.agentId);
    const terms = query.q?.toLowerCase().split(/\s+/).filter(Boolean) ?? [];
    const result = await this.pool.query(
      `SELECT * FROM memories WHERE tenant_id=$1 AND deleted_at IS NULL AND (retention_until IS NULL OR retention_until>now())
       AND (scope='tenant' OR (scope='user' AND owner_user_id=$2) OR (scope='agent' AND agent_id=$3))
       AND ($4::text IS NULL OR memory_type=$4) ORDER BY created_at DESC LIMIT $5`,
      [principal.tenantId, principal.userId, query.agentId ?? null, query.memoryType ?? null, Math.min(500, query.limit * 5)],
    );
    return result.rows.map((row) => this.fromRow(row)).filter((memory) => !terms.length || terms.some((term) => `${memory.content.text} ${memory.relevance.tags.join(" ")}`.toLowerCase().includes(term))).slice(0, query.limit);
  }

  async provenance(principal: Principal, memoryId: string): Promise<{ memory: MemoryRecord; correctionChain: MemoryRecord[] }> {
    const memory = await this.accessible(principal, memoryId, true);
    const chain = await this.pool.query(
      `WITH RECURSIVE history AS (
        SELECT * FROM memories WHERE tenant_id=$1 AND id=$2
        UNION ALL SELECT prior.* FROM memories prior JOIN history current ON prior.tenant_id=current.tenant_id AND prior.id=current.corrected_from_id
      ) SELECT * FROM history ORDER BY version DESC`, [principal.tenantId, memoryId],
    );
    return { memory, correctionChain: chain.rows.map((row) => this.fromRow(row)) };
  }

  async delete(principal: Principal, memoryId: string, reason = "user_requested_deletion"): Promise<MemoryRecord> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("SELECT * FROM memories WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, memoryId]);
      if (!result.rowCount) throw new NotFoundError("Memory not found");
      this.authorizeManage(principal, result.rows[0]);
      if (!result.rows[0].deleted_at) await client.query("UPDATE memories SET deleted_at=now(),deleted_by=$1,deletion_reason=$2,updated_at=now() WHERE id=$3", [principal.userId, reason, memoryId]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'user',$3,'memory.deleted',$4)", [principal.tenantId, result.rows[0].source_run_id, principal.userId, JSON.stringify({ memoryId, reason })]);
      const updated = await client.query("SELECT * FROM memories WHERE id=$1", [memoryId]);
      await client.query("COMMIT"); return this.fromRow(updated.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async correct(principal: Principal, memoryId: string, raw: unknown): Promise<MemoryRecord> {
    const input = CorrectMemorySchema.parse(raw); this.validateRetention(input.retentionUntil);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const oldResult = await client.query("SELECT * FROM memories WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, memoryId]);
      if (!oldResult.rowCount || oldResult.rows[0].deleted_at) throw new NotFoundError("Active memory not found");
      const old = oldResult.rows[0]; this.authorizeManage(principal, old);
      const inserted = await client.query(
        `INSERT INTO memories(tenant_id,scope,owner_user_id,agent_id,memory_type,content,content_text,provenance,creation_reason,source_run_id,created_by,write_source,relevance_metadata,retention_until,version,corrected_from_id)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'user',$12,$13,$14,$15) RETURNING *`,
        [principal.tenantId, old.scope, old.owner_user_id, old.agent_id, old.memory_type, JSON.stringify(input.content), input.content.text,
         JSON.stringify({ ...input.provenance, recordedBy: "authenticated_api", correctionOf: memoryId }), input.creationReason, old.source_run_id, principal.userId,
         JSON.stringify(input.relevance), input.retentionUntil, Number(old.version) + 1, memoryId],
      );
      await client.query("UPDATE memories SET deleted_at=now(),deleted_by=$1,deletion_reason='corrected',updated_at=now() WHERE id=$2", [principal.userId, memoryId]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'user',$3,'memory.corrected',$4)", [principal.tenantId, old.source_run_id, principal.userId, JSON.stringify({ previousMemoryId: memoryId, memoryId: inserted.rows[0].id, version: Number(old.version) + 1 })]);
      await client.query("COMMIT"); return this.fromRow(inserted.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async searchForContext(command: { tenantId: string; userId: string; agentId: string; query: string; limit: number; maxTokens: number }): Promise<MemoryRecord[]> {
    const result = await this.pool.query(
      `SELECT * FROM memories WHERE tenant_id=$1 AND deleted_at IS NULL AND (retention_until IS NULL OR retention_until>now())
       AND (scope='tenant' OR (scope='user' AND owner_user_id=$2) OR (scope='agent' AND agent_id=$3)) ORDER BY created_at DESC LIMIT 200`,
      [command.tenantId, command.userId, command.agentId],
    );
    const terms = new Set(command.query.toLowerCase().split(/[^a-z0-9]+/).filter((term) => term.length > 2));
    const ranked = result.rows.map((row) => this.fromRow(row)).map((memory) => {
      const words = `${memory.content.text} ${memory.relevance.tags.join(" ")}`.toLowerCase().split(/[^a-z0-9]+/);
      const overlap = words.filter((word) => terms.has(word)).length;
      return { memory, score: overlap * 100 + memory.relevance.importance * 10 + memory.createdAt.getTime() / 1e13 };
    }).filter((item) => item.score >= 100).sort((a, b) => b.score - a.score);
    const selected: MemoryRecord[] = []; let tokens = 0;
    for (const item of ranked) {
      const estimate = Math.ceil(JSON.stringify({ content: item.memory.content, provenance: item.memory.provenance }).length / 4);
      if (selected.length >= command.limit || tokens + estimate > command.maxTokens) continue;
      selected.push(item.memory); tokens += estimate;
    }
    return selected;
  }

  async authorizeToolWrite(principal: Principal, runId: string, raw: unknown, idempotencyKey?: string): Promise<{ input: MemoryToolInput; agentId: string }> {
    const input = MemoryToolInputSchema.parse(raw);
    const result = await this.pool.query(
      `SELECT r.cancellation_requested_at,r.agent_id,r.agent_configuration_snapshot->'memoryPolicy' AS memory_policy FROM runs r
       WHERE r.tenant_id=$1 AND r.id=$2`, [principal.tenantId, runId],
    );
    if (!result.rowCount) throw new AuthorizationError("Memory write run was not found");
    if (result.rows[0].cancellation_requested_at) throw new ConflictError("Cancellation was requested");
    const policy = result.rows[0].memory_policy as { writeEnabled?: boolean; allowedScopes?: string[]; maxWritesPerRun?: number };
    if (!policy.writeEnabled || !policy.allowedScopes?.includes(input.scope)) throw new AuthorizationError("Agent memory write policy does not allow this scope");
    if (idempotencyKey) {
      const existing = await this.pool.query("SELECT source_run_id FROM memories WHERE tenant_id=$1 AND idempotency_key=$2", [principal.tenantId, idempotencyKey]);
      if (existing.rowCount && existing.rows[0].source_run_id === runId) return { input, agentId: result.rows[0].agent_id };
    }
    const count = await this.pool.query("SELECT count(*) AS count FROM memories WHERE tenant_id=$1 AND source_run_id=$2 AND write_source='approved_tool' AND deleted_at IS NULL", [principal.tenantId, runId]);
    if (Number(count.rows[0].count) >= Number(policy.maxWritesPerRun ?? 0)) throw new ConflictError("Run memory write quota is exhausted");
    return { input, agentId: result.rows[0].agent_id };
  }

  async createFromApprovedTool(principal: Principal, runId: string, idempotencyKey: string, raw: unknown): Promise<{ memoryId: string; version: number }> {
    const { input, agentId } = await this.authorizeToolWrite(principal, runId, raw, idempotencyKey);
    const retentionUntil = input.retentionDays === null ? null : new Date(Date.now() + input.retentionDays * 86_400_000);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const existing = await client.query("SELECT * FROM memories WHERE tenant_id=$1 AND idempotency_key=$2 FOR UPDATE", [principal.tenantId, idempotencyKey]);
      if (existing.rowCount) { await client.query("COMMIT"); return { memoryId: existing.rows[0].id, version: Number(existing.rows[0].version) }; }
      const inserted = await client.query(
        `INSERT INTO memories(tenant_id,scope,owner_user_id,agent_id,memory_type,content,content_text,provenance,creation_reason,source_run_id,created_by,write_source,relevance_metadata,retention_until,idempotency_key)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'approved_tool',$12,$13,$14) RETURNING id,version`,
        [principal.tenantId, input.scope, input.scope === "user" ? principal.userId : null, input.scope === "agent" ? agentId : null, input.memoryType,
         JSON.stringify(input.content), input.content.text, JSON.stringify({ origin: "approved_memory_tool", sourceRunId: runId }), input.creationReason, runId, principal.userId, JSON.stringify(input.relevance), retentionUntil, idempotencyKey],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker','memory-tool','memory.created',$3)", [principal.tenantId, runId, JSON.stringify({ memoryId: inserted.rows[0].id, scope: input.scope, memoryType: input.memoryType, writeSource: "approved_tool" })]);
      await client.query("COMMIT"); return { memoryId: inserted.rows[0].id, version: Number(inserted.rows[0].version) };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async reconcileToolWrite(principal: Principal, idempotencyKey: string): Promise<{ memoryId: string; version: number } | null> {
    const result = await this.pool.query("SELECT id,version FROM memories WHERE tenant_id=$1 AND idempotency_key=$2", [principal.tenantId, idempotencyKey]);
    return result.rowCount ? { memoryId: result.rows[0].id, version: Number(result.rows[0].version) } : null;
  }

  private authorizeCreate(principal: Principal, input: CreateMemoryInput): void {
    if (input.scope !== "user" && !principal.roles.includes("memory_manager")) throw new AuthorizationError("Shared memory requires memory_manager role");
  }
  private authorizeManage(principal: Principal, row: Row): void {
    if (!(row.scope === "user" && row.owner_user_id === principal.userId) && !principal.roles.includes("memory_manager")) throw new AuthorizationError("Memory management is not authorized");
  }
  private async accessible(principal: Principal, id: string, includeDeleted = false): Promise<MemoryRecord> {
    const result = await this.pool.query(`SELECT * FROM memories WHERE tenant_id=$1 AND id=$2 ${includeDeleted ? "" : "AND deleted_at IS NULL"}`, [principal.tenantId, id]);
    if (!result.rowCount) throw new NotFoundError("Memory not found");
    const row = result.rows[0];
    if (row.scope === "user" && row.owner_user_id !== principal.userId && !principal.roles.includes("memory_manager")) throw new NotFoundError("Memory not found");
    return this.fromRow(row);
  }
  private async verifyReferences(client: Pick<pg.PoolClient, "query">, tenantId: string, agentId: string | null, runId: string | null): Promise<void> {
    if (agentId) await this.verifyAgent(client, tenantId, agentId);
    if (runId && !(await client.query("SELECT 1 FROM runs WHERE tenant_id=$1 AND id=$2", [tenantId, runId])).rowCount) throw new NotFoundError("Source run not found");
  }
  private async verifyAgent(client: Pick<pg.Pool, "query"> | Pick<pg.PoolClient, "query">, tenantId: string, agentId: string): Promise<void> {
    if (!(await client.query("SELECT 1 FROM agents WHERE tenant_id=$1 AND id=$2", [tenantId, agentId])).rowCount) throw new NotFoundError("Agent not found");
  }
  private validateRetention(value: string | null): void { if (value && new Date(value) <= new Date()) throw new ConflictError("Memory retention must be in the future"); }
  private fromRow(row: Row): MemoryRecord {
    return { id: row.id as string, tenantId: row.tenant_id as string, scope: row.scope as MemoryRecord["scope"], ownerUserId: row.owner_user_id as string | null,
      agentId: row.agent_id as string | null, memoryType: row.memory_type as MemoryRecord["memoryType"], content: row.content as MemoryRecord["content"], provenance: row.provenance as Record<string, unknown>,
      creationReason: row.creation_reason as string, sourceRunId: row.source_run_id as string | null, createdBy: row.created_by as string, writeSource: row.write_source as string,
      relevance: row.relevance_metadata as MemoryRecord["relevance"], retentionUntil: row.retention_until as Date | null, version: Number(row.version), correctedFromId: row.corrected_from_id as string | null,
      deletedAt: row.deleted_at as Date | null, createdAt: row.created_at as Date, updatedAt: row.updated_at as Date };
  }
}
