import { createHash } from "node:crypto";
import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { CreateSkillSchema, RevokeSkillSchema, type CreateSkillInput } from "./skill-schema.js";

type Row = Record<string, unknown>;
export interface SkillRecord { id: string; tenantId: string; logicalId: string; version: number; name: string; instructions: string; allowedTools: string[]; provenance: unknown; contentHash: string; createdBy: string; supersedesId: string | null; isCurrent: boolean; revokedAt: Date | null; createdAt: Date }
const hash = (value: string): string => createHash("sha256").update(value, "utf8").digest("hex");

export class SkillService {
  constructor(private readonly pool: pg.Pool) {}

  async create(principal: Principal, raw: unknown, supersedesId?: string): Promise<SkillRecord> {
    this.manage(principal); const input = CreateSkillSchema.parse(raw); const client = await this.pool.connect();
    try {
      await client.query("BEGIN"); let logicalId = crypto.randomUUID(); let version = 1; let priorId: string | null = null;
      if (supersedesId) {
        const prior = await client.query("SELECT * FROM skills WHERE tenant_id=$1 AND id=$2 AND revoked_at IS NULL FOR UPDATE", [principal.tenantId, supersedesId]);
        if (!prior.rowCount) throw new NotFoundError("Skill not found");
        logicalId = prior.rows[0].logical_id; version = Number(prior.rows[0].version) + 1; priorId = prior.rows[0].id;
        await client.query("UPDATE skills SET is_current=false WHERE tenant_id=$1 AND logical_id=$2 AND is_current", [principal.tenantId, logicalId]);
      }
      const canonical = JSON.stringify({ name: input.name, instructions: input.instructions, allowedTools: [...input.allowedTools].sort(), provenance: input.provenance });
      const inserted = await client.query(
        `INSERT INTO skills(tenant_id,logical_id,version,name,instructions,allowed_tools,provenance,content_hash,created_by,supersedes_id,is_current)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,true) RETURNING *`,
        [principal.tenantId, logicalId, version, input.name, input.instructions, JSON.stringify(input.allowedTools), JSON.stringify(input.provenance), hash(canonical), principal.userId, priorId],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'skill.version_created',$3)", [principal.tenantId, principal.userId, JSON.stringify({ skillId: inserted.rows[0].id, logicalId, version, allowedTools: input.allowedTools })]);
      await client.query("COMMIT"); return this.fromRow(inserted.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); if ((error as { code?: string }).code === "23505") throw new ConflictError("An active skill with this name already exists"); throw error; }
    finally { client.release(); }
  }

  async list(principal: Principal): Promise<SkillRecord[]> {
    return (await this.pool.query("SELECT * FROM skills WHERE tenant_id=$1 AND is_current AND revoked_at IS NULL ORDER BY name,id", [principal.tenantId])).rows.map((row) => this.fromRow(row));
  }
  async versions(principal: Principal, id: string): Promise<SkillRecord[]> {
    const selected = await this.pool.query("SELECT logical_id FROM skills WHERE tenant_id=$1 AND id=$2", [principal.tenantId, id]); if (!selected.rowCount) throw new NotFoundError("Skill not found");
    return (await this.pool.query("SELECT * FROM skills WHERE tenant_id=$1 AND logical_id=$2 ORDER BY version DESC", [principal.tenantId, selected.rows[0].logical_id])).rows.map((row) => this.fromRow(row));
  }
  async revoke(principal: Principal, id: string, raw: unknown): Promise<SkillRecord> {
    this.manage(principal); const input = RevokeSkillSchema.parse(raw);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("UPDATE skills SET is_current=false,revoked_at=COALESCE(revoked_at,now()),revoked_by=COALESCE(revoked_by,$1),revocation_reason=COALESCE(revocation_reason,$2) WHERE tenant_id=$3 AND id=$4 RETURNING *", [principal.userId, input.reason, principal.tenantId, id]);
      if (!result.rowCount) throw new NotFoundError("Skill not found");
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'skill.revoked',$3)", [principal.tenantId, principal.userId, JSON.stringify({ skillId: id, reason: input.reason })]);
      await client.query("COMMIT"); return this.fromRow(result.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async select(tenantId: string, ids: string[], agentAllowedTools: string[], maximumTokens: number): Promise<SkillRecord[]> {
    if (!ids.length) return [];
    const rows = (await this.pool.query("SELECT * FROM skills WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND is_current AND revoked_at IS NULL", [tenantId, ids])).rows.map((row) => this.fromRow(row));
    const byId = new Map(rows.map((row) => [row.id, row])); const selected = ids.map((id) => byId.get(id));
    if (selected.some((entry) => !entry)) throw new ConflictError("Configured skill is missing, superseded, revoked, or outside the tenant");
    const tools = new Set(agentAllowedTools); let tokens = 0;
    for (const skill of selected as SkillRecord[]) { if (skill.allowedTools.some((tool) => !tools.has(tool))) throw new ConflictError(`Skill ${skill.name} cannot expand the agent tool allowlist`); tokens += Math.ceil(skill.instructions.length / 4); }
    if (tokens > maximumTokens) throw new ConflictError("Configured skills exceed the agent skill context budget");
    return selected as SkillRecord[];
  }

  private manage(principal: Principal): void { if (!principal.roles.includes("skill_manager")) throw new AuthorizationError("Skill management requires skill_manager role"); }
  private fromRow(row: Row): SkillRecord { return { id: row.id as string, tenantId: row.tenant_id as string, logicalId: row.logical_id as string, version: Number(row.version), name: row.name as string, instructions: row.instructions as string, allowedTools: row.allowed_tools as string[], provenance: row.provenance, contentHash: row.content_hash as string, createdBy: row.created_by as string, supersedesId: row.supersedes_id as string | null, isCurrent: row.is_current as boolean, revokedAt: row.revoked_at as Date | null, createdAt: row.created_at as Date }; }
}
