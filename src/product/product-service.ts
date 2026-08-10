import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import type { z } from "zod";
import type { AuditQuerySchema, CreateMembershipSchema, RunListQuerySchema } from "./product-schema.js";

export class ProductService {
  constructor(private readonly database: pg.Pool) {}
  private role(principal: Principal, ...roles: string[]): void { if (!roles.some((role) => principal.roles.includes(role))) throw new AuthorizationError(`One of these roles is required: ${roles.join(", ")}`); }

  async agents(principal: Principal): Promise<unknown[]> { return (await this.database.query("SELECT id,name,model_config,allowed_tools,maximum_steps,token_budget,cost_budget_microusd,version,created_at,updated_at FROM agents WHERE tenant_id=$1 ORDER BY updated_at DESC,id LIMIT 200", [principal.tenantId])).rows; }
  async runs(principal: Principal, query: z.infer<typeof RunListQuerySchema>): Promise<unknown[]> { return (await this.database.query(`SELECT id,agent_id,goal,status,current_step,input_tokens,output_tokens,cost_microusd,final_output,error_details,created_at,updated_at FROM runs WHERE tenant_id=$1 AND ($2::text IS NULL OR status::text=$2) ORDER BY created_at DESC,id LIMIT $3`, [principal.tenantId, query.status ?? null, query.limit])).rows; }
  async approvals(principal: Principal): Promise<unknown[]> { return (await this.database.query(`SELECT id,run_id,step_id,tool_name,validated_arguments,risk_explanation,requester_id,required_approver_role,decision,created_at FROM approvals WHERE tenant_id=$1 AND decision='pending' AND required_approver_role=ANY($2::text[]) ORDER BY created_at,id LIMIT 200`, [principal.tenantId, principal.roles])).rows; }
  async usage(principal: Principal): Promise<unknown> { this.role(principal, "usage_viewer", "tenant_admin"); return (await this.database.query(`SELECT coalesce(sum(input_tokens),0)::bigint AS input_tokens,coalesce(sum(output_tokens),0)::bigint AS output_tokens,coalesce(sum(cost_microusd),0)::bigint AS cost_microusd,count(*)::int AS records FROM usage_records WHERE tenant_id=$1`, [principal.tenantId])).rows[0]; }
  async audit(principal: Principal, query: z.infer<typeof AuditQuerySchema>): Promise<unknown[]> { this.role(principal, "audit_viewer", "tenant_admin"); return (await this.database.query("SELECT id,run_id,actor_type,actor_id,event_type,details,created_at FROM audit_events WHERE tenant_id=$1 ORDER BY created_at DESC,id LIMIT $2", [principal.tenantId, query.limit])).rows; }
  async memberships(principal: Principal): Promise<unknown[]> { this.role(principal, "tenant_admin"); return (await this.database.query("SELECT identity_id,identity_type,roles,status,created_at,updated_at FROM tenant_memberships WHERE tenant_id=$1 ORDER BY created_at,identity_id", [principal.tenantId])).rows; }
  async addMembership(principal: Principal, input: z.infer<typeof CreateMembershipSchema>): Promise<unknown> {
    this.role(principal, "tenant_admin");
    try { const result = await this.database.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,$3,$4) RETURNING identity_id,identity_type,roles,status,created_at,updated_at", [principal.tenantId, input.identityId, input.identityType, JSON.stringify(input.roles)]); await this.database.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'membership.created',$3)", [principal.tenantId, principal.userId, JSON.stringify({ identityId: input.identityId, identityType: input.identityType, roles: input.roles })]); return result.rows[0]; }
    catch (error) { if ((error as { code?: string }).code === "23505") throw new ConflictError("Membership already exists"); throw error; }
  }
  async revokeMembership(principal: Principal, identityId: string): Promise<unknown> {
    this.role(principal, "tenant_admin"); if (identityId === principal.userId) throw new ConflictError("An administrator cannot revoke their own active membership");
    const result = await this.database.query("UPDATE tenant_memberships SET status='revoked',updated_at=now() WHERE tenant_id=$1 AND identity_id=$2 AND status='active' RETURNING identity_id,identity_type,roles,status,created_at,updated_at", [principal.tenantId, identityId]); if (!result.rowCount) throw new NotFoundError("Active membership not found");
    await this.database.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'membership.revoked',$3)", [principal.tenantId, principal.userId, JSON.stringify({ identityId })]); return result.rows[0];
  }
}
