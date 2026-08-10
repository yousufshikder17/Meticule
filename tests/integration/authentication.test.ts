import pg from "pg";
import { SignJWT } from "jose";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../../src/api/app.js";
import { JwtAuthenticator } from "../../src/auth/authentication.js";
import { withTenantSession } from "../../src/db/tenant-session.js";

const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const secret = "stage-6-test-secret-with-at-least-32-characters";
const issuer = "durable-agent-platform-test";
const audience = "durable-agent-api-test";
const tenantA = "11111111-1111-4111-8111-111111111111";
const tenantB = "22222222-2222-4222-8222-222222222222";
const userA = "33333333-3333-4333-8333-333333333333";
const userB = "44444444-4444-4444-8444-444444444444";
const app = createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience));

async function addMembership(tenantId: string, identityId: string, roles: string[] = ["member"], identityType = "user") {
  await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,$3,$4)", [tenantId, identityId, identityType, JSON.stringify(roles)]);
}

async function token(options: { tenantId?: string; subject?: string; roles?: string[]; tokenIssuer?: string; tokenAudience?: string; expiresIn?: string | number; jti?: string; identityType?: "user" | "service" } = {}) {
  return new SignJWT({ tenant_id: options.tenantId ?? tenantA, roles: options.roles ?? ["member"], identity_type: options.identityType ?? "user" })
    .setProtectedHeader({ alg: "HS256" }).setSubject(options.subject ?? userA).setIssuer(options.tokenIssuer ?? issuer)
    .setAudience(options.tokenAudience ?? audience).setJti(options.jti ?? crypto.randomUUID()).setIssuedAt()
    .setExpirationTime(options.expiresIn ?? "5m").sign(new TextEncoder().encode(secret));
}

beforeEach(async () => {
  await pool.query("TRUNCATE revoked_tokens,tenant_memberships,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE");
});
afterAll(async () => pool.end());

describe("Stage 6 authenticated tenant boundary", () => {
  it("requires a valid signed token and ignores development identity headers", async () => {
    expect((await app.request("/tools", { headers: { "x-tenant-id": tenantA, "x-user-id": userA } })).status).toBe(401);
    expect((await app.request("/tools", { headers: { authorization: "Bearer malformed" } })).status).toBe(401);
    await addMembership(tenantA, userA);
    expect((await app.request("/tools", { headers: { authorization: `Bearer ${await token({ tokenIssuer: "wrong" })}` } })).status).toBe(401);
    expect((await app.request("/tools", { headers: { authorization: `Bearer ${await token({ tokenAudience: "wrong" })}` } })).status).toBe(401);
    expect((await app.request("/tools", { headers: { authorization: `Bearer ${await token({ expiresIn: -1 })}` } })).status).toBe(401);
  });

  it("requires active membership, intersects roles, and honors revocation", async () => {
    const authenticator = new JwtAuthenticator(pool, secret, issuer, audience);
    const jti = crypto.randomUUID();
    const signed = await token({ roles: ["member", "ungranted"], jti });
    const request = new Request("http://local/tools", { headers: { authorization: `Bearer ${signed}` } });
    await expect(authenticator.authenticate(request)).rejects.toThrow(/membership/);
    await addMembership(tenantA, userA, ["member", "approval_reviewer"]);
    expect((await authenticator.authenticate(request)).roles).toEqual(["member"]);
    await pool.query("INSERT INTO revoked_tokens(issuer,token_id,expires_at,reason) VALUES($1,$2,now()+interval '10 minutes','test revocation')", [issuer, jti]);
    await expect(authenticator.authenticate(request)).rejects.toThrow(/revoked/);
  });

  it("supports signed service identities with explicit tenant membership", async () => {
    await addMembership(tenantA, userA, ["worker"], "service");
    const signed = await token({ roles: ["worker"], identityType: "service" });
    expect((await new JwtAuthenticator(pool, secret, issuer, audience).authenticate(new Request("http://local", { headers: { authorization: `Bearer ${signed}` } }))).roles).toEqual(["worker"]);
  });

  it("uses RLS to constrain queries even when an application query omits a tenant predicate", async () => {
    await pool.query(`INSERT INTO agents(tenant_id,created_by,name,system_instructions,model_config,allowed_tools,maximum_steps,token_budget,cost_budget_microusd,approval_policy,composition_config)
      VALUES($1,$3,'tenant-a','test','{}','[]',5,10,10,'{}','{}'),($2,$4,'tenant-b','test','{}','[]',5,10,10,'{}','{}')`, [tenantA, tenantB, userA, userB]);
    await withTenantSession(pool, tenantA, async (database) => {
      const visible = await database.query("SELECT tenant_id FROM agents ORDER BY tenant_id");
      expect(visible.rows).toEqual([{ tenant_id: tenantA }]);
      await expect(database.query(`INSERT INTO agents(tenant_id,created_by,name,system_instructions,model_config,allowed_tools,maximum_steps,token_budget,cost_budget_microusd,approval_policy,composition_config)
        VALUES($1,$2,'cross-tenant','test','{}','[]',5,10,10,'{}','{}')`, [tenantB, userA])).rejects.toThrow(/row-level security/);
    });
  });

  it("prevents a tenant token from reading another tenant's API resource", async () => {
    await addMembership(tenantA, userA);
    await addMembership(tenantB, userB);
    const create = await app.request("/agents", { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${await token()}` }, body: JSON.stringify({
      name: "tenant-a-agent", systemInstructions: "test", model: { provider: "disabled", model: "test" }, allowedTools: [], maximumSteps: 5,
      tokenBudget: 10, costBudgetMicrousd: 10, approvalPolicy: {}, outputSchema: null,
    }) });
    expect(create.status).toBe(201);
    const agent = await create.json() as { id: string };
    const tenantBToken = await token({ tenantId: tenantB, subject: userB });
    expect((await app.request(`/agents/${agent.id}`, { headers: { authorization: `Bearer ${tenantBToken}` } })).status).toBe(404);
  });
});
