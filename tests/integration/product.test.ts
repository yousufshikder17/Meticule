import pg from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../../src/api/app.js";
import { DevelopmentHeaderAuthenticator } from "../../src/auth/authentication.js";

const { Pool } = pg; const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform" });
const tenant = "fc000000-0000-4000-8000-000000000001"; const admin = "fc000000-0000-4000-8000-000000000002"; const member = "fc000000-0000-4000-8000-000000000003"; const otherTenant = "fd000000-0000-4000-8000-000000000001";
const headers = (tenantId=tenant, userId=admin, roles="tenant_admin,usage_viewer,audit_viewer") => ({ "content-type": "application/json", "x-tenant-id": tenantId, "x-user-id": userId, "x-roles": roles });
const app = createApp(pool, new DevelopmentHeaderAuthenticator());
beforeEach(async () => { await pool.query("TRUNCATE tenant_memberships,operational_instances,structured_notes,audit_events,usage_records,approvals,model_attempts,tool_executions,worker_leases,checkpoints,steps,runs,agents CASCADE"); await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,roles) VALUES($1,$2,$3)", [tenant, admin, JSON.stringify(["tenant_admin","usage_viewer","audit_viewer"])]); });
afterAll(async () => pool.end());

describe("Stage 15 real product workflow", () => {
  it("serves a CSP-protected unauthenticated shell and requires auth for data", async () => {
    const page = await app.request("/"); expect(page.status).toBe(200); expect(page.headers.get("content-security-policy")).toContain("script-src 'self'"); expect(await page.text()).toContain("Durable Agent Platform");
    expect((await app.request("/app.js")).status).toBe(200); expect((await app.request("/agents")).status).toBe(401);
  });

  it("creates and lists agents/runs, exposes trace-cost evidence, and scopes every list to the tenant", async () => {
    const createdAgent = await app.request("/agents", { method: "POST", headers: headers(), body: JSON.stringify({ name: "Product Agent", systemInstructions: "Operate through the canonical runtime", model: { provider: "ollama", model: "configured-model" }, allowedTools: ["calculator"], maximumSteps: 5, tokenBudget: 1000, costBudgetMicrousd: 100, approvalPolicy: {}, outputSchema: null }) });
    expect(createdAgent.status).toBe(201); const agent = await createdAgent.json() as { id: string };
    const createdRun = await app.request("/agents/"+agent.id+"/runs", { method: "POST", headers: headers(), body: JSON.stringify({ goal: "Show durable status" }) }); expect(createdRun.status).toBe(201); const run = await createdRun.json() as { id: string };
    expect((await app.request("/agents", { headers: headers() }).then(r=>r.json()) as unknown[])).toHaveLength(1); expect((await app.request("/runs", { headers: headers() }).then(r=>r.json()) as unknown[])).toHaveLength(1);
    expect((await app.request("/runs/"+run.id+"/trace", { headers: headers() })).status).toBe(200); expect((await app.request("/usage", { headers: headers() })).status).toBe(200); expect((await app.request("/audit", { headers: headers() })).status).toBe(200);
    expect((await app.request("/runs", { headers: headers(otherTenant, member, "tenant_admin") }).then(r=>r.json()) as unknown[])).toHaveLength(0);
  });

  it("administers memberships through RLS without self-revocation or cross-tenant visibility", async () => {
    const created = await app.request("/memberships", { method: "POST", headers: headers(), body: JSON.stringify({ identityId: member, identityType: "user", roles: ["approval_reviewer"] }) }); expect(created.status).toBe(201);
    expect((await app.request("/memberships", { headers: headers() }).then(r=>r.json()) as unknown[])).toHaveLength(2);
    expect((await app.request("/memberships", { headers: headers(otherTenant, member, "tenant_admin") }).then(r=>r.json()) as unknown[])).toHaveLength(0);
    expect((await app.request("/memberships/"+admin+"/revoke", { method: "POST", headers: headers() })).status).toBe(409);
    expect((await app.request("/memberships/"+member+"/revoke", { method: "POST", headers: headers() })).status).toBe(200);
    expect(Number((await pool.query("SELECT count(*) AS count FROM audit_events WHERE tenant_id=$1 AND event_type LIKE 'membership.%'", [tenant])).rows[0].count)).toBe(2);
  });

  it("enforces product inspection roles and exposes an empty role-filtered approval inbox", async () => {
    expect((await app.request("/usage", { headers: headers(tenant, admin, "") })).status).toBe(403); expect((await app.request("/memberships", { headers: headers(tenant, admin, "audit_viewer") })).status).toBe(403);
    const approvals = await app.request("/approvals", { headers: headers(tenant, admin, "approval_reviewer") }); expect(approvals.status).toBe(200); expect(await approvals.json()).toEqual([]);
    const me = await app.request("/me", { headers: headers() }); expect(await me.json()).toMatchObject({ tenantId: tenant, userId: admin }); expect((await app.request("/providers", { headers: headers() })).status).toBe(200);
  });
});
