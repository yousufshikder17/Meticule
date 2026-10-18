import { randomUUID } from "node:crypto";
import { SignJWT } from "jose";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { createApp } from "../../src/api/app.js";
import { JwtAuthenticator } from "../../src/auth/authentication.js";
import { approver, manager, outsider, pool, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

// The MLOps routes through the real app: signed JWTs, tenant memberships and the API's RLS tenant session.
const secret = "public-mlops-test-only-secret-at-least-32-characters", issuer = "public-mlops", audience = "public-mlops-api";
const service: Principal = { ...approver, userId: "66666666-6666-4666-8666-666666666666" };
const memberships: [Principal, "user" | "service"][] = [[manager, "user"], [approver, "user"], [viewer, "user"], [outsider, "user"], [service, "service"]];
const jwt = (p: Principal, type: "user" | "service" = "user") => new SignJWT({ tenant_id: p.tenantId, identity_type: type, roles: p.roles }).setProtectedHeader({ alg: "HS256" }).setSubject(p.userId).setIssuer(issuer).setAudience(audience).setJti(randomUUID()).setIssuedAt().setExpirationTime("5m").sign(new TextEncoder().encode(secret));

beforeEach(async () => {
  await truncate();
  await pool.query("DELETE FROM tenant_memberships WHERE identity_id=ANY($1::uuid[])", [memberships.map(([p]) => p.userId)]);
  for (const [p, type] of memberships) await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,$3,$4)", [p.tenantId, p.userId, type, JSON.stringify(p.roles)]);
});
afterAll(async () => { await pool.query("DELETE FROM tenant_memberships WHERE identity_id=ANY($1::uuid[])", [memberships.map(([p]) => p.userId)]); await pool.end(); });

describe("public authorization and tenancy for the MLOps routes", () => {
  it("lets members read, keeps writes for tenant_admin, isolates tenants and keeps approvals with another person", () => withStore(async store => {
    const f = await trainedModel(store);
    const app = createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience), null, null, null, undefined, null, { enabled: true, backends: f.backends });
    const tokens = { admin: await jwt(manager), approver: await jwt(approver), member: await jwt(viewer), other: await jwt(outsider), service: await jwt(service, "service") };
    const call = (path: string, token: string, body?: unknown, method = body === undefined ? "GET" : "POST") => app.request(path, { method, headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    const ok = async (response: Response, status: number) => { expect(response.status, await response.clone().text()).toBe(status); return response.headers.get("content-type")?.includes("json") ? response.json() : response.text(); };

    // Tracking and lineage: members read; only tenant_admin writes; other tenants see nothing.
    expect((await call("/ml/experiments", tokens.admin)).status).toBe(200);
    await ok(await call("/ml/experiments", tokens.member, { name: "member" }), 403);
    await ok(await call(`/ml/experiments/${f.experiment.id}`, tokens.member, { description: "x" }, "PATCH"), 403);
    await ok(await call(`/ml/experiments/${f.experiment.id}`, tokens.admin, { description: "tracked" }, "PATCH"), 200);
    expect((await ok(await call(`/ml/experiments/${f.experiment.id}`, tokens.member), 200)).experiment.description).toBe("tracked");
    expect(await ok(await call("/ml/experiments", tokens.other), 200)).toEqual([]);
    await ok(await call(`/ml/experiments/${f.experiment.id}`, tokens.other), 404);
    await ok(await call(`/ml/training-jobs/${f.job.id}/reproduce`, tokens.member, {}), 403);
    await ok(await call(`/ml/lineage/training-runs/${f.attempt.id}`, tokens.member), 200);
    await ok(await call(`/ml/lineage/training-runs/${f.attempt.id}`, tokens.other), 404);

    // Governance: the registrant cannot approve, a service identity cannot approve, another tenant_admin can.
    const registry = await ok(await call("/ml/registry", tokens.admin, { name: "governed" }), 201);
    await ok(await call(`/ml/registry/${registry.id}/policy`, tokens.member, { requireApproval: true }, "PUT"), 403);
    await ok(await call(`/ml/registry/${registry.id}/policy`, tokens.admin, { requireApproval: true }, "PUT"), 201);
    const version = await ok(await call(`/ml/registry/${registry.id}/versions`, tokens.admin, { trainingRunId: f.attempt.id }), 201);
    expect(version.registered_by).toBe(manager.userId);
    expect((await ok(await call(`/ml/model-versions/${version.id}/transition`, tokens.admin, { status: "READY" }), 409)).reasons).toEqual([{ code: "approval_required" }]);
    await ok(await call(`/ml/model-versions/${version.id}/approvals`, tokens.admin, { decision: "APPROVED" }), 409);
    await ok(await call(`/ml/model-versions/${version.id}/approvals`, tokens.service, { decision: "APPROVED" }), 409);
    await ok(await call(`/ml/model-versions/${version.id}/approvals`, tokens.member, { decision: "APPROVED" }), 403);
    await ok(await call(`/ml/model-versions/${version.id}/approvals`, tokens.other, { decision: "APPROVED" }), 404);
    await ok(await call(`/ml/model-versions/${version.id}/approvals`, tokens.approver, { decision: "APPROVED", note: "independent review" }), 201);
    await ok(await call(`/ml/model-versions/${version.id}/transition`, tokens.member, { status: "READY" }), 403);
    await ok(await call(`/ml/model-versions/${version.id}/transition`, tokens.admin, { status: "READY" }), 200);
    const events = await ok(await call(`/ml/model-versions/${version.id}/events`, tokens.member), 200);
    expect(events.map((e: { event: string }) => e.event)).toEqual(["registered", "transition_refused", "approved", "transitioned"]);
    await ok(await call(`/ml/model-versions/${version.id}/events`, tokens.other), 404);

    // Deployments and monitoring: members read and predict; writes and drift evaluations need tenant_admin.
    await ok(await call("/ml/deployments", tokens.member, { name: "nope" }), 403);
    const deployment = await ok(await call("/ml/deployments", tokens.admin, { name: "serving" }), 201);
    await ok(await call(`/ml/deployments/${deployment.id}/revisions`, tokens.admin, { modelVersionId: version.id }), 201);
    await ok(await call(`/ml/deployments/${deployment.id}/activate`, tokens.member, {}), 403);
    await ok(await call(`/ml/deployments/${deployment.id}/activate`, tokens.admin, {}), 200);
    const prediction = await ok(await call(`/ml/deployments/${deployment.id}/predict`, tokens.member, { rows: [{ x: 1 }, { x: 2 }] }), 200);
    expect(prediction.output).toEqual([3, 5]);
    await ok(await call(`/ml/deployments/${deployment.id}/predict`, tokens.other, { rows: [{ x: 1 }] }), 404);
    await ok(await call(`/ml/predictions/${prediction.id}/feedback`, tokens.member, { labels: [3, 5] }), 403);
    await ok(await call(`/ml/predictions/${prediction.id}/feedback`, tokens.admin, { labels: [3, 5] }), 201);
    expect((await ok(await call(`/ml/deployments/${deployment.id}/monitoring`, tokens.member), 200)).versions[0]).toMatchObject({ requests: 1, groundTruth: { labelledRows: 2, mae: 0 } });
    await ok(await call(`/ml/deployments/${deployment.id}/monitoring`, tokens.other), 404);
    await ok(await call(`/ml/deployments/${deployment.id}/drift-evaluations`, tokens.member, {}), 403);
    await ok(await call(`/ml/deployments/${deployment.id}/drift-evaluations`, tokens.admin, {}), 201);
    await ok(await call(`/ml/lineage/deployments/${deployment.id}`, tokens.other), 404);

    // Optional features stay off unless configured, and the private operational routes are not part of the public surface.
    for (const [path, body] of [[`/ml/training-runs/${f.attempt.id}/mlflow-export`, {}], [`/ml/model-versions/${version.id}/onnx-exports`, {}], ["/ml/reports", { kind: "training_curves" }]] as const)
      expect((await call(path, tokens.admin, body)).status, path).toBe(503);
    await ok(await call(`/ml/deployments/${deployment.id}/monitor`, tokens.admin), 404);
    await ok(await call(`/ml/endpoints/${randomUUID()}/monitor`, tokens.admin), 404);
  }), 60_000);

  it("leaves the pruning function to the database owner", async () => {
    // Retention runs as the owner role; the API role has no EXECUTE on it.
    const c = await pool.connect();
    try { await c.query("BEGIN"); await c.query("SET LOCAL ROLE durable_agent_api"); await expect(c.query("SELECT ml_prune_monitoring(interval '1 day')")).rejects.toThrow(/permission denied/); }
    finally { await c.query("ROLLBACK"); c.release(); }
    expect(Number((await pool.query("SELECT ml_prune_monitoring(interval '0 seconds') AS n")).rows[0].n)).toBeGreaterThanOrEqual(0);
  });
});
