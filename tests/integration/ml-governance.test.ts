import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { withTenantSession } from "../../src/db/tenant-session.js";
import { GovernanceRefusal, MlGovernanceService } from "../../src/ml/governance.js";
import { MlLineageService } from "../../src/ml/lineage.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { createApp } from "../../src/api/app.js";
import { approver, dropBlob, json, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

beforeEach(truncate);
afterAll(() => pool.end());

const refusal = (p: Promise<unknown>) => p.then(() => { throw new Error("expected a refusal"); }, (e: unknown) => { if (!(e instanceof GovernanceRefusal)) throw e; return e.reasons.map(r => r.code).sort(); });

describe("Phase 5 registry governance", () => {
  it("enforces metric thresholds and approval with separation of duties, recording every decision", () => withStore(async store => {
    const f = await trainedModel(store, undefined, { alpha: 5 }); // mae = 0.21 on every partition
    const governance = new MlGovernanceService(pool, store);
    const registry = await f.models.registry(manager, { name: "governed" });
    const policy = await governance.setPolicy(manager, registry.id, { minimumMetrics: [{ name: "mae", partition: "test", threshold: .1 }, { name: "r2", partition: "test", threshold: .5 }], requireApproval: true });
    expect(policy.version).toBe(1);
    const version = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id });
    expect(version.registered_by).toBe(manager.userId);
    expect(await refusal(f.models.transition(manager, version.id, "READY", "first try"))).toEqual(["approval_required", "policy_metric_missing", "policy_metric_threshold"]);
    const readiness = await governance.readiness(viewer, version.id);
    expect(readiness).toMatchObject({ policyVersion: 1, eligibleForReady: false });
    expect(readiness.reasons.find(r => r.code === "policy_metric_threshold")!.detail).toMatchObject({ name: "mae", threshold: .1, value: expect.closeTo(.21), direction: "lower" });
    // The registrant cannot approve; the database enforces the same rule.
    await expect(governance.approve(manager, version.id, { decision: "APPROVED" })).rejects.toThrow(/registrant/);
    await expect(pool.query("INSERT INTO ml_model_approvals(tenant_id,model_version_id,decision,approver) VALUES($1,$2,'APPROVED',$3)", [manager.tenantId, version.id, manager.userId])).rejects.toThrow(/registrant/);
    await expect(governance.approve(viewer, version.id, { decision: "APPROVED" })).rejects.toThrow(/role/);
    // Public: identity is the authenticated subject, so extra roles change nothing, and a service identity that a tenant
    // admin added cannot approve either, through the service or directly in the database.
    await expect(governance.approve({ ...manager, roles: ["tenant_admin", "reviewer"] }, version.id, { decision: "APPROVED" })).rejects.toThrow(/registrant/);
    const service = { ...approver, userId: "66666666-6666-4666-8666-666666666666" };
    await pool.query("DELETE FROM tenant_memberships WHERE tenant_id=$1 AND identity_id=$2", [service.tenantId, service.userId]);
    await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,'service','[\"tenant_admin\"]')", [service.tenantId, service.userId]);
    try {
      await expect(governance.approve(service, version.id, { decision: "APPROVED" })).rejects.toThrow(/Service identities/);
      await expect(pool.query("INSERT INTO ml_model_approvals(tenant_id,model_version_id,decision,approver) VALUES($1,$2,'APPROVED',$3)", [manager.tenantId, version.id, service.userId])).rejects.toThrow(/service identity/);
    } finally { await pool.query("DELETE FROM tenant_memberships WHERE tenant_id=$1 AND identity_id=$2", [service.tenantId, service.userId]); }
    await governance.approve(approver, version.id, { decision: "APPROVED", note: "reviewed" });
    await governance.setPolicy(manager, registry.id, { minimumMetrics: [{ name: "mae", partition: "test", threshold: .25 }], requireApproval: true });
    // A later rejection withdraws the approval; a new approval restores it.
    await governance.approve(approver, version.id, { decision: "REJECTED", note: "regression found" });
    expect(await refusal(f.models.transition(manager, version.id, "READY"))).toEqual(["approval_rejected"]);
    await governance.approve(approver, version.id, { decision: "APPROVED" });
    const ready = await f.models.transition(manager, version.id, "READY", "meets policy v2");
    expect(ready.status).toBe("READY");
    await expect(governance.approve(approver, version.id, { decision: "APPROVED" })).rejects.toThrow(/REGISTERED/);
    const events = await governance.events(viewer, version.id);
    expect(events.map(e => e.event)).toEqual(["registered", "transition_refused", "approved", "approval_rejected", "transition_refused", "approved", "transitioned"]);
    const last = events.at(-1)!;
    expect(last).toMatchObject({ from_status: "REGISTERED", to_status: "READY", actor: manager.userId, reason: "meets policy v2", details: { policyVersion: 2 } });
    expect(events[1].details.reasons.map((r: { code: string }) => r.code)).toContain("policy_metric_threshold");
    // Lifecycle history is immutable and part of the version's lineage.
    await expect(pool.query("UPDATE ml_model_events SET reason='edited' WHERE id=$1", [last.id])).rejects.toThrow();
    await expect(pool.query("DELETE FROM ml_model_events WHERE id=$1", [last.id])).rejects.toThrow();
    expect((await new MlLineageService(pool, store).modelVersion(manager, version.id)).lifecycle.map(e => e.event).at(-1)).toBe("transitioned");
    // Retirement is recorded by the trigger even without a reason.
    await f.models.transition(manager, version.id, "RETIRED");
    expect((await governance.events(manager, version.id)).at(-1)).toMatchObject({ event: "transitioned", from_status: "READY", to_status: "RETIRED", reason: null });
    await expect(governance.events(outsider, version.id)).rejects.toThrow(/not found/);
    await expect(governance.setPolicy(outsider, registry.id, {})).rejects.toThrow(/not found/);
    await expect(governance.setPolicy(manager, registry.id, { minimumMetrics: [], unknown: true })).rejects.toThrow();
  }));

  it("keeps Phase 1 promotion without a policy, refuses a damaged artifact, and persists HTTP refusals", () => withStore(async store => {
    const f = await trainedModel(store);
    const registry = await f.models.registry(manager, { name: "plain" });
    const first = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id });
    expect((await f.models.transition(manager, first.id, "READY")).status).toBe("READY");
    // Public registration is idempotent per attempt, so the second version comes from a second attempt (its own artifact).
    const retrain = await f.training.queue(manager, { experimentId: f.experiment.id, spec: { ...f.spec, hyperparameters: { alpha: 4 } } }); await f.worker.tick();
    const second = await f.models.register(manager, registry.id, { trainingRunId: (await f.training.getJob(manager, retrain.id)).attempts[0].id });
    const sha = (await pool.query("SELECT content->>'sha256' AS s FROM ml_artifacts WHERE id=$1", [second.artifact_id])).rows[0].s;
    await dropBlob(manager.tenantId, sha);
    expect(await refusal(f.models.transition(manager, second.id, "READY"))).toEqual(["artifact_integrity_failed"]);
    // Through the API's tenant session (one transaction per request) the refusal is answered with 409 and still recorded.
    const real = await createApp(pool, { authenticate: async () => manager }, null, null, null, undefined, null, { enabled: true, backends: f.backends }).request(`/ml/model-versions/${second.id}/transition`, json({ status: "READY" }));
    expect(real.status).toBe(409); expect((await real.json()).reasons).toEqual([{ code: "artifact_integrity_failed" }]);
    // A service caller inside a tenant session that lets the refusal escape rolls the whole session back, event included.
    await expect(withTenantSession(pool, manager.tenantId, db => new MlModelService(db, store, f.backends).transition(manager, second.id, "READY"))).rejects.toBeInstanceOf(GovernanceRefusal);
    const app = testApp(store, f.backends);
    const response = await app.request(`/ml/model-versions/${second.id}/transition`, json({ status: "READY", reason: "retry" }));
    expect(response.status).toBe(409); expect((await response.json()).reasons).toEqual([{ code: "artifact_integrity_failed" }]);
    expect((await new MlGovernanceService(pool, store).events(manager, second.id)).filter(e => e.event === "transition_refused")).toHaveLength(3);
    expect((await app.request(`/ml/model-versions/${second.id}/transition`, json({ status: "REJECTED", reason: "corrupt" }))).status).toBe(200);
    expect((await app.request(`/ml/model-versions/${second.id}/readiness`)).status).toBe(200);
    expect((await testApp(store, f.backends, viewer).request(`/ml/registry/${registry.id}/policy`, json({ requireApproval: true }, "PUT"))).status).toBe(403);
  }));
});
