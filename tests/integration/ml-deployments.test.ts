import { randomUUID } from "node:crypto";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { MlDeploymentService } from "../../src/ml/deployments.js";
import { GovernanceRefusal } from "../../src/ml/governance.js";
import { MlLineageService } from "../../src/ml/lineage.js";
import { MlOnnxService } from "../../src/ml/onnx.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { json, manager, outsider, pool, testApp, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

beforeEach(truncate);
afterAll(() => pool.end());
const signal = () => new AbortController().signal;
const codes = (p: Promise<unknown>) => p.then(() => { throw new Error("expected a refusal"); }, (e: unknown) => { if (!(e instanceof GovernanceRefusal)) throw e; return e.reasons.map(r => r.code); });

describe("Phase 5 deployment lifecycle", () => {
  it("validates, activates, updates, rolls back and deactivates version-pinned revisions", () => withStore(async store => {
    const f = await trainedModel(store);
    const registry = await f.models.registry(manager, { name: "served" });
    const v1 = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id });
    const deployments = new MlDeploymentService(pool, store, f.backends);
    const d = await deployments.create(manager, { name: "pricing" });
    expect(d.status).toBe("CREATED");
    await expect(deployments.create(manager, { name: "pricing" })).rejects.toThrow(/already used/);
    await expect(deployments.activate(manager, d.id, {})).rejects.toThrow(/CREATED cannot be activated/);
    // An unverified (REGISTERED) model cannot pass validation; with nothing deployable the deployment is FAILED.
    expect(await codes(deployments.addRevision(manager, d.id, { modelVersionId: v1.id }, signal()))).toEqual(["model_not_ready"]);
    expect((await deployments.get(manager, d.id)).deployment.status).toBe("FAILED");
    await f.models.transition(manager, v1.id, "READY");
    const r2 = await deployments.addRevision(manager, d.id, { modelVersionId: v1.id, reason: "first release" }, signal());
    expect(r2).toMatchObject({ revision: 2, status: "READY", runtime: "native" });
    expect(r2.validation.smoke).toMatchObject({ rows: 8, runtime: "native" });
    expect((await deployments.get(manager, d.id)).deployment.status).toBe("READY");
    const active = await deployments.activate(manager, d.id, {});
    expect(active).toMatchObject({ status: "ACTIVE", active_revision: 2 });
    const first = await deployments.predict(viewer, d.id, [{ x: 10 }], signal());
    expect(first).toMatchObject({ status: "COMPLETED", output: [21], revision: 2, model_version_id: v1.id });

    // Version update: a second model joins as revision 3 while revision 2 keeps serving.
    const g = await trainedModel(store, undefined, { alpha: 4 });
    const v2 = await f.models.register(manager, registry.id, { trainingRunId: g.attempt.id }); await f.models.transition(manager, v2.id, "READY");
    expect((await deployments.addRevision(manager, d.id, { modelVersionId: v2.id }, signal())).status).toBe("READY");
    expect((await deployments.get(manager, d.id)).deployment).toMatchObject({ status: "ACTIVE", active_revision: 2 });
    const updated = await deployments.activate(manager, d.id, { revision: 3, reason: "v2" });
    expect(updated.active_revision).toBe(3);
    expect((await pool.query("SELECT status FROM ml_endpoints WHERE id=$1", [active.active_endpoint_id])).rows[0].status).toBe("DISABLED");
    expect((await deployments.predict(manager, d.id, [{ x: 1 }], signal())).model_version_id).toBe(v2.id);
    await expect(deployments.activate(manager, d.id, { revision: 3 })).rejects.toThrow(/already active/);
    const rolled = await deployments.rollback(manager, d.id, { reason: "regression in v2" });
    expect(rolled).toMatchObject({ status: "ACTIVE", active_revision: 2 });
    expect((await deployments.predict(manager, d.id, [{ x: 1 }], signal())).model_version_id).toBe(v1.id);

    const inactive = await deployments.deactivate(manager, d.id, { reason: "maintenance" });
    expect(inactive).toMatchObject({ status: "INACTIVE", active_revision: null, active_endpoint_id: null });
    await expect(deployments.predict(manager, d.id, [{ x: 1 }], signal())).rejects.toThrow(/not active/);
    expect((await deployments.activate(manager, d.id, { revision: 3 })).status).toBe("ACTIVE");
    // Retiring the serving model disables its endpoint, which deactivates the deployment (database trigger).
    await f.models.transition(manager, v2.id, "RETIRED", "superseded");
    expect((await deployments.get(manager, d.id)).deployment).toMatchObject({ status: "INACTIVE", active_endpoint_id: null });
    expect(await codes(deployments.activate(manager, d.id, { revision: 3 }))).toEqual(["model_not_ready"]);
    const history = (await deployments.get(viewer, d.id)).events;
    expect(history.map(e => e.event)).toEqual(["created", "revision_added", "validation_failed", "revision_added", "validated", "activated", "revision_added", "validated", "activated", "rolled_back", "deactivated", "activated", "deactivated", "refused"]);
    expect(history.find(e => e.event === "rolled_back")).toMatchObject({ revision: 2, actor: manager.userId, reason: "regression in v2", details: { previousRevision: 3 } });
    expect(history.filter(e => e.event === "deactivated").map(e => e.reason)).toEqual(["maintenance", "superseded"]);
    expect((await pool.query("SELECT details->>'action' AS a FROM ml_model_events WHERE model_version_id=$1 AND event='deployment_changed' ORDER BY created_at", [v1.id])).rows.map(r => r.a)).toEqual(["activated", "rolled_back", "deactivated"]);

    // The database refuses lifecycle shortcuts and edits of validated revisions.
    await expect(pool.query("UPDATE ml_deployments SET status='VALIDATING' WHERE id=$1", [d.id])).resolves.toBeDefined();
    await expect(pool.query("UPDATE ml_deployments SET status='ACTIVE' WHERE id=$1", [d.id])).rejects.toThrow(/invalid deployment transition/);
    await expect(pool.query("UPDATE ml_deployment_revisions SET model_version_id=$1 WHERE deployment_id=$2 AND revision=2", [v2.id, d.id])).rejects.toThrow(/immutable/);
    await expect(pool.query("INSERT INTO ml_deployments(tenant_id,name,status,created_by) VALUES($1,'x','ACTIVE',$2)", [manager.tenantId, manager.userId])).rejects.toThrow(/start CREATED/);
    await expect(pool.query("UPDATE ml_deployment_events SET reason='x' WHERE deployment_id=$1", [d.id])).rejects.toThrow();
  }));

  it("refuses incompatible input schemas, missing runtimes and other tenants, and answers refusals with 409 over HTTP", () => withStore(async store => {
    const f = await trainedModel(store);
    const registry = await f.models.registry(manager, { name: "schema" });
    const v1 = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id }); await f.models.transition(manager, v1.id, "READY");
    // A model whose prediction rows carry an extra column has a different input contract.
    const wide = await f.training.ingest(manager, { name: "wide", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "z", type: "string" }, { name: "y", type: "number" }], target: "y" }, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, z: "a", y: 2 * i + 1 })) });
    const job = await f.training.queue(manager, { experimentId: f.experiment.id, spec: { ...f.spec, datasetVersionId: wide.id } }); await f.worker.tick();
    const v2 = await f.models.register(manager, registry.id, { trainingRunId: (await f.training.getJob(manager, job.id)).attempts[0].id }); await f.models.transition(manager, v2.id, "READY");
    const deployments = new MlDeploymentService(pool, store, f.backends);
    const d = await deployments.create(manager, { name: "contract" });
    await deployments.addRevision(manager, d.id, { modelVersionId: v1.id }, signal()); await deployments.activate(manager, d.id, {});
    expect(await codes(deployments.addRevision(manager, d.id, { modelVersionId: v2.id }, signal()))).toEqual(["input_schema_incompatible"]);
    expect((await deployments.addRevision(manager, d.id, { modelVersionId: v2.id, allowSchemaChange: true }, signal())).status).toBe("READY");
    expect(await codes(new MlDeploymentService(pool, store, new Map()).addRevision(manager, d.id, { modelVersionId: v1.id }, signal()))).toEqual(["backend_unavailable"]);
    expect(await codes(deployments.addRevision(manager, d.id, { modelVersionId: v1.id, runtime: "onnx" }, signal()))).toEqual(["onnx_export_required"]);
    await expect(deployments.get(outsider, d.id)).rejects.toThrow(/not found/);
    await expect(deployments.predict(outsider, d.id, [{ x: 1 }], signal())).rejects.toThrow(/not found/);
    await expect(deployments.create(viewer, { name: "nope" })).rejects.toThrow(/role/);
    const app = testApp(store, f.backends);
    const refused = await app.request(`/ml/deployments/${d.id}/revisions`, json({ modelVersionId: v2.id }));
    expect(refused.status).toBe(409); expect((await refused.json()).reasons[0].code).toBe("input_schema_incompatible");
    expect((await app.request(`/ml/deployments/${d.id}/predict`, json({ rows: [{ x: 3 }] }))).status).toBe(200);
    const traced = await new MlLineageService(pool, store).deployment(viewer, d.id);
    expect(traced.traced).toMatchObject({ revision: 1, serving: true });
    expect(traced.model!.upstream.experiment.id).toBe(f.experiment.id); expect(traced.model!.upstream.datasetVersion.id).toBe(f.dataset.id);
    expect(traced.model!.downstream.deployments.map(x => x.deployment_id)).toContain(d.id);
  }));
});

describe.runIf(Boolean(process.env.ML_TEST_PYTHON && process.env.ML_TEST_MLOPS_PYTHON))("ONNX runtime deployments", () => {
  it("serves a verified ONNX export through a deployment and refuses ONNX endpoints without one", () => withStore(async store => {
    const backend = new SklearnBackend(process.env.ML_TEST_PYTHON!), onnx = { python: process.env.ML_TEST_MLOPS_PYTHON!, timeoutMs: 120_000 };
    const f = await trainedModel(store, backend, { rows: 80, algorithm: "ridge" });
    const registry = await f.models.registry(manager, { name: "onnx-served" });
    const version = await f.models.register(manager, registry.id, { trainingRunId: f.attempt.id }); await f.models.transition(manager, version.id, "READY");
    await expect(pool.query("INSERT INTO ml_endpoints(tenant_id,name,model_version_id,runtime,onnx_export_id) VALUES($1,'x',$2,'onnx',$3)", [manager.tenantId, version.id, randomUUID()])).rejects.toThrow();
    const exported = await new MlOnnxService(pool, store, onnx).export(manager, version.id, signal());
    expect(exported.status).toBe("VERIFIED");
    const deployments = new MlDeploymentService(pool, store, f.backends, { onnx });
    const d = await deployments.create(manager, { name: "onnx" });
    const revision = await deployments.addRevision(manager, d.id, { modelVersionId: version.id, runtime: "onnx" }, signal());
    expect(revision).toMatchObject({ status: "READY", runtime: "onnx", onnx_export_id: exported.id });
    await deployments.activate(manager, d.id, {});
    const result = await deployments.predict(manager, d.id, [{ x: 10 }, { x: 33 }], signal());
    expect(result.status).toBe("COMPLETED"); expect(result.output[0]).toBeCloseTo(21, 1); expect(result.output[1]).toBeCloseTo(67, 1); // ridge shrinks slightly; parity with native is in the export
    expect((await pool.query("SELECT runtime,onnx_export_id FROM ml_endpoints WHERE id=$1", [result.endpoint_id])).rows[0]).toEqual({ runtime: "onnx", onnx_export_id: exported.id });
    // Without the ONNX runtime configured, the same deployment's endpoint refuses to serve rather than falling back.
    await expect(new MlDeploymentService(pool, store, f.backends).predict(manager, d.id, [{ x: 1 }], signal())).rejects.toThrow(/not configured/);
  }), 300_000);
});
