import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJson } from "../domain/canonical-json.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { failureCategory } from "./contract.js";
import { TrainingSnapshot, type Row, type Snapshot } from "./domain.js";
import { GovernanceRefusal, lifecycleContext, MlGovernanceService, type Reason } from "./governance.js";
import { getModel } from "./model-catalog.js";
import { MlModelService } from "./model-service.js";
import { MlOnnxService, type OnnxConfig } from "./onnx.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { predictionCount, TASKS, validatePredictions } from "./tasks.js";

const Name = z.string().trim().min(1).max(120);
const Reason = z.string().trim().min(1).max(1000).optional();
export const DeploymentInput = z.strictObject({ name: Name });
export const RevisionInput = z.strictObject({ modelVersionId: z.uuid(), runtime: z.enum(["native", "onnx"]).default("native"), onnxExportId: z.uuid().optional(), allowSchemaChange: z.boolean().default(false), reason: Reason });
export const ActivationInput = z.strictObject({ revision: z.int().positive().optional(), allowSchemaChange: z.boolean().default(false), reason: Reason });
/** Rows of the training run's test partition used for the readiness inference check. Never client data. */
const SMOKE_ROWS = 8;
export interface DeploymentOptions { onnx?: OnnxConfig | null | undefined }
interface Revision { revision: number; model_version_id: string; runtime: "native" | "onnx"; onnx_export_id: string | null; status: string }

/**
 * Durable deployment lifecycle over the existing registry and endpoints. A deployment record says what should serve; the
 * serving unit is an ordinary pinned endpoint created on activation, so prediction, history and monitoring are unchanged.
 * Serving is synchronous HTTP inference inside the API process: there is no separate serving process, autoscaling or
 * remote host behind a deployment.
 */
export class MlDeploymentService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>, private readonly options: DeploymentOptions = {}) {}

  async create(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = DeploymentInput.parse(raw);
    return transaction(this.pool, async c => {
      if ((await c.query("SELECT 1 FROM ml_deployments WHERE tenant_id=$1 AND name=$2", [principal.tenantId, input.name])).rowCount) throw new ConflictError("Deployment name is already used");
      const d = (await c.query("INSERT INTO ml_deployments(tenant_id,name,created_by) VALUES($1,$2,$3) RETURNING *", [principal.tenantId, input.name, principal.userId])).rows[0];
      await this.event(c, principal, d.id, "created", null, "CREATED"); await audit(c, principal.tenantId, principal.userId, "ml.deployment_created", { deploymentId: d.id });
      return d;
    });
  }

  async list(principal: Principal) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT * FROM ml_deployments WHERE tenant_id=$1 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId])).rows;
  }

  async get(principal: Principal, id: string) {
    authorizeMl(principal);
    const d = (await this.pool.query("SELECT * FROM ml_deployments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, id])).rows[0];
    if (!d) throw new NotFoundError("Deployment not found");
    const revisions = (await this.pool.query("SELECT * FROM ml_deployment_revisions WHERE tenant_id=$1 AND deployment_id=$2 ORDER BY revision", [principal.tenantId, id])).rows;
    const events = (await this.pool.query("SELECT * FROM ml_deployment_events WHERE tenant_id=$1 AND deployment_id=$2 ORDER BY created_at,id LIMIT 200", [principal.tenantId, id])).rows;
    return { deployment: d, revisions, events, serving: d.active_endpoint_id ? { endpointId: d.active_endpoint_id, revision: d.active_revision, mode: "synchronous in-process inference through the API" } : null };
  }

  /**
   * Adds a revision pinned to one model version and runtime, then validates it: eligibility (READY, current registry policy,
   * artifact integrity), runtime availability, input-schema compatibility with the active revision, and a smoke inference
   * on the training run's own test rows. A failed validation is recorded, then thrown as GovernanceRefusal.
   */
  async addRevision(principal: Principal, id: string, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal, true); const input = RevisionInput.parse(raw);
    const added = await transaction(this.pool, async c => {
      const d = await this.lock(c, principal.tenantId, id);
      const m = (await c.query("SELECT id FROM ml_model_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.modelVersionId])).rows[0];
      if (!m) throw new NotFoundError("Model version not found");
      let exportId: string | null = null;
      if (input.runtime === "onnx") {
        exportId = (await c.query("SELECT id FROM ml_onnx_exports WHERE tenant_id=$1 AND model_version_id=$2 AND status='VERIFIED' AND ($3::uuid IS NULL OR id=$3) ORDER BY created_at DESC LIMIT 1", [principal.tenantId, input.modelVersionId, input.onnxExportId ?? null])).rows[0]?.id ?? null;
        if (!exportId) {
          await this.event(c, principal, id, "refused", d.status, d.status, null, input.reason, { reasons: [{ code: "onnx_export_required" }], modelVersionId: input.modelVersionId });
          return { refused: [{ code: "onnx_export_required" } as Reason] };
        }
      }
      const revision = (await c.query("SELECT COALESCE(max(revision),0)+1 AS n FROM ml_deployment_revisions WHERE tenant_id=$1 AND deployment_id=$2", [principal.tenantId, id])).rows[0].n as number;
      await c.query("INSERT INTO ml_deployment_revisions(tenant_id,deployment_id,revision,model_version_id,runtime,onnx_export_id,created_by) VALUES($1,$2,$3,$4,$5,$6,$7)", [principal.tenantId, id, revision, input.modelVersionId, input.runtime, exportId, principal.userId]);
      const next = d.status === "ACTIVE" ? "ACTIVE" : "VALIDATING";
      if (next !== d.status) await c.query("UPDATE ml_deployments SET status=$1,updated_at=now() WHERE id=$2", [next, id]);
      await this.event(c, principal, id, "revision_added", d.status, next, revision, input.reason, { modelVersionId: input.modelVersionId, runtime: input.runtime, onnxExportId: exportId });
      return { revision, exportId, activeRevision: d.active_revision as number | null };
    });
    if ("refused" in added) throw new GovernanceRefusal(added.refused, "Deployment revision refused");
    // Any unexpected error is a failed validation, so the revision always leaves VALIDATING.
    const { reasons, evidence } = await this.validate(principal.tenantId, id, input, added.exportId, added.activeRevision, signal)
      .catch((error: unknown) => ({ reasons: [{ code: "inference_check_failed", detail: { category: failureCategory(error, "prediction_failed") } }] as Reason[], evidence: { error: "validation_error" } }));
    const result = await transaction(this.pool, async c => {
      const d = await this.lock(c, principal.tenantId, id);
      const status = reasons.length ? "FAILED" : "READY";
      const r = (await c.query("UPDATE ml_deployment_revisions SET status=$1,validation=$2,validated_at=now() WHERE tenant_id=$3 AND deployment_id=$4 AND revision=$5 RETURNING *", [status, JSON.stringify({ reasons, ...evidence }), principal.tenantId, id, added.revision])).rows[0];
      // FAILED means nothing deployable exists; an earlier READY revision keeps the deployment READY.
      let next = d.status;
      if (d.status === "VALIDATING") next = status === "READY" || (await c.query("SELECT 1 FROM ml_deployment_revisions WHERE tenant_id=$1 AND deployment_id=$2 AND status='READY'", [principal.tenantId, id])).rowCount ? "READY" : "FAILED";
      if (next !== d.status) await c.query("UPDATE ml_deployments SET status=$1,updated_at=now() WHERE id=$2", [next, id]);
      await this.event(c, principal, id, reasons.length ? "validation_failed" : "validated", d.status, next, added.revision, null, { reasons });
      await audit(c, principal.tenantId, principal.userId, "ml.deployment_revision_validated", { deploymentId: id, revision: added.revision, status });
      return r;
    });
    if (reasons.length) throw new GovernanceRefusal(reasons, "Deployment revision failed validation");
    return result;
  }

  /** Serves a READY revision (default: the newest) through a new pinned endpoint; the previous endpoint is disabled. */
  async activate(principal: Principal, id: string, raw: unknown) { return this.switch(principal, id, ActivationInput.parse(raw), "activated"); }

  /** Returns to an earlier verified revision: the given one, or the revision that served before the current one. */
  async rollback(principal: Principal, id: string, raw: unknown) { return this.switch(principal, id, ActivationInput.parse(raw), "rolled_back"); }

  async deactivate(principal: Principal, id: string, raw: unknown) {
    authorizeMl(principal, true); const input = ActivationInput.parse(raw);
    return transaction(this.pool, async c => {
      const d = await this.lock(c, principal.tenantId, id);
      if (d.status !== "ACTIVE") throw new ConflictError("Deployment is not active");
      // The endpoint trigger moves the deployment to INACTIVE and records the deployment and model events with this actor and reason.
      await lifecycleContext(c, principal.userId, input.reason ?? "deactivated");
      await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND id=$2", [principal.tenantId, d.active_endpoint_id]);
      await audit(c, principal.tenantId, principal.userId, "ml.deployment_deactivated", { deploymentId: id });
      return (await c.query("SELECT * FROM ml_deployments WHERE id=$1", [id])).rows[0];
    });
  }

  /** Inference through the active revision's endpoint, with the deployment and revision identity on the result. */
  async predict(principal: Principal, id: string, rows: unknown, signal: AbortSignal) {
    authorizeMl(principal);
    const d = (await this.pool.query("SELECT id,status,active_revision,active_endpoint_id FROM ml_deployments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, id])).rows[0];
    if (!d) throw new NotFoundError("Deployment not found");
    if (d.status !== "ACTIVE") throw new ConflictError("Deployment is not active");
    const record = await new MlModelService(this.pool, this.artifacts, this.backends, { onnx: this.options.onnx }).predict(principal, d.active_endpoint_id, rows, signal);
    return { ...record, deployment_id: id, revision: d.active_revision };
  }

  private async switch(principal: Principal, id: string, input: z.infer<typeof ActivationInput>, kind: "activated" | "rolled_back") {
    authorizeMl(principal, true);
    const outcome = await transaction(this.pool, async (c): Promise<{ row: Record<string, unknown> } | { refused: Reason[] }> => {
      const d = await this.lock(c, principal.tenantId, id);
      if (!["READY", "ACTIVE", "INACTIVE"].includes(d.status)) throw new ConflictError(`Deployment in status ${d.status} cannot be activated`);
      let target = input.revision;
      if (target === undefined && kind === "rolled_back") {
        // The revision that served before the current one; with nothing active, the last one that served.
        target = (await c.query("SELECT revision FROM ml_deployment_events WHERE tenant_id=$1 AND deployment_id=$2 AND event IN ('activated','rolled_back') AND revision IS DISTINCT FROM $3 ORDER BY created_at DESC,id DESC LIMIT 1", [principal.tenantId, id, d.active_revision])).rows[0]?.revision;
        if (target === undefined) throw new ConflictError("No earlier revision to roll back to");
      }
      const r: Revision | undefined = (await c.query("SELECT * FROM ml_deployment_revisions WHERE tenant_id=$1 AND deployment_id=$2 AND ($3::int IS NULL OR revision=$3) AND status='READY' ORDER BY revision DESC LIMIT 1", [principal.tenantId, id, target ?? null])).rows[0];
      if (!r) throw new ConflictError("No READY revision to activate");
      if (r.revision === d.active_revision) throw new ConflictError("Revision is already active");
      // Eligibility can change after validation (a retired model, a stricter policy, a removed runtime): check it again.
      const reasons = await this.eligibility(c, principal.tenantId, r.model_version_id, r.runtime);
      // The input contract is checked against the revision that served last, even when nothing serves right now.
      const last = (await c.query("SELECT revision FROM ml_deployment_events WHERE tenant_id=$1 AND deployment_id=$2 AND event IN ('activated','rolled_back') ORDER BY created_at DESC,id DESC LIMIT 1", [principal.tenantId, id])).rows[0]?.revision as number | undefined;
      if (last !== undefined && last !== r.revision && !input.allowSchemaChange && canonicalJson(await this.signatureOf(c, principal.tenantId, id, last)) !== canonicalJson(await this.signatureOf(c, principal.tenantId, id, r.revision))) reasons.push({ code: "input_schema_incompatible", detail: { lastServedRevision: last } });
      if (reasons.length) {
        await this.event(c, principal, id, "refused", d.status, d.status, r.revision, input.reason, { reasons, action: kind });
        return { refused: reasons };
      }
      const endpoint = (await c.query("INSERT INTO ml_endpoints(tenant_id,name,model_version_id,runtime,onnx_export_id) VALUES($1,$2,$3,$4,$5) RETURNING id", [principal.tenantId, `${d.name}#r${r.revision}`.slice(0, 120), r.model_version_id, r.runtime, r.onnx_export_id])).rows[0].id as string;
      await c.query("UPDATE ml_deployments SET status='ACTIVE',active_revision=$1,active_endpoint_id=$2,updated_at=now() WHERE id=$3", [r.revision, endpoint, id]);
      // The old endpoint is disabled only after the deployment points elsewhere, so the disable trigger leaves it ACTIVE.
      if (d.active_endpoint_id) { await lifecycleContext(c, principal.userId, `replaced by revision ${r.revision}`); await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND id=$2", [principal.tenantId, d.active_endpoint_id]); }
      await this.event(c, principal, id, kind, d.status, "ACTIVE", r.revision, input.reason, { previousRevision: d.active_revision, endpointId: endpoint, modelVersionId: r.model_version_id, runtime: r.runtime }, endpoint);
      await this.modelEvent(c, principal, { ...d, active_revision: r.revision, active_endpoint_id: endpoint, model_version_id: r.model_version_id }, kind);
      await audit(c, principal.tenantId, principal.userId, `ml.deployment_${kind}`, { deploymentId: id, revision: r.revision, endpointId: endpoint });
      return { row: (await c.query("SELECT * FROM ml_deployments WHERE id=$1", [id])).rows[0] };
    });
    if ("refused" in outcome) throw new GovernanceRefusal(outcome.refused, "Deployment activation refused");
    return outcome.row;
  }

  /** Whether a model version may serve on a runtime now. Shared by validation and every activation. */
  private async eligibility(db: pg.Pool | pg.PoolClient, tenantId: string, modelVersionId: string, runtime: "native" | "onnx"): Promise<Reason[]> {
    const check = await new MlGovernanceService(this.pool, this.artifacts).evaluate(db, tenantId, modelVersionId);
    const reasons = [...check.reasons];
    if (check.modelVersion.status !== "READY") reasons.unshift({ code: "model_not_ready", detail: { status: check.modelVersion.status } });
    const backend = (await db.query("SELECT t.snapshot->>'backend' AS backend FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE m.tenant_id=$1 AND m.id=$2", [tenantId, modelVersionId])).rows[0].backend as string;
    if (runtime === "native" ? !this.backends.has(backend) : !this.options.onnx) reasons.push({ code: "backend_unavailable", detail: { runtime } });
    return reasons;
  }

  private async validate(tenantId: string, id: string, input: z.infer<typeof RevisionInput>, exportId: string | null, activeRevision: number | null, signal: AbortSignal) {
    const reasons = await this.eligibility(this.pool, tenantId, input.modelVersionId, input.runtime);
    const model = (await this.pool.query("SELECT t.snapshot,t.environment,t.split_indices,a.content FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id WHERE m.tenant_id=$1 AND m.id=$2", [tenantId, input.modelVersionId])).rows[0];
    const snapshot = TrainingSnapshot.parse(model.snapshot), signature = inputSignature(snapshot);
    if (activeRevision !== null && !input.allowSchemaChange) {
      const active = (await this.pool.query("SELECT t.snapshot FROM ml_deployment_revisions r JOIN ml_model_versions m ON m.tenant_id=r.tenant_id AND m.id=r.model_version_id JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE r.tenant_id=$1 AND r.deployment_id=$2 AND r.revision=$3", [tenantId, id, activeRevision])).rows[0];
      if (canonicalJson(inputSignature(TrainingSnapshot.parse(active.snapshot))) !== canonicalJson(signature)) reasons.push({ code: "input_schema_incompatible", detail: { activeRevision } });
    }
    let smoke: Record<string, unknown> | null = null;
    if (!reasons.length) {
      const started = Date.now();
      try {
        const rows = await this.smokeRows(tenantId, snapshot, model.split_indices.test ?? []);
        const task = getModel(snapshot.backend, snapshot.algorithm).taskType;
        if (input.runtime === "onnx") {
          const e = (await this.pool.query("SELECT content FROM ml_onnx_exports WHERE tenant_id=$1 AND id=$2", [tenantId, exportId])).rows[0];
          await new MlOnnxService(this.pool, this.artifacts, this.options.onnx!).predict(tenantId, e.content, snapshot, rows, signal);
        } else {
          const output = await this.backends.get(snapshot.backend)!.predict(snapshot, await this.artifacts.get(tenantId, model.content), rows, model.environment, signal);
          validatePredictions(task, output, predictionCount(getModel(snapshot.backend, snapshot.algorithm), snapshot, rows.length), model.environment);
        }
        smoke = { rows: rows.length, latencyMs: Date.now() - started, runtime: input.runtime };
      } catch (error) { reasons.push({ code: "inference_check_failed", detail: { category: failureCategory(error, "prediction_failed") } }); }
    }
    return { reasons, evidence: { signature, smoke, checkedAt: new Date().toISOString() } };
  }

  /** Prediction-shaped rows from the training run's test partition (every column except the target, forecasting aside). */
  private async smokeRows(tenantId: string, snapshot: Snapshot, test: number[]): Promise<Row[]> {
    const data = JSON.parse((await this.artifacts.get(tenantId, snapshot.dataset.content)).toString("utf8")) as Row[];
    const target = snapshot.dataset.schema.target, task = getModel(snapshot.backend, snapshot.algorithm).taskType;
    // Forecasting predicts from history, so its check uses the newest stored rows (bounded like a prediction request).
    const pick = task === "forecasting" ? data.slice(-1000) : test.slice(0, SMOKE_ROWS).map(i => data[i]!);
    if (!pick.length) throw new ConflictError("No rows for the readiness check");
    return pick.map(row => Object.fromEntries(Object.entries(row).filter(([k]) => task === "forecasting" || k !== target)) as Row);
  }

  private async signatureOf(db: pg.Pool | pg.PoolClient, tenantId: string, id: string, revision: number) {
    const row = (await db.query("SELECT t.snapshot FROM ml_deployment_revisions r JOIN ml_model_versions m ON m.tenant_id=r.tenant_id AND m.id=r.model_version_id JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id WHERE r.tenant_id=$1 AND r.deployment_id=$2 AND r.revision=$3", [tenantId, id, revision])).rows[0];
    return inputSignature(TrainingSnapshot.parse(row.snapshot));
  }
  private async lock(c: pg.PoolClient, tenantId: string, id: string) {
    const d = (await c.query("SELECT * FROM ml_deployments WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [tenantId, id])).rows[0];
    if (!d) throw new NotFoundError("Deployment not found"); return d;
  }
  private async event(c: pg.PoolClient, principal: Principal, id: string, event: string, from: string | null, to: string | null, revision: number | null = null, reason: string | null | undefined = null, details: unknown = {}, endpoint: string | null = null) {
    await c.query("INSERT INTO ml_deployment_events(tenant_id,deployment_id,event,from_status,to_status,revision,endpoint_id,actor,reason,details) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)", [principal.tenantId, id, event, from, to, revision, endpoint, principal.userId, reason ?? null, JSON.stringify(details)]);
  }
  /** Mirrors deployment changes into the model version's own lifecycle history, so lineage shows where a version served. */
  private async modelEvent(c: pg.PoolClient, principal: Principal, d: Record<string, any>, action: string) {
    const version = d.model_version_id ?? (await c.query("SELECT model_version_id FROM ml_deployment_revisions WHERE tenant_id=$1 AND deployment_id=$2 AND revision=$3", [principal.tenantId, d.id, d.active_revision])).rows[0]?.model_version_id;
    if (version) await c.query("INSERT INTO ml_model_events(tenant_id,model_version_id,event,actor,details) VALUES($1,$2,'deployment_changed',$3,$4)", [principal.tenantId, version, principal.userId, JSON.stringify({ deploymentId: d.id, revision: d.active_revision, endpointId: d.active_endpoint_id, action })]);
  }
}

/** What a client sends and receives: the prediction row columns (the dataset schema without the target, except for forecasting history) and the task. */
export function inputSignature(snapshot: Snapshot) {
  const task = getModel(snapshot.backend, snapshot.algorithm).taskType, target = snapshot.dataset.schema.target;
  return { task, columns: snapshot.dataset.schema.columns.filter(c => TASKS[task].predictionIncludesTarget || c.name !== target).map(c => ({ name: c.name, type: c.type, nullable: c.nullable })) };
}
