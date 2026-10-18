import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore } from "./backend.js";
import { audit, authorizeMl, transaction } from "./persistence.js";

const Name = z.string().trim().min(1).max(120);
/**
 * A registry entry's promotion policy. Thresholds follow the stored metric's own direction: a "higher" metric must be at
 * least the threshold, a "lower" metric at most. An empty policy keeps the Phase 1 behaviour (an explicit operator decision).
 */
export const RegistryPolicy = z.strictObject({
  minimumMetrics: z.array(z.strictObject({ name: Name, partition: z.enum(["train", "validation", "test", "cv"]), threshold: z.number().finite() })).max(20).default([]),
  requireApproval: z.boolean().default(false),
  // READY also needs a VERIFIED ONNX export of this version (converted, accepted by the runtime gate, within parity tolerance).
  requireVerifiedOnnx: z.boolean().default(false),
});
export type RegistryPolicyType = z.infer<typeof RegistryPolicy>;
export const ApprovalInput = z.strictObject({ decision: z.enum(["APPROVED", "REJECTED"]), note: z.string().max(1000).default("") });

/** The closed set of reason codes a refused lifecycle operation can carry. */
export const REASON_CODES = ["artifact_integrity_failed", "policy_metric_missing", "policy_metric_threshold", "approval_required", "approval_rejected", "model_not_ready", "backend_unavailable", "onnx_export_required", "inference_check_failed", "input_schema_incompatible"] as const;
export type ReasonCode = (typeof REASON_CODES)[number];
export interface Reason { code: ReasonCode; detail?: Record<string, unknown> }
/** A governed operation that was refused. Its evidence is already persisted when this is thrown; callers return 409. */
export class GovernanceRefusal extends ConflictError { constructor(readonly reasons: Reason[], message = "Lifecycle operation refused") { super(message); } }

/**
 * Sets the actor, reason and details that the lifecycle trigger records for the next status change in this transaction.
 * Values are passed as bound parameters; the trigger reads them back, so nothing here is interpolated into SQL.
 */
export async function lifecycleContext(c: pg.PoolClient, actor: string, reason: string | null, details: unknown = {}) {
  await c.query("SELECT set_config('app.ml_actor',$1,true),set_config('app.ml_reason',$2,true),set_config('app.ml_details',$3,true)", [actor, reason ?? "", JSON.stringify(details)]);
}

export class MlGovernanceService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore) {}

  async setPolicy(principal: Principal, registryId: string, raw: unknown) {
    authorizeMl(principal, true); const policy = RegistryPolicy.parse(raw);
    return transaction(this.pool, async c => {
      if (!(await c.query("SELECT 1 FROM ml_registry_entries WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, registryId])).rowCount) throw new NotFoundError("Registry entry not found");
      const r = await c.query("INSERT INTO ml_registry_policies(tenant_id,registry_entry_id,version,policy,created_by) SELECT $1,$2,COALESCE(max(version),0)+1,$3,$4 FROM ml_registry_policies WHERE tenant_id=$1 AND registry_entry_id=$2 RETURNING *", [principal.tenantId, registryId, JSON.stringify(policy), principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.registry_policy_set", { registryEntryId: registryId, version: r.rows[0].version }); return r.rows[0];
    });
  }

  async policies(principal: Principal, registryId: string) {
    authorizeMl(principal);
    const rows = (await this.pool.query("SELECT * FROM ml_registry_policies WHERE tenant_id=$1 AND registry_entry_id=$2 ORDER BY version DESC LIMIT 100", [principal.tenantId, registryId])).rows;
    return { current: rows[0] ?? null, history: rows };
  }

  /** Approval by a tenant admin other than the registrant (the database refuses self-approval as well). */
  async approve(principal: Principal, modelVersionId: string, raw: unknown) {
    authorizeMl(principal, true); const input = ApprovalInput.parse(raw);
    return transaction(this.pool, async c => {
      const m = (await c.query("SELECT id,status,registered_by FROM ml_model_versions WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, modelVersionId])).rows[0];
      if (!m) throw new NotFoundError("Model version not found");
      if (m.status !== "REGISTERED") throw new ConflictError("Only REGISTERED versions can be approved");
      if (m.registered_by === principal.userId) throw new ConflictError("The registrant cannot approve their own model version");
      // Public tenant admins can create service identities; only a person may approve (the database refuses it as well).
      if ((await c.query("SELECT 1 FROM tenant_memberships WHERE tenant_id=$1 AND identity_id=$2 AND identity_type<>'user'", [principal.tenantId, principal.userId])).rowCount) throw new ConflictError("Service identities cannot approve model versions");
      const a = await c.query("INSERT INTO ml_model_approvals(tenant_id,model_version_id,decision,approver,note) VALUES($1,$2,$3,$4,$5) RETURNING *", [principal.tenantId, modelVersionId, input.decision, principal.userId, input.note]);
      await c.query("INSERT INTO ml_model_events(tenant_id,model_version_id,event,actor,reason,details) VALUES($1,$2,$3,$4,$5,$6)", [principal.tenantId, modelVersionId, input.decision === "APPROVED" ? "approved" : "approval_rejected", principal.userId, input.note || null, JSON.stringify({ approvalId: a.rows[0].id })]);
      await audit(c, principal.tenantId, principal.userId, "ml.model_approval_recorded", { modelVersionId, decision: input.decision }); return a.rows[0];
    });
  }

  /**
   * Checks a version against its registry's current policy and its own artifact. Read-only, so it can back both the
   * readiness view and the READY transition (which runs it under the version's row lock).
   */
  async evaluate(db: pg.Pool | pg.PoolClient, tenantId: string, modelVersionId: string) {
    const m = (await db.query("SELECT m.*,a.content FROM ml_model_versions m JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id WHERE m.tenant_id=$1 AND m.id=$2", [tenantId, modelVersionId])).rows[0];
    if (!m) throw new NotFoundError("Model version not found");
    const p = (await db.query("SELECT version,policy FROM ml_registry_policies WHERE tenant_id=$1 AND registry_entry_id=$2 ORDER BY version DESC LIMIT 1", [tenantId, m.registry_entry_id])).rows[0];
    const policy = RegistryPolicy.parse(p?.policy ?? {}), failed: Reason[] = [], passed: string[] = [];
    try { await this.artifacts.get(tenantId, m.content); passed.push("artifact_integrity"); } catch { failed.push({ code: "artifact_integrity_failed" }); }
    const metrics = (await db.query("SELECT name,partition,value,direction FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=$2 AND fold IS NULL", [tenantId, m.training_run_id])).rows;
    for (const rule of policy.minimumMetrics) {
      const metric = metrics.find(x => x.name === rule.name && x.partition === rule.partition);
      if (!metric) { failed.push({ code: "policy_metric_missing", detail: { name: rule.name, partition: rule.partition } }); continue; }
      const ok = metric.direction === "higher" ? metric.value >= rule.threshold : metric.value <= rule.threshold;
      if (ok) passed.push(`metric:${rule.partition}/${rule.name}`); else failed.push({ code: "policy_metric_threshold", detail: { name: rule.name, partition: rule.partition, threshold: rule.threshold, value: metric.value, direction: metric.direction } });
    }
    if (policy.requireApproval) {
      // The newest decision counts: a later rejection withdraws an earlier approval.
      const latest = (await db.query("SELECT decision,approver FROM ml_model_approvals WHERE tenant_id=$1 AND model_version_id=$2 ORDER BY created_at DESC,id DESC LIMIT 1", [tenantId, modelVersionId])).rows[0];
      if (!latest) failed.push({ code: "approval_required" }); else if (latest.decision !== "APPROVED") failed.push({ code: "approval_rejected" }); else passed.push("approval");
    }
    if (policy.requireVerifiedOnnx) {
      if ((await db.query("SELECT 1 FROM ml_onnx_exports WHERE tenant_id=$1 AND model_version_id=$2 AND status='VERIFIED'", [tenantId, modelVersionId])).rowCount) passed.push("onnx_parity"); else failed.push({ code: "onnx_export_required" });
    }
    return { modelVersion: { id: m.id, status: m.status, registryEntryId: m.registry_entry_id }, policyVersion: p?.version ?? null, policy, passed, reasons: failed, eligibleForReady: m.status === "REGISTERED" && failed.length === 0 };
  }

  async readiness(principal: Principal, modelVersionId: string) { authorizeMl(principal); return this.evaluate(this.pool, principal.tenantId, modelVersionId); }

  async events(principal: Principal, modelVersionId: string) {
    authorizeMl(principal);
    if (!(await this.pool.query("SELECT 1 FROM ml_model_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, modelVersionId])).rowCount) throw new NotFoundError("Model version not found");
    return (await this.pool.query("SELECT * FROM ml_model_events WHERE tenant_id=$1 AND model_version_id=$2 ORDER BY created_at,id LIMIT 500", [principal.tenantId, modelVersionId])).rows;
  }
}
