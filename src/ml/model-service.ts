import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { failureCategory, MlBackendFailure } from "./contract.js";
import { TrainingSnapshot, assertModelTransition, cohortKey, compareMetrics, validateRows, type ModelState, type MetricValue } from "./domain.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { GovernanceRefusal, lifecycleContext, MlGovernanceService, type Reason } from "./governance.js";
import { MlOnnxService, type OnnxConfig } from "./onnx.js";
import { MlMonitoringService } from "./monitoring.js";
import { getModel } from "./model-catalog.js";
import { TASKS, embeddingDimension, maxPredictionRows, predictionCount, validatePredictions } from "./tasks.js";

const Name = z.string().trim().min(1).max(120);
export const RegistryInput = z.object({ name: Name });
export const ModelVersionInput = z.object({ trainingRunId: z.uuid() });
export const EndpointInput = z.object({ name: Name, modelVersionId: z.uuid() });
export const ModelTransitionInput = z.object({ status: z.enum(["READY", "RETIRED", "REJECTED"]), reason: z.string().trim().min(1).max(1000).optional() });

export class MlModelService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>, private readonly options: { onnx?: OnnxConfig | null | undefined } = {}) {}
  async registry(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = RegistryInput.parse(raw);
    return transaction(this.pool, async c => {
      const r = await c.query("INSERT INTO ml_registry_entries(tenant_id,name,created_by) VALUES($1,$2,$3) RETURNING *", [principal.tenantId, input.name, principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.registry_created", { id: r.rows[0].id }); return r.rows[0];
    });
  }
  async register(principal: Principal, registryId: string, raw: unknown) {
    authorizeMl(principal, true); const input = ModelVersionInput.parse(raw);
    return transaction(this.pool, async c => {
      const registry = await c.query("SELECT id FROM ml_registry_entries WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, registryId]);
      const attempt = await c.query("SELECT a.id FROM ml_training_runs t JOIN ml_artifacts a ON a.tenant_id=t.tenant_id AND a.training_run_id=t.id AND a.kind='model' WHERE t.tenant_id=$1 AND t.id=$2 AND t.status='COMPLETED'", [principal.tenantId, input.trainingRunId]);
      if (!registry.rowCount || !attempt.rowCount) throw new NotFoundError("Registry or completed model artifact not found");
      const existing = await c.query("SELECT * FROM ml_model_versions WHERE tenant_id=$1 AND registry_entry_id=$2 AND training_run_id=$3", [principal.tenantId, registryId, input.trainingRunId]);
      if (existing.rowCount) return existing.rows[0];
      const version = (await c.query("SELECT COALESCE(max(version),0)+1 AS version FROM ml_model_versions WHERE tenant_id=$1 AND registry_entry_id=$2", [principal.tenantId, registryId])).rows[0].version;
      const m = await c.query("INSERT INTO ml_model_versions(tenant_id,registry_entry_id,training_run_id,artifact_id,version,registered_by) VALUES($1,$2,$3,$4,$5,$6) RETURNING *", [principal.tenantId, registryId, input.trainingRunId, attempt.rows[0].id, version, principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.model_registered", { modelVersionId: m.rows[0].id }); return m.rows[0];
    });
  }
  /**
   * READY is governed: the registry's current policy (metric thresholds, approval) and the artifact's integrity must hold.
   * A refusal is recorded as a lifecycle event in its own committed step and then thrown as GovernanceRefusal (409).
   */
  async transition(principal: Principal, id: string, to: ModelState, reason?: string) {
    authorizeMl(principal, true);
    const outcome = await transaction(this.pool, async (c): Promise<{ row: Record<string, unknown> } | { refused: Reason[] }> => {
      const selected = await c.query("SELECT m.* FROM ml_model_versions m WHERE m.tenant_id=$1 AND m.id=$2 FOR UPDATE OF m", [principal.tenantId, id]);
      if (!selected.rowCount) throw new NotFoundError("Model version not found"); const m = selected.rows[0]; assertModelTransition(m.status, to);
      let details: unknown = {};
      if (to === "READY") {
        const check = await new MlGovernanceService(this.pool, this.artifacts).evaluate(c, principal.tenantId, id);
        if (check.reasons.length) {
          await c.query("INSERT INTO ml_model_events(tenant_id,model_version_id,event,from_status,to_status,actor,reason,details) VALUES($1,$2,'transition_refused',$3,$4,$5,$6,$7)", [principal.tenantId, id, m.status, to, principal.userId, reason ?? null, JSON.stringify({ reasons: check.reasons, policyVersion: check.policyVersion })]);
          await audit(c, principal.tenantId, principal.userId, "ml.model_transition_refused", { modelVersionId: id, to, reasons: check.reasons.map(r => r.code) });
          return { refused: check.reasons };
        }
        details = { policyVersion: check.policyVersion, passed: check.passed };
      }
      await lifecycleContext(c, principal.userId, reason ?? null, details);
      const updated = await c.query("UPDATE ml_model_versions SET status=$1 WHERE tenant_id=$2 AND id=$3 RETURNING *", [to, principal.tenantId, id]);
      if (to === "RETIRED") await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND model_version_id=$2 AND status='ACTIVE'", [principal.tenantId, id]);
      await audit(c, principal.tenantId, principal.userId, "ml.model_transitioned", { modelVersionId: id, from: m.status, to }); return { row: updated.rows[0] };
    });
    if ("refused" in outcome) throw new GovernanceRefusal(outcome.refused, "Model transition refused by governance");
    return outcome.row;
  }
  async endpoint(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = EndpointInput.parse(raw);
    return transaction(this.pool, async c => {
      const m = await c.query("SELECT id FROM ml_model_versions WHERE tenant_id=$1 AND id=$2 AND status='READY' FOR SHARE", [principal.tenantId, input.modelVersionId]);
      if (!m.rowCount) throw new ConflictError("Endpoint requires a READY model version");
      // The same eligibility as a deployment: the registry's current policy and the artifact's integrity still hold.
      const check = await new MlGovernanceService(this.pool, this.artifacts).evaluate(c, principal.tenantId, input.modelVersionId);
      if (check.reasons.length) throw new GovernanceRefusal(check.reasons, "Endpoint refused by governance");
      const e = await c.query("INSERT INTO ml_endpoints(tenant_id,name,model_version_id) VALUES($1,$2,$3) RETURNING *", [principal.tenantId, input.name, input.modelVersionId]);
      await audit(c, principal.tenantId, principal.userId, "ml.endpoint_created", { endpointId: e.rows[0].id, modelVersionId: input.modelVersionId }); return e.rows[0];
    });
  }
  async disableEndpoint(principal: Principal, id: string) {
    authorizeMl(principal, true);
    return transaction(this.pool, async c => {
      await lifecycleContext(c, principal.userId, "endpoint disabled");
      const e = await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND id=$2 RETURNING *", [principal.tenantId, id]);
      if (!e.rowCount) throw new NotFoundError("Endpoint not found");
      await audit(c, principal.tenantId, principal.userId, "ml.endpoint_disabled", { endpointId: id }); return e.rows[0];
    });
  }
  async monitoring(principal: Principal, endpointId: string) {
    authorizeMl(principal);
    const result = await this.pool.query(`SELECT e.id AS endpoint_id,e.model_version_id,e.status AS endpoint_status,
      now()-interval '24 hours' AS window_start,now() AS window_end,
      count(p.id)::int AS requests,coalesce(sum(p.count),0)::float8 AS rows,
      count(p.id) FILTER (WHERE p.status='COMPLETED')::int AS completed,
      count(p.id) FILTER (WHERE p.status='FAILED')::int AS failed,
      count(p.id) FILTER (WHERE p.status='PENDING')::int AS pending,
      (count(p.id) FILTER (WHERE p.status='FAILED'))::float8 /
        NULLIF(count(p.id) FILTER (WHERE p.status IN ('COMPLETED','FAILED')),0) AS error_rate,
      avg(p.latency_ms) FILTER (WHERE p.status IN ('COMPLETED','FAILED'))::float8 AS average_latency_ms,
      percentile_cont(0.5) WITHIN GROUP (ORDER BY p.latency_ms)
        FILTER (WHERE p.status IN ('COMPLETED','FAILED')) AS p50_latency_ms,
      percentile_cont(0.95) WITHIN GROUP (ORDER BY p.latency_ms)
        FILTER (WHERE p.status IN ('COMPLETED','FAILED')) AS p95_latency_ms,
      COALESCE((SELECT jsonb_object_agg(f.category,f.n) FROM (SELECT COALESCE(x.failure->>'category','unknown') AS category,count(*)::int AS n
        FROM ml_predictions x WHERE x.tenant_id=e.tenant_id AND x.endpoint_id=e.id AND x.status='FAILED'
          AND x.created_at BETWEEN now()-interval '24 hours' AND now() GROUP BY 1) f),'{}') AS failures_by_category
      FROM ml_endpoints e LEFT JOIN ml_predictions p ON p.tenant_id=e.tenant_id AND p.endpoint_id=e.id
        AND p.created_at BETWEEN now()-interval '24 hours' AND now()
      WHERE e.tenant_id=$1 AND e.id=$2 GROUP BY e.id,e.model_version_id,e.status`, [principal.tenantId, endpointId]);
    if (!result.rowCount) throw new NotFoundError("Endpoint not found");
    return result.rows[0];
  }
  async predict(principal: Principal, endpointId: string, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal);
    const pending = await transaction(this.pool, async c => {
      const selected = await c.query(`SELECT e.id,e.runtime,x.content AS onnx_content,m.id AS model_version_id,t.snapshot,t.environment,a.content,a.format FROM ml_endpoints e
        JOIN ml_model_versions m ON m.tenant_id=e.tenant_id AND m.id=e.model_version_id
        LEFT JOIN ml_onnx_exports x ON x.tenant_id=e.tenant_id AND x.id=e.onnx_export_id
        JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id
        JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id
        WHERE e.tenant_id=$1 AND e.id=$2 AND e.status='ACTIVE' AND m.status='READY' FOR SHARE OF e,m`, [principal.tenantId, endpointId]);
      if (!selected.rowCount) throw new NotFoundError("Active endpoint with READY model not found"); const model = selected.rows[0];
      // The task comes from the catalog only (B1). It decides whether rows carry the target (forecasting history) and the
      // row cap, which is checked before any cell is decoded.
      const snapshot = TrainingSnapshot.parse(model.snapshot), definition = getModel(snapshot.backend, snapshot.algorithm), task = TASKS[definition.taskType];
      const columns = snapshot.dataset.schema.columns.filter(c => task.predictionIncludesTarget || c.name !== snapshot.dataset.schema.target);
      const limit = maxPredictionRows(definition.taskType, columns.map(c => c.type));
      if (Array.isArray(raw) && raw.length > limit) throw new ConflictError(`Prediction batch exceeds ${limit} rows`);
      const rows = validateRows(snapshot.dataset.schema, raw, true, task.predictionIncludesTarget);
      // An ONNX endpoint needs the ONNX runtime, not the training backend; a native endpoint needs its backend.
      const backend = this.backends.get(snapshot.backend);
      if (model.runtime === "onnx" ? !this.options.onnx : !backend) throw new ConflictError("Inference backend is not configured");
      const p = await c.query("INSERT INTO ml_predictions(tenant_id,endpoint_id,model_version_id,input_hash,count) VALUES($1,$2,$3,$4,$5) RETURNING *", [principal.tenantId, endpointId, model.model_version_id, canonicalJsonHash(rows), rows.length]);
      // Monitoring compares bin counts with the version's training reference (built once, on first use); raw rows are not kept.
      const reference = await new MlMonitoringService(this.pool, this.artifacts).reference(c, principal.tenantId, model.model_version_id);
      return { record: p.rows[0], model, snapshot, rows, backend, definition, reference };
    });
    const started = Date.now(); let output: unknown[] | null = null; let failure: { code: string; category: string; message: string } | null = null;
    try {
      if (pending.model.runtime === "onnx") output = await new MlOnnxService(this.pool, this.artifacts, this.options.onnx!).predict(principal.tenantId, pending.model.onnx_content, pending.snapshot, pending.rows, signal);
      else {
        if (!pending.backend!.capabilities.artifactFormats.includes(pending.model.format)) throw new MlBackendFailure("incompatible_artifact", "Unsupported artifact format");
        const bytes = await this.artifacts.get(principal.tenantId, pending.model.content);
        output = await pending.backend!.predict(pending.snapshot, bytes, pending.rows, pending.model.environment, signal);
      }
      output = validatePredictions(pending.definition.taskType, output, predictionCount(pending.definition, pending.snapshot, pending.rows.length), pending.model.environment);
    } catch (error) { output = null; failure = { code: "ml_prediction_failed", category: error instanceof z.ZodError ? "invalid_response" : failureCategory(error, "prediction_failed"), message: "Inference failed; inspect worker environment and artifact integrity" }; }
    return transaction(this.pool, async c => {
      // Embedding vectors are returned once and never stored (E2): the record keeps their digest and shape.
      const embedding = pending.definition.taskType === "embedding" && output !== null;
      const stored = embedding ? { digest: canonicalJsonHash(output), shape: [pending.rows.length, embeddingDimension(pending.model.environment)] } : output;
      const p = await c.query("UPDATE ml_predictions SET status=$1,latency_ms=$2,output=$3,failure=$4 WHERE tenant_id=$5 AND id=$6 AND status='PENDING' RETURNING *", [failure ? "FAILED" : "COMPLETED", Date.now() - started, stored ? JSON.stringify(stored) : null, failure ? JSON.stringify(failure) : null, principal.tenantId, pending.record.id]);
      if (output !== null && pending.reference) await new MlMonitoringService(this.pool, this.artifacts).record(c, principal.tenantId, p.rows[0], pending.reference, pending.rows, output);
      await audit(c, principal.tenantId, principal.userId, "ml.prediction_recorded", { predictionId: pending.record.id, status: p.rows[0].status, endpointId });
      return embedding ? { ...p.rows[0], vectors: output } : p.rows[0];
    });
  }
  async compare(principal: Principal, baselineId: string, candidateId: string) {
    authorizeMl(principal);
    const results = await this.pool.query("SELECT id,snapshot,split_indices FROM ml_training_runs WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='COMPLETED'", [principal.tenantId, [baselineId, candidateId]]);
    const baseline = results.rows.find(r => r.id === baselineId), candidate = results.rows.find(r => r.id === candidateId);
    if (!baseline || !candidate) throw new NotFoundError("Completed comparison attempts not found");
    if (cohortKey(baseline.snapshot, baseline.split_indices) !== cohortKey(candidate.snapshot, candidate.split_indices)) throw new ConflictError("Regression comparison requires matching data, schema, features, actual split and seed");
    const load = async (id: string): Promise<MetricValue[]> => (await this.pool.query("SELECT name,partition,value,direction,fold,std FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY partition,name,fold", [principal.tenantId, id])).rows as MetricValue[];
    const [b, c] = await Promise.all([load(baselineId), load(candidateId)]); return compareMetrics(b, c);
  }
}
