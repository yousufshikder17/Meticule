import type pg from "pg";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { TrainingSnapshot, assertModelTransition, compareMetrics, validateRows, type ModelState, type MetricValue } from "./domain.js";
import { audit, authorizeMl, transaction } from "./persistence.js";

const Name = z.string().trim().min(1).max(120);
export const RegistryInput = z.object({ name: Name });
export const ModelVersionInput = z.object({ trainingRunId: z.uuid() });
export const EndpointInput = z.object({ name: Name, modelVersionId: z.uuid() });
export const ModelTransitionInput = z.object({ status: z.enum(["READY", "RETIRED", "REJECTED"]) });

export class MlModelService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>) {}
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
      const m = await c.query("INSERT INTO ml_model_versions(tenant_id,registry_entry_id,training_run_id,artifact_id,version) VALUES($1,$2,$3,$4,$5) RETURNING *", [principal.tenantId, registryId, input.trainingRunId, attempt.rows[0].id, version]);
      await audit(c, principal.tenantId, principal.userId, "ml.model_registered", { modelVersionId: m.rows[0].id }); return m.rows[0];
    });
  }
  async transition(principal: Principal, id: string, to: ModelState) {
    authorizeMl(principal, true);
    return transaction(this.pool, async c => {
      const selected = await c.query("SELECT m.*,a.content FROM ml_model_versions m JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id WHERE m.tenant_id=$1 AND m.id=$2 FOR UPDATE OF m", [principal.tenantId, id]);
      if (!selected.rowCount) throw new NotFoundError("Model version not found"); const m = selected.rows[0]; assertModelTransition(m.status, to);
      if (to === "READY") await this.artifacts.get(principal.tenantId, m.content);
      const updated = await c.query("UPDATE ml_model_versions SET status=$1 WHERE tenant_id=$2 AND id=$3 RETURNING *", [to, principal.tenantId, id]);
      if (to === "RETIRED") await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND model_version_id=$2 AND status='ACTIVE'", [principal.tenantId, id]);
      await audit(c, principal.tenantId, principal.userId, "ml.model_transitioned", { modelVersionId: id, from: m.status, to }); return updated.rows[0];
    });
  }
  async endpoint(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = EndpointInput.parse(raw);
    return transaction(this.pool, async c => {
      const m = await c.query("SELECT id FROM ml_model_versions WHERE tenant_id=$1 AND id=$2 AND status='READY' FOR SHARE", [principal.tenantId, input.modelVersionId]);
      if (!m.rowCount) throw new ConflictError("Endpoint requires a READY model version");
      const e = await c.query("INSERT INTO ml_endpoints(tenant_id,name,model_version_id) VALUES($1,$2,$3) RETURNING *", [principal.tenantId, input.name, input.modelVersionId]);
      await audit(c, principal.tenantId, principal.userId, "ml.endpoint_created", { endpointId: e.rows[0].id, modelVersionId: input.modelVersionId }); return e.rows[0];
    });
  }
  async disableEndpoint(principal: Principal, id: string) {
    authorizeMl(principal, true);
    return transaction(this.pool, async c => {
      const e = await c.query("UPDATE ml_endpoints SET status='DISABLED' WHERE tenant_id=$1 AND id=$2 RETURNING *", [principal.tenantId, id]);
      if (!e.rowCount) throw new NotFoundError("Endpoint not found");
      await audit(c, principal.tenantId, principal.userId, "ml.endpoint_disabled", { endpointId: id }); return e.rows[0];
    });
  }
  async predict(principal: Principal, endpointId: string, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal);
    const pending = await transaction(this.pool, async c => {
      const selected = await c.query(`SELECT e.id,m.id AS model_version_id,t.snapshot,t.environment,a.content,a.format FROM ml_endpoints e
        JOIN ml_model_versions m ON m.tenant_id=e.tenant_id AND m.id=e.model_version_id
        JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id
        JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id
        WHERE e.tenant_id=$1 AND e.id=$2 AND e.status='ACTIVE' AND m.status='READY' FOR SHARE OF e,m`, [principal.tenantId, endpointId]);
      if (!selected.rowCount) throw new NotFoundError("Active endpoint with READY model not found"); const model = selected.rows[0];
      const snapshot = TrainingSnapshot.parse(model.snapshot); const rows = validateRows(snapshot.dataset.schema, raw, true);
      if (rows.length > 1000) throw new ConflictError("Prediction batch exceeds 1000 rows");
      const backend = this.backends.get(snapshot.backend); if (!backend) throw new ConflictError("Inference backend is not configured");
      const p = await c.query("INSERT INTO ml_predictions(tenant_id,endpoint_id,model_version_id,input_hash,count) VALUES($1,$2,$3,$4,$5) RETURNING *", [principal.tenantId, endpointId, model.model_version_id, canonicalJsonHash(rows), rows.length]);
      return { record: p.rows[0], model, snapshot, rows, backend };
    });
    const started = Date.now(); let output: unknown[] | null = null; let failure: { code: string; message: string } | null = null;
    try {
      const bytes = await this.artifacts.get(principal.tenantId, pending.model.content);
      output = await pending.backend.predict(pending.snapshot, bytes, pending.rows, pending.model.environment, signal);
      if (output.length !== pending.rows.length) throw new Error("Prediction count mismatch");
      z.array(z.json()).parse(output);
    } catch { output = null; failure = { code: "ml_prediction_failed", message: "Inference failed; inspect worker environment and artifact integrity" }; }
    return transaction(this.pool, async c => {
      const p = await c.query("UPDATE ml_predictions SET status=$1,latency_ms=$2,output=$3,failure=$4 WHERE tenant_id=$5 AND id=$6 AND status='PENDING' RETURNING *", [failure ? "FAILED" : "COMPLETED", Date.now() - started, output ? JSON.stringify(output) : null, failure ? JSON.stringify(failure) : null, principal.tenantId, pending.record.id]);
      await audit(c, principal.tenantId, principal.userId, "ml.prediction_recorded", { predictionId: pending.record.id, status: p.rows[0].status, endpointId }); return p.rows[0];
    });
  }
  async compare(principal: Principal, baselineId: string, candidateId: string) {
    authorizeMl(principal);
    const results = await this.pool.query("SELECT id,snapshot,split_indices FROM ml_training_runs WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='COMPLETED'", [principal.tenantId, [baselineId, candidateId]]);
    const baseline = results.rows.find(r => r.id === baselineId), candidate = results.rows.find(r => r.id === candidateId);
    if (!baseline || !candidate) throw new NotFoundError("Completed comparison attempts not found");
    const comparisonKey = (raw: unknown, indices: unknown) => { const s = TrainingSnapshot.parse(raw); return canonicalJsonHash({ dataset: s.dataset.content.sha256, schema: s.dataset.schema, pipeline: s.pipeline.definition, split: { ...s.split, id: null }, seed: s.seed, indices }); };
    if (comparisonKey(baseline.snapshot, baseline.split_indices) !== comparisonKey(candidate.snapshot, candidate.split_indices)) throw new ConflictError("Regression comparison requires matching data, schema, features, actual split and seed");
    const load = async (id: string): Promise<MetricValue[]> => (await this.pool.query("SELECT name,partition,value,direction FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY partition,name", [principal.tenantId, id])).rows as MetricValue[];
    const [b, c] = await Promise.all([load(baselineId), load(candidateId)]); return compareMetrics(b, c);
  }
}
