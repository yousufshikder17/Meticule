import type pg from "pg";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { z } from "zod";
import { NotFoundError } from "../domain/errors.js";
import { canonicalJson } from "../domain/canonical-json.js";
import type { Principal } from "../db/types.js";
import { failureCategory, MlBackendFailure } from "./contract.js";
import { TrainingSnapshot } from "./domain.js";
import { MlLineageService, sequence } from "./lineage.js";
import type { ArtifactStore } from "./backend.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { callRunner } from "./runner-process.js";

export interface MlflowConfig { python: string; timeoutMs: number; trackingUri: string; token: string; prefix: string }
export const MlflowImportInput = z.strictObject({ experimentId: z.uuid(), mlflowRunId: z.string().regex(/^[0-9a-f]{32}$/) });
// MLflow's own limits: parameter keys are word characters, '.', '-', ' ' and '/'; values at most 6000 characters.
const PARAM_KEY = /^[\w.\- /]{1,250}$/;
const ExportResponse = z.object({ runId: z.string().regex(/^[0-9a-f]{32}$/), experimentId: z.string().min(1).max(64), reused: z.boolean() });
const ImportedRun = z.object({ run: z.object({
  runId: z.string().regex(/^[0-9a-f]{32}$/), experimentId: z.string().max(64), name: z.string().max(500), status: z.string().max(20),
  startTime: z.number().nullable(), endTime: z.number().nullable(),
  params: z.record(z.string().max(500), z.string().max(500)), metrics: z.record(z.string().max(500), z.number().finite()), tags: z.record(z.string().max(500), z.string().max(500)),
}) });

/**
 * MLflow is an optional mirror of Meticule's own records, never their source. Export copies one completed attempt's
 * parameters, metrics, epoch curves and lineage document; import copies one MLflow run's metadata as data. Neither
 * direction moves a model artifact, so the artifact trust boundary is unchanged.
 */
export class MlMlflowService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly config: MlflowConfig, private readonly script = resolve(process.cwd(), "ml/mlflow_runner.py")) {}
  private get uriHash() { return createHash("sha256").update(this.config.trackingUri).digest("hex"); }

  private call(command: string, request: object, signal: AbortSignal) {
    // The token reaches only this runner, and MLflow's own retry loop is cut short so a dead server fails within the timeout.
    const environment: Record<string, string> = { MLFLOW_HTTP_REQUEST_MAX_RETRIES: "1", MLFLOW_HTTP_REQUEST_TIMEOUT: "15", MLFLOW_DISABLE_TELEMETRY: "true", DO_NOT_TRACK: "true", ...(this.config.token ? { MLFLOW_TRACKING_TOKEN: this.config.token } : {}) };
    return callRunner({ label: "mlflow", python: this.config.python, script: this.script, timeoutMs: this.config.timeoutMs, signal, environment, request: { command, trackingUri: this.config.trackingUri, ...request } });
  }

  /**
   * Idempotent per attempt and server: a completed export is returned as is. A failure is recorded and returned (not thrown)
   * so the evidence survives the request transaction; the next call retries it.
   */
  async export(principal: Principal, trainingRunId: string, signal: AbortSignal) {
    authorizeMl(principal, true);
    return transaction(this.pool, async c => {
      const t = (await c.query("SELECT t.*,j.experiment_id,e.name AS experiment_name FROM ml_training_runs t JOIN ml_training_jobs j ON j.tenant_id=t.tenant_id AND j.id=t.job_id JOIN ml_experiments e ON e.tenant_id=j.tenant_id AND e.id=j.experiment_id WHERE t.tenant_id=$1 AND t.id=$2 AND t.status='COMPLETED'", [principal.tenantId, trainingRunId])).rows[0];
      if (!t) throw new NotFoundError("Completed training run not found");
      await c.query("INSERT INTO ml_mlflow_exports(tenant_id,training_run_id,tracking_uri_hash,status,created_by) VALUES($1,$2,$3,'PENDING',$4) ON CONFLICT (tenant_id,training_run_id,tracking_uri_hash) DO NOTHING", [principal.tenantId, trainingRunId, this.uriHash, principal.userId]);
      const record = (await c.query("SELECT * FROM ml_mlflow_exports WHERE tenant_id=$1 AND training_run_id=$2 AND tracking_uri_hash=$3 FOR UPDATE", [principal.tenantId, trainingRunId, this.uriHash])).rows[0];
      if (record.status === "COMPLETED") return { ...record, reused: true };
      const { run, unsupported } = await this.document(c, principal.tenantId, t);
      let result: z.infer<typeof ExportResponse> | null = null, failure: { category: string; message: string } | null = null;
      try {
        const parsed = ExportResponse.safeParse(await this.call("export", { experimentName: `${this.config.prefix}/${principal.tenantId}/${t.experiment_id}`, experimentTags: { "meticule.experiment_name": t.experiment_name }, run }, signal));
        if (!parsed.success) throw new MlBackendFailure("invalid_response", "MLflow runner response did not match the contract");
        result = parsed.data;
      } catch (error) { failure = { category: failureCategory(error, "backend_unavailable"), message: "MLflow export failed; check the tracking server and its configuration" }; }
      const updated = (await c.query("UPDATE ml_mlflow_exports SET status=$1,mlflow_experiment_id=$2,mlflow_run_id=$3,unsupported=$4,failure=$5,attempts=attempts+1,updated_at=now() WHERE id=$6 RETURNING *",
        [result ? "COMPLETED" : "FAILED", result?.experimentId ?? null, result?.runId ?? null, JSON.stringify(unsupported), failure ? JSON.stringify(failure) : null, record.id])).rows[0];
      await audit(c, principal.tenantId, principal.userId, "ml.mlflow_exported", { trainingRunId, exportId: record.id, status: updated.status });
      return { ...updated, reused: false };
    });
  }

  async exports(principal: Principal, trainingRunId: string) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT * FROM ml_mlflow_exports WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY created_at", [principal.tenantId, trainingRunId])).rows;
  }

  /** Reads one run from the configured server into an experiment as metadata. Repeating an import returns the stored copy. */
  async import(principal: Principal, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal, true); const input = MlflowImportInput.parse(raw);
    if (!(await this.pool.query("SELECT 1 FROM ml_experiments WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.experimentId])).rowCount) throw new NotFoundError("Experiment not found");
    const parsed = ImportedRun.safeParse(await this.call("read", { mlflowRunId: input.mlflowRunId }, signal));
    if (!parsed.success) throw new MlBackendFailure("invalid_response", "MLflow runner response did not match the contract");
    return transaction(this.pool, async c => {
      const inserted = await c.query("INSERT INTO ml_external_runs(tenant_id,experiment_id,source,tracking_uri_hash,external_run_id,document,imported_by) VALUES($1,$2,'mlflow',$3,$4,$5,$6) ON CONFLICT (tenant_id,experiment_id,tracking_uri_hash,external_run_id) DO NOTHING RETURNING *",
        [principal.tenantId, input.experimentId, this.uriHash, input.mlflowRunId, JSON.stringify(parsed.data.run), principal.userId]);
      if (!inserted.rowCount) return { ...(await c.query("SELECT * FROM ml_external_runs WHERE tenant_id=$1 AND experiment_id=$2 AND tracking_uri_hash=$3 AND external_run_id=$4", [principal.tenantId, input.experimentId, this.uriHash, input.mlflowRunId])).rows[0], reused: true };
      await audit(c, principal.tenantId, principal.userId, "ml.mlflow_imported", { experimentId: input.experimentId, externalRunId: inserted.rows[0].id });
      return { ...inserted.rows[0], reused: false };
    });
  }

  async externalRuns(principal: Principal, experimentId: string) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT * FROM ml_external_runs WHERE tenant_id=$1 AND experiment_id=$2 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, experimentId])).rows;
  }

  /** The MLflow view of one attempt. Anything MLflow cannot represent is listed in `unsupported`, never silently dropped. */
  private async document(c: pg.PoolClient, tenantId: string, t: Record<string, any>) {
    const s = TrainingSnapshot.parse(t.snapshot), unsupported: { kind: string; name?: string; reason: string }[] = [];
    const [metrics, epochs, models, artifact, checkpoints] = await sequence([
      () => c.query("SELECT name,partition,value,fold FROM ml_metrics WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY partition,name,fold", [tenantId, t.id]),
      // Epoch history spans the job's attempts (a resumed attempt continues an earlier one); the newest attempt wins per epoch.
      () => c.query("SELECT DISTINCT ON (e.epoch) e.epoch,e.train_loss,e.validation_loss FROM ml_training_epochs e JOIN ml_training_runs r ON r.tenant_id=e.tenant_id AND r.id=e.training_run_id WHERE r.tenant_id=$1 AND r.job_id=$2 AND r.attempt<=$3 ORDER BY e.epoch,r.attempt DESC", [tenantId, t.job_id, t.attempt]),
      () => c.query("SELECT id FROM ml_model_versions WHERE tenant_id=$1 AND training_run_id=$2 ORDER BY created_at", [tenantId, t.id]),
      () => c.query("SELECT id,format,content->>'sha256' AS sha256 FROM ml_artifacts WHERE tenant_id=$1 AND training_run_id=$2 AND kind='model'", [tenantId, t.id]),
      () => c.query("SELECT count(*)::int AS n FROM ml_training_checkpoints k JOIN ml_training_runs r ON r.tenant_id=k.tenant_id AND r.id=k.training_run_id WHERE r.tenant_id=$1 AND r.job_id=$2", [tenantId, t.job_id]),
    ]);
    const params: Record<string, string> = { backend: s.backend, algorithm: s.algorithm, seed: String(s.seed), "split.strategy": s.split.strategy, "split.train": String(s.split.train), "split.validation": String(s.split.validation), "split.test": String(s.split.test), dataset_version_id: s.datasetVersionId, feature_pipeline_id: s.featurePipelineId };
    const add = (prefix: string, values: Record<string, unknown>) => { for (const [k, v] of Object.entries(values)) { const key = `${prefix}.${k}`; if (PARAM_KEY.test(key)) params[key] = canonicalJson(v).slice(0, 6000); else unsupported.push({ kind: "parameter", name: key.slice(0, 120), reason: "Key is not a valid MLflow parameter name" }); } };
    add("hp", s.hyperparameters); add("resolved", t.resolved_hyperparameters ?? {});
    const values = [...metrics.rows.map(m => ({ key: `${m.partition}.${m.name}`, value: Number(m.value), step: m.fold ?? 0 })),
      ...epochs.rows.flatMap(e => [{ key: "epoch.train_loss", value: Number(e.train_loss), step: e.epoch }, ...(e.validation_loss === null ? [] : [{ key: "epoch.validation_loss", value: Number(e.validation_loss), step: e.epoch }])])];
    const a = artifact.rows[0];
    unsupported.push({ kind: "model_artifact", name: a.format, reason: "Native Meticule model artifacts are not exported to MLflow; the lineage document references them by checksum" });
    if (checkpoints.rows[0].n) unsupported.push({ kind: "checkpoint", reason: `${checkpoints.rows[0].n} training checkpoints stay in Meticule's artifact store` });
    const tags: Record<string, string> = { "meticule.tenant_id": tenantId, "meticule.experiment_id": t.experiment_id, "meticule.job_id": t.job_id, "meticule.training_run_id": t.id, "meticule.attempt": String(t.attempt),
      "meticule.dataset_version_id": s.datasetVersionId, "meticule.dataset_sha256": s.dataset.content.sha256, "meticule.feature_pipeline_id": s.featurePipelineId, "meticule.backend": s.backend,
      "meticule.model_artifact_sha256": a.sha256, "meticule.model_version_ids": models.rows.map(m => m.id).join(","), ...(s.source ? { "meticule.source_revision": s.source.revision } : {}) };
    const upstream = await new MlLineageService(this.pool, this.artifacts).upstream(c, tenantId, t.id, []);
    const { content: _content, ...datasetVersion } = upstream.datasetVersion;
    const lineage = { schema: "meticule-lineage-v1", ...upstream, datasetVersion, modelArtifact: { id: a.id, format: a.format, sha256: a.sha256 }, modelVersionIds: models.rows.map(m => m.id) };
    return { run: { name: `${s.algorithm}#${t.attempt}`, startTime: new Date(t.started_at).getTime(), endTime: new Date(t.ended_at).getTime(), params, metrics: values, tags, lineage: JSON.parse(JSON.stringify(lineage)) }, unsupported };
  }
}
