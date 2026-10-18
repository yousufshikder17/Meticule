import type { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import type pg from "pg";
import { z } from "zod";
import type { Principal } from "../db/types.js";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { RunRepository } from "../db/repositories.js";
import { PostgresArtifactStore } from "./artifact-store.js";
import { mlConfiguration } from "./configuration.js";
import { MlBackendFailure } from "./contract.js";
import { MlTrainingService, QueueTraining, authorizeMl } from "./persistence.js";
import { MlModelService, ModelTransitionInput } from "./model-service.js";
import { registerEmbeddingRoutes } from "./embedding-routes.js";
import { registerExperimentRoutes } from "./experiment-routes.js";
import { MlTrackingService } from "./tracking.js";
import { MlLineageService } from "./lineage.js";
import { MlMlflowService } from "./mlflow.js";

type Variables = { principal: Principal; database: pg.Pool; correlationId: string };
/** An optional Phase 5 feature that this deployment has not enabled. Mapped to 503 like ML itself being disabled. */
export class FeatureDisabled extends Error { readonly status = 503; constructor(feature: string) { super(`ML feature not configured: ${feature}`); } }
type Configuration = ReturnType<typeof mlConfiguration>;
// Phase 5 features are optional: a caller that omits one gets 503 for its routes, exactly as when it is disabled.
export type MlRouteConfiguration = Pick<Configuration, "enabled" | "backends"> & Partial<Omit<Configuration, "enabled" | "backends">>;
export function registerMlRoutes(app: Hono<{ Variables: Variables }>, ml: MlRouteConfiguration = mlConfiguration()): void {
  const id = (value: string) => z.uuid().parse(value);
  app.use("/ml/*", bodyLimit({ maxSize: 8_000_000, onError: c => c.json({ error: "payload_too_large" }, 413) }));
  app.use("/ml/*", async (c, next) => { if (!ml.enabled) return c.json({ error: "ml_not_configured" }, 503); return next(); });
  const tables = new Map([
    ["datasets", "ml_datasets"], ["pipelines", "ml_feature_pipelines"],
    ["registry", "ml_registry_entries"], ["model-versions", "ml_model_versions"], ["endpoints", "ml_endpoints"],
  ]);
  for (const [path, table] of tables) app.get(`/ml/${path}`, async c => {
    const p = c.get("principal"); authorizeMl(p);
    return c.json((await c.get("database").query(`SELECT * FROM ${table} WHERE tenant_id=$1 ORDER BY created_at DESC,id LIMIT 100`, [p.tenantId])).rows);
  });
  app.post("/ml/datasets", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).ingest(c.get("principal"), await c.req.json()), 201));
  app.post("/ml/datasets/:id/versions", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).ingest(c.get("principal"), await c.req.json(), id(c.req.param("id"))), 201));
  app.get("/ml/datasets/:id/versions", async c => { const p = c.get("principal"); authorizeMl(p); return c.json((await c.get("database").query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND dataset_id=$2 ORDER BY version DESC LIMIT 100", [p.tenantId, id(c.req.param("id"))])).rows.map(r => r.document)); });
  app.post("/ml/pipelines", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).pipeline(c.get("principal"), await c.req.json()), 201));
  app.post("/ml/pipelines/:id/versions", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).pipeline(c.get("principal"), await c.req.json(), id(c.req.param("id"))), 201));
  const tracking = (database: pg.Pool) => new MlTrackingService(database, new PostgresArtifactStore(database), ml.backends);
  const lineage = (database: pg.Pool) => new MlLineageService(database, new PostgresArtifactStore(database));
  app.get("/ml/experiments", async c => c.json(await tracking(c.get("database")).list(c.get("principal"), { ...c.req.query(), tag: c.req.queries("tag") ?? [] })));
  app.post("/ml/experiments", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).experiment(c.get("principal"), await c.req.json()), 201));
  app.get("/ml/experiments/:id", async c => c.json(await tracking(c.get("database")).get(c.get("principal"), id(c.req.param("id")))));
  app.patch("/ml/experiments/:id", async c => c.json(await tracking(c.get("database")).update(c.get("principal"), id(c.req.param("id")), await c.req.json())));
  app.get("/ml/experiments/:id/runs", async c => c.json(await tracking(c.get("database")).runs(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/training-runs/compare", async c => c.json(await tracking(c.get("database")).compare(c.get("principal"), await c.req.json())));
  app.post("/ml/training-jobs/:id/reproduce", async c => c.json(await tracking(c.get("database")).reproduce(c.get("principal"), id(c.req.param("id"))), 202));
  app.get("/ml/training-runs/:id/reproducibility/:candidate", async c => c.json(await tracking(c.get("database")).reproducibility(c.get("principal"), id(c.req.param("id")), id(c.req.param("candidate")))));
  app.get("/ml/lineage/model-versions/:id", async c => c.json(await lineage(c.get("database")).modelVersion(c.get("principal"), id(c.req.param("id")), c.req.query("verify") === "true")));
  app.get("/ml/lineage/training-runs/:id", async c => c.json(await lineage(c.get("database")).trainingRun(c.get("principal"), id(c.req.param("id")), c.req.query("verify") === "true")));
  const mlflow = (database: pg.Pool) => { if (!ml.mlflow) throw new FeatureDisabled("mlflow"); return new MlMlflowService(database, new PostgresArtifactStore(database), ml.mlflow); };
  app.post("/ml/training-runs/:id/mlflow-export", async c => { const r = await mlflow(c.get("database")).export(c.get("principal"), id(c.req.param("id")), c.req.raw.signal); return c.json(r, r.status === "FAILED" ? 502 : 200); });
  app.get("/ml/training-runs/:id/mlflow-exports", async c => c.json(await mlflow(c.get("database")).exports(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/mlflow/imports", async c => {
    try { return c.json(await mlflow(c.get("database")).import(c.get("principal"), await c.req.json(), c.req.raw.signal), 201); }
    catch (error) { if (error instanceof MlBackendFailure) return c.json({ error: "mlflow_import_failed", category: error.category }, 502); throw error; }
  });
  app.get("/ml/experiments/:id/external-runs", async c => c.json(await mlflow(c.get("database")).externalRuns(c.get("principal"), id(c.req.param("id")))));
  app.get("/ml/lineage/dataset-versions/:id", async c => c.json(await lineage(c.get("database")).datasetVersion(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/training-jobs", async c => {
    const p = c.get("principal"); authorizeMl(p, true); const input = QueueTraining.parse(await c.req.json());
    if (!ml.backends.has(input.spec.backend)) throw new ConflictError("Requested ML backend is not configured");
    return c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).queue(p, input), 202);
  });
  app.get("/ml/training-jobs/:id", async c => c.json(await new MlTrainingService(c.get("database"), new PostgresArtifactStore(c.get("database"))).getJob(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/training-jobs/:id/cancel", async c => {
    const p = c.get("principal"), database = c.get("database"); authorizeMl(p, true);
    const { job } = await new MlTrainingService(database, new PostgresArtifactStore(c.get("database"))).getJob(p, id(c.req.param("id")));
    return c.json(await new RunRepository(database).requestCancellation(p, job.run_id));
  });
  app.get("/ml/training-runs/:id/compare/:candidate", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).compare(c.get("principal"), id(c.req.param("id")), id(c.req.param("candidate")))));
  app.post("/ml/registry", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).registry(c.get("principal"), await c.req.json()), 201));
  app.post("/ml/registry/:id/versions", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).register(c.get("principal"), id(c.req.param("id")), await c.req.json()), 201));
  app.get("/ml/registry/:id/versions", async c => {
    const p = c.get("principal"); authorizeMl(p);
    return c.json((await c.get("database").query("SELECT * FROM ml_model_versions WHERE tenant_id=$1 AND registry_entry_id=$2 ORDER BY version DESC LIMIT 100", [p.tenantId, id(c.req.param("id"))])).rows);
  });
  app.get("/ml/model-versions/:id", async c => {
    const p = c.get("principal"); authorizeMl(p);
    const result = await c.get("database").query("SELECT m.*,t.snapshot,t.environment,t.resolved_hyperparameters,a.content,a.format FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id WHERE m.tenant_id=$1 AND m.id=$2", [p.tenantId, id(c.req.param("id"))]);
    if (!result.rowCount) throw new NotFoundError("Model version not found"); return c.json(result.rows[0]);
  });
  app.post("/ml/model-versions/:id/transition", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).transition(c.get("principal"), id(c.req.param("id")), ModelTransitionInput.parse(await c.req.json()).status)));
  app.post("/ml/endpoints", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).endpoint(c.get("principal"), await c.req.json()), 201));
  app.post("/ml/endpoints/:id/disable", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).disableEndpoint(c.get("principal"), id(c.req.param("id")))));
  app.get("/ml/endpoints/:id/monitoring", async c => c.json(await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).monitoring(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/endpoints/:id/predict", async c => {
    const record = await new MlModelService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends).predict(c.get("principal"), id(c.req.param("id")), z.object({ rows: z.unknown() }).parse(await c.req.json()).rows, c.req.raw.signal);
    return c.json(record, record.status === "FAILED" ? 422 : 200);
  });
  registerExperimentRoutes(app, ml);
  registerEmbeddingRoutes(app, ml);
  app.get("/ml/endpoints/:id/predictions", async c => {
    const p = c.get("principal"); authorizeMl(p);
    return c.json((await c.get("database").query("SELECT * FROM ml_predictions WHERE tenant_id=$1 AND endpoint_id=$2 ORDER BY created_at DESC,id LIMIT 100", [p.tenantId, id(c.req.param("id"))])).rows);
  });
}
