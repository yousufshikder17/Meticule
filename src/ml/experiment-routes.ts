import type { Hono } from "hono";
import type pg from "pg";
import { z } from "zod";
import type { Principal } from "../db/types.js";
import type { MlBackend } from "./backend.js";
import { PostgresArtifactStore } from "./artifact-store.js";
import { MlExperimentService, type OrchestrationKind } from "./experiments.js";
import { METRICS } from "./metrics.js";
import { describeModel, listModels } from "./model-catalog.js";
import { authorizeMl } from "./persistence.js";

type Variables = { principal: Principal; database: pg.Pool; correlationId: string };
// Benchmark, search and AutoML share one service and route shape; each path only fixes the kind.
const KINDS: [string, OrchestrationKind][] = [["benchmarks", "benchmark"], ["searches", "search"], ["automl", "automl"]];

export function registerExperimentRoutes(app: Hono<{ Variables: Variables }>, ml: { backends: ReadonlyMap<string, MlBackend> }): void {
  const id = (value: string) => z.uuid().parse(value);
  const service = (database: pg.Pool) => new MlExperimentService(database, new PostgresArtifactStore(database), ml.backends);
  app.get("/ml/models", c => { authorizeMl(c.get("principal")); return c.json({ models: listModels().map(describeModel), metrics: METRICS }); });
  for (const [path, kind] of KINDS) {
    app.get(`/ml/${path}`, async c => c.json(await service(c.get("database")).list(c.get("principal"), kind)));
    app.post(`/ml/${path}`, async c => {
      const s = service(c.get("database")), p = c.get("principal"), body = await c.req.json();
      return c.json(kind === "benchmark" ? await s.benchmark(p, body) : kind === "search" ? await s.search(p, body) : await s.automl(p, body), 202);
    });
    app.get(`/ml/${path}/:id`, async c => c.json(await service(c.get("database")).get(c.get("principal"), id(c.req.param("id")), kind)));
  }
  app.get("/ml/experiments/:id/ranking", async c => c.json(await service(c.get("database")).ranking(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/experiments/:id/cancel", async c => c.json(await service(c.get("database")).cancel(c.get("principal"), id(c.req.param("id")))));
}
