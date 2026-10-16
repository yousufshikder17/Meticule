import type { Hono } from "hono";
import type pg from "pg";
import { z } from "zod";
import type { Principal } from "../db/types.js";
import { PostgresArtifactStore } from "./artifact-store.js";
import type { mlConfiguration } from "./configuration.js";
import { MlEmbeddingService } from "./embedding-service.js";
import { authorizeMl } from "./persistence.js";

type Variables = { principal: Principal; database: pg.Pool; correlationId: string };

/** Embedding indexes (build, search) and pairwise similarity. Generation stays in the backends behind the endpoints. */
export function registerEmbeddingRoutes(app: Hono<{ Variables: Variables }>, ml: ReturnType<typeof mlConfiguration>): void {
  const id = (value: string) => z.uuid().parse(value);
  const service = (c: { get(key: "database"): pg.Pool }) => new MlEmbeddingService(c.get("database"), new PostgresArtifactStore(c.get("database")), ml.backends);
  app.post("/ml/embedding-indexes", async c => c.json(await service(c).create(c.get("principal"), await c.req.json()), 201));
  app.get("/ml/embedding-indexes", async c => {
    const p = c.get("principal"); authorizeMl(p);
    return c.json((await c.get("database").query("SELECT * FROM ml_embedding_indexes WHERE tenant_id=$1 ORDER BY created_at DESC,id LIMIT 100", [p.tenantId])).rows);
  });
  app.get("/ml/embedding-indexes/:id", async c => c.json(await service(c).get(c.get("principal"), id(c.req.param("id")))));
  app.post("/ml/embedding-indexes/:id/build", async c => {
    const result = await service(c).build(c.get("principal"), id(c.req.param("id")), c.req.raw.signal);
    return c.json(result, result.status === "FAILED" ? 422 : 200);
  });
  app.post("/ml/embedding-indexes/:id/search", async c => c.json(await service(c).search(c.get("principal"), id(c.req.param("id")), await c.req.json(), c.req.raw.signal)));
  app.post("/ml/endpoints/:id/similarity", async c => c.json(await service(c).pairwise(c.get("principal"), id(c.req.param("id")), await c.req.json(), c.req.raw.signal)));
}
