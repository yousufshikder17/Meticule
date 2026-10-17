import pg from "pg";
import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { SignJWT } from "jose";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "../../src/api/app.js";
import { JwtAuthenticator } from "../../src/auth/authentication.js";
import { withTenantSession } from "../../src/db/tenant-session.js";
import type { Principal } from "../../src/db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../../src/domain/errors.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { AudioBackend } from "../../src/ml/audio-backend.js";
import type { MlBackend } from "../../src/ml/backend.js";
import { MlEmbeddingService } from "../../src/ml/embedding-service.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { rank, type SimilarityMetric } from "../../src/ml/similarity.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { TransformersBackend } from "../../src/ml/transformers-backend.js";
import { AUDIO_ROWS, TEXT_ROWS, wav } from "../conformance/modality-rows.js";

// Embedding indexes and similarity over pgvector, end to end: real embedding models trained through the processor and
// served by endpoints; indexes built in bounded steps, searched exactly, guarded by RLS-style tenant filters and triggers.
const audioPython = process.env.ML_TEST_AUDIO_PYTHON, hfPython = process.env.ML_TEST_TRANSFORMERS_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const viewer: Principal = { ...manager, userId: "55555555-5555-4555-8555-555555555555", roles: [] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
const signal = () => new AbortController().signal;
let store: PostgresArtifactStore; let modelRoot = "";
beforeAll(async () => {
  await pool.query("SELECT 1 FROM ml_embedding_indexes LIMIT 1");
  store = new PostgresArtifactStore(pool);
  if (hfPython) { modelRoot = await mkdtemp(join(tmpdir(), "meticule-ml-embeddings-hf-")); execFileSync(hfPython, ["-I", resolve("tests/python/hf_fixtures.py"), modelRoot], { encoding: "utf8", maxBuffer: 16_000_000 }); }
}, 300_000);
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); if (modelRoot) await rm(modelRoot, { recursive: true, force: true }); });

// 40 distinct 0.1 s tones, each with a payload name. Audio build steps hold at most 16 rows (one prediction's cap), so three steps.
const CLIPS = Array.from({ length: 40 }, (_, i) => ({ x: wav(100 + 10 * i, 0.1, 8000), name: `tone-${i}` }));

/** Trains a model through the processor, registers it, marks it READY and opens an endpoint. */
async function served(backend: MlBackend, columns: { name: string; type: string }[], rows: Record<string, unknown>[], target: string | undefined, algorithm: string, hyperparameters: Record<string, unknown> = {}) {
  const map = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const dataset = await training.ingest(manager, { name: `${algorithm}-${randomUUID().slice(0, 8)}`, schema: { id: randomUUID(), columns, ...(target ? { target } : {}) }, rows });
  const pipeline = await training.pipeline(manager, { name: algorithm, definition: { features: ["x"], steps: [] } });
  const experiment = await training.experiment(manager, algorithm);
  const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: backend.id, algorithm, hyperparameters, seed: 1,
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: false } } });
  const worker = new LifecycleWorker(pool, { workerId: "embeddings-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  while (await worker.tick());
  const result = await training.getJob(manager, job.id);
  expect(result.job.status).toBe("COMPLETED");
  const registry = await models.registry(manager, { name: `${algorithm}-${randomUUID().slice(0, 8)}` });
  const version = await models.register(manager, registry.id, { trainingRunId: result.attempts[0].id });
  await models.transition(manager, version.id, "READY");
  const endpoint = await models.endpoint(manager, { name: `${algorithm}-${randomUUID().slice(0, 8)}`, modelVersionId: version.id });
  return { endpoint, dataset, embeddings: new MlEmbeddingService(pool, store, map) };
}
const audioEmbedder = () => served(new AudioBackend(audioPython!), [{ name: "x", type: "audio" }, { name: "name", type: "string" }], CLIPS, undefined, "audio_features", { sampleRate: 8000, nMfcc: 13 });
const buildAll = async (embeddings: MlEmbeddingService, indexId: string) => { let state; do state = await embeddings.build(manager, indexId, signal()); while (state.status === "BUILDING"); return state; };
const storedVectors = async (indexId: string) => (await pool.query("SELECT row_index, embedding::text AS v FROM ml_embedding_vectors WHERE index_id=$1 ORDER BY row_index", [indexId])).rows.map(r => JSON.parse(r.v) as number[]);

describe.runIf(Boolean(audioPython))("embedding indexes over pgvector (Phase 4 M6)", () => {
  it("builds in bounded steps, refuses search and concurrent builds while BUILDING, then finds the identical clip first", async () => {
    const { endpoint, dataset, embeddings } = await audioEmbedder();
    const index = await embeddings.create(manager, { name: "tones", endpointId: endpoint.id, datasetVersionId: dataset.id, metric: "cosine", payloadColumns: ["name"] });
    expect(index).toMatchObject({ status: "BUILDING", built: 0, row_count: 40, dimension: 26, metric: "cosine" });
    const first = await embeddings.build(manager, index.id, signal());
    expect(first).toMatchObject({ status: "BUILDING", built: 16 });
    await expect(embeddings.search(manager, index.id, { rows: [{ x: CLIPS[0]!.x, name: "q" }] }, signal())).rejects.toThrow(/Only READY/);
    const racing = await Promise.allSettled([embeddings.build(manager, index.id, signal()), embeddings.build(manager, index.id, signal())]);
    expect(racing.filter(r => r.status === "fulfilled")).toHaveLength(1);
    const lost = racing.find(r => r.status === "rejected") as PromiseRejectedResult;
    expect(lost.reason).toBeInstanceOf(ConflictError); expect(String(lost.reason.message)).toMatch(/being built/);
    expect(await embeddings.get(manager, index.id)).toMatchObject({ status: "BUILDING", built: 32 });
    expect(await embeddings.build(manager, index.id, signal())).toMatchObject({ status: "READY", built: 40 });
    const done = await embeddings.get(manager, index.id);
    expect(done).toMatchObject({ status: "READY", built: 40 }); expect(done.ready_at).not.toBeNull();
    await expect(embeddings.build(manager, index.id, signal())).rejects.toThrow(/READY/);
    const found = await embeddings.search(manager, index.id, { rows: [{ x: CLIPS[23]!.x, name: "q" }], k: 5 }, signal());
    const hits = found.results[0]!;
    expect(hits[0]).toMatchObject({ rowIndex: 23, payload: { name: "tone-23" } }); expect(hits[0]!.score).toBeGreaterThan(0.9999);
    expect(hits.every(h => h.score >= -1 && h.score <= 1.0001)).toBe(true);
    expect(hits.map(h => h.score)).toEqual([...hits.map(h => h.score)].sort((a, b) => b - a));
  }, 600_000);

  it("ranks exactly as the TypeScript reference for cosine, dot and euclidean", async () => {
    const { endpoint, dataset, embeddings } = await audioEmbedder();
    for (const metric of ["cosine", "dot", "euclidean"] as SimilarityMetric[]) {
      const index = await embeddings.create(manager, { name: `rank-${metric}`, endpointId: endpoint.id, datasetVersionId: dataset.id, metric });
      await buildAll(embeddings, index.id);
      const vectors = await storedVectors(index.id);
      const result = await embeddings.search(manager, index.id, { rows: [{ x: CLIPS[27]!.x, name: "q" }], k: 25 }, signal());
      const expected = rank(metric, vectors[27]!, vectors).slice(0, 25);
      expect(result.results[0]!.map(h => h.rowIndex), metric).toEqual(expected.map(e => e.index));
      // pgvector computes in float4; the reference in float64 over the same stored values, so compare relatively.
      result.results[0]!.forEach((h, i) => expect(Math.abs(h.score - expected[i]!.score)).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(expected[i]!.score))));
      expect(result.closer).toBe(metric === "euclidean" ? "lower" : "higher");
    }
  }, 900_000);

  it("enforces bounds, the embedding task, roles, tenant isolation, and the database guards", async () => {
    const { endpoint, dataset, embeddings } = await audioEmbedder();
    const index = await embeddings.create(manager, { name: "guarded", endpointId: endpoint.id, datasetVersionId: dataset.id });
    // Roles and tenants: a viewer cannot create or build; another tenant finds nothing.
    await expect(embeddings.create(viewer, { name: "v", endpointId: endpoint.id, datasetVersionId: dataset.id })).rejects.toBeInstanceOf(AuthorizationError);
    await expect(embeddings.build(viewer, index.id, signal())).rejects.toBeInstanceOf(AuthorizationError);
    await expect(embeddings.create(outsider, { name: "o", endpointId: endpoint.id, datasetVersionId: dataset.id })).rejects.toBeInstanceOf(NotFoundError);
    await expect(embeddings.build(outsider, index.id, signal())).rejects.toBeInstanceOf(NotFoundError);
    await expect(embeddings.pairwise(outsider, endpoint.id, { left: [{ x: CLIPS[0]!.x, name: "a" }], right: [{ x: CLIPS[1]!.x, name: "b" }] }, signal())).rejects.toBeInstanceOf(NotFoundError);
    await buildAll(embeddings, index.id);
    await expect(embeddings.search(outsider, index.id, { rows: [{ x: CLIPS[0]!.x, name: "q" }] }, signal())).rejects.toBeInstanceOf(NotFoundError);
    expect((await embeddings.search(viewer, index.id, { rows: [{ x: CLIPS[0]!.x, name: "q" }], k: 1 }, signal())).results[0]![0]!.rowIndex).toBe(0);
    // Bounds: k 1..100, at most 16 query rows, at most 64 rows per side (the request schema).
    for (const bad of [{ rows: [{ x: CLIPS[0]!.x, name: "q" }], k: 0 }, { rows: [{ x: CLIPS[0]!.x, name: "q" }], k: 101 }, { rows: CLIPS.slice(0, 17) }]) await expect(embeddings.search(manager, index.id, bad, signal())).rejects.toThrow();
    await expect(embeddings.pairwise(manager, endpoint.id, { left: Array.from({ length: 65 }, () => CLIPS[0]), right: CLIPS.slice(0, 1) }, signal())).rejects.toThrow();
    // Pairwise: identical clips are maximally similar.
    const matrix = (await embeddings.pairwise(manager, endpoint.id, { left: CLIPS.slice(0, 2), right: CLIPS.slice(0, 3) }, signal())).matrix;
    expect(matrix).toHaveLength(2); expect(matrix[0]).toHaveLength(3); expect(matrix[0]![0]).toBeCloseTo(1, 6); expect(matrix[1]![1]).toBeCloseTo(1, 6);
    // A non-embedding model is refused for indexes and similarity.
    const classifier = await served(new AudioBackend(audioPython!), [{ name: "x", type: "audio" }, { name: "label", type: "string" }], AUDIO_ROWS, "label", "audio_mfcc_classifier", { l2: 0.1 });
    await expect(classifier.embeddings.create(manager, { name: "c", endpointId: classifier.endpoint.id, datasetVersionId: classifier.dataset.id })).rejects.toThrow(/embeddings/);
    await expect(classifier.embeddings.pairwise(manager, classifier.endpoint.id, { left: [{ x: wav(300) }], right: [{ x: wav(300) }] }, signal())).rejects.toThrow(/embeddings/);
    // Database guards: no vectors after READY, no lifecycle reversal, immutable identity, dimension-checked vectors, no deletes.
    const zeros = (n: number) => `[${Array(n).fill(0.5).join(",")}]`;
    await expect(pool.query("INSERT INTO ml_embedding_vectors(tenant_id,index_id,dimension,row_index,embedding) VALUES($1,$2,26,4999,$3::vector)", [manager.tenantId, index.id, zeros(26)])).rejects.toThrow(/building index/);
    await expect(pool.query("UPDATE ml_embedding_indexes SET status='BUILDING', ready_at=NULL WHERE id=$1", [index.id])).rejects.toThrow(/transition/);
    await expect(pool.query("UPDATE ml_embedding_indexes SET name='renamed' WHERE id=$1", [index.id])).rejects.toThrow(/immutable/);
    await expect(pool.query("DELETE FROM ml_embedding_indexes WHERE id=$1", [index.id])).rejects.toThrow(/append-only/);
    await expect(pool.query("UPDATE ml_embedding_vectors SET payload='{}' WHERE index_id=$1", [index.id])).rejects.toThrow();
    const building = await embeddings.create(manager, { name: "dims", endpointId: endpoint.id, datasetVersionId: dataset.id });
    await expect(pool.query("INSERT INTO ml_embedding_vectors(tenant_id,index_id,dimension,row_index,embedding) VALUES($1,$2,26,0,$3::vector)", [manager.tenantId, building.id, zeros(25)])).rejects.toThrow();
    await expect(pool.query("INSERT INTO ml_embedding_vectors(tenant_id,index_id,dimension,row_index,embedding) VALUES($1,$2,25,0,$3::vector)", [manager.tenantId, building.id, zeros(25)])).rejects.toThrow();
  }, 900_000);

  it("holds every runner call to one prediction's rows and media: build steps are cut, pairwise sides are one request", async () => {
    const { endpoint, dataset, embeddings } = await audioEmbedder();
    // Pairwise: both sides share the audio row cap (16), checked before any cell is decoded, and one media budget (120 s).
    await expect(embeddings.pairwise(manager, endpoint.id, { left: CLIPS.slice(0, 9), right: CLIPS.slice(9, 17) }, signal())).rejects.toThrow(/at most 16 rows/);
    expect((await embeddings.pairwise(manager, endpoint.id, { left: CLIPS.slice(0, 8), right: CLIPS.slice(8, 16) }, signal())).matrix).toHaveLength(8);
    const long = Array.from({ length: 16 }, (_, i) => ({ x: wav(200 + 15 * i, 8, 8000), name: `long-${i}` })); // 8 s each
    // 8 + 8 clips of 8 s are 128 s: refused together, although either side alone (64 s) is within one prediction's budget.
    await expect(embeddings.pairwise(manager, endpoint.id, { left: long.slice(0, 8), right: long.slice(8) }, signal())).rejects.toThrow(/exceeds 120 seconds/);
    // Build: 16 stored clips of 8 s fit the row cap but not the media budget, so the first step stops at 15 clips (120 s).
    const training = new MlTrainingService(pool, store);
    const longSet = await training.ingest(manager, { name: "long clips", schema: { id: randomUUID(), columns: [{ name: "x", type: "audio" }, { name: "name", type: "string" }] }, rows: long });
    const index = await embeddings.create(manager, { name: "long", endpointId: endpoint.id, datasetVersionId: longSet.id, payloadColumns: ["name"] });
    expect(await embeddings.build(manager, index.id, signal())).toMatchObject({ status: "BUILDING", built: 15 });
    expect(await embeddings.build(manager, index.id, signal())).toMatchObject({ status: "READY", built: 16 });
    const hits = (await embeddings.search(manager, index.id, { rows: [{ x: long[15]!.x, name: "q" }], k: 1 }, signal())).results[0]!;
    expect(hits[0]).toMatchObject({ rowIndex: 15, payload: { name: "long-15" } });
    expect(dataset.rowCount).toBe(40);
  }, 600_000);
});

describe.runIf(Boolean(audioPython))("embedding indexes under the API role", () => {
  it("creates, builds and searches inside a tenant session (restricted role, row-level security), invisible to another tenant", async () => {
    const { endpoint, dataset } = await audioEmbedder();
    const backends = new Map<string, MlBackend>([["audio", new AudioBackend(audioPython!)]]);
    const asTenant = <T>(principal: Principal, work: (service: MlEmbeddingService, database: pg.Pool) => Promise<T>) =>
      withTenantSession(pool, principal.tenantId, database => work(new MlEmbeddingService(database, store, backends), database));
    const index = await asTenant(manager, service => service.create(manager, { name: "rls", endpointId: endpoint.id, datasetVersionId: dataset.id, payloadColumns: ["name"] }));
    let state;
    do state = await asTenant(manager, service => service.build(manager, index.id, signal())); while (state.status === "BUILDING");
    expect(state.status).toBe("READY");
    const hits = await asTenant(manager, service => service.search(manager, index.id, { rows: [{ x: CLIPS[9]!.x, name: "q" }], k: 1 }, signal()));
    expect(hits.results[0]![0]).toMatchObject({ rowIndex: 9, payload: { name: "tone-9" } });
    // Row-level security: another tenant's session sees no index and no vectors, even reading the tables directly.
    const seen = await asTenant(outsider, (_, database) => database.query("SELECT (SELECT count(*)::int FROM ml_embedding_indexes) AS i, (SELECT count(*)::int FROM ml_embedding_vectors) AS v"));
    expect(seen.rows[0]).toEqual({ i: 0, v: 0 });
  }, 600_000);
  it("serves the embedding routes over public JWT membership: members read and search, only tenant_admin writes, other tenants see nothing", async () => {
    const { endpoint, dataset } = await audioEmbedder();
    const secret = "public-ml-test-only-secret-at-least-32-characters", issuer = "public-ml-embeddings", audience = "public-ml-api";
    await pool.query("DELETE FROM tenant_memberships WHERE identity_id=$1", [manager.userId]);
    for (const p of [manager, outsider]) await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,'user',$3)", [p.tenantId, p.userId, JSON.stringify(p.roles)]);
    const jwt = (p: Principal, roles = p.roles) => new SignJWT({ tenant_id: p.tenantId, identity_type: "user", roles }).setProtectedHeader({ alg: "HS256" }).setSubject(p.userId).setIssuer(issuer).setAudience(audience).setJti(randomUUID()).setIssuedAt().setExpirationTime("5m").sign(new TextEncoder().encode(secret));
    const app = createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience), null, null, null, undefined, null, { enabled: true, backends: new Map<string, MlBackend>([["audio", new AudioBackend(audioPython!)]]) });
    const call = (path: string, token: string, body?: unknown) => app.request(path, { method: body === undefined ? "GET" : "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    const admin = await jwt(manager), member = await jwt(manager, []), other = await jwt(outsider);
    const create = { name: "http", endpointId: endpoint.id, datasetVersionId: dataset.id, payloadColumns: ["name"] };
    expect((await app.request("/ml/embedding-indexes")).status).toBe(401);
    expect((await call("/ml/embedding-indexes", member, create)).status).toBe(403);
    const created = await call("/ml/embedding-indexes", admin, create); expect(created.status).toBe(201); const index = await created.json();
    expect((await call(`/ml/embedding-indexes/${index.id}/build`, member, {})).status).toBe(403);
    // Each HTTP build step embeds at most one prediction's rows: 16 for an audio model.
    let state = await (await call(`/ml/embedding-indexes/${index.id}/build`, admin, {})).json();
    expect(state).toMatchObject({ status: "BUILDING", built: 16 });
    while (state.status === "BUILDING") state = await (await call(`/ml/embedding-indexes/${index.id}/build`, admin, {})).json();
    expect(state.status).toBe("READY");
    const search = await call(`/ml/embedding-indexes/${index.id}/search`, member, { rows: [{ x: CLIPS[9]!.x, name: "q" }], k: 1 });
    expect(search.status).toBe(200); expect((await search.json()).results[0][0]).toMatchObject({ rowIndex: 9, payload: { name: "tone-9" } });
    const pair = await call(`/ml/endpoints/${endpoint.id}/similarity`, member, { left: [{ x: CLIPS[0]!.x, name: "a" }], right: [{ x: CLIPS[0]!.x, name: "a" }] });
    expect(pair.status).toBe(200); expect((await pair.json()).matrix[0][0]).toBeCloseTo(1, 6);
    // Both sides together are one prediction's rows: 9 + 8 audio clips is a 409 for any member.
    expect((await call(`/ml/endpoints/${endpoint.id}/similarity`, member, { left: CLIPS.slice(0, 9), right: CLIPS.slice(9, 17) })).status).toBe(409);
    expect((await (await call("/ml/embedding-indexes", member)).json()).map((i: { id: string }) => i.id)).toEqual([index.id]);
    expect(await (await call("/ml/embedding-indexes", other)).json()).toEqual([]);
    expect((await call(`/ml/embedding-indexes/${index.id}`, other)).status).toBe(404);
    expect((await call(`/ml/embedding-indexes/${index.id}/search`, other, { rows: [{ x: CLIPS[0]!.x, name: "q" }] })).status).toBe(404);
    expect((await call(`/ml/endpoints/${endpoint.id}/similarity`, other, { left: [{ x: CLIPS[0]!.x, name: "a" }], right: [{ x: CLIPS[0]!.x, name: "a" }] })).status).toBe(404);
  }, 600_000);
});

describe.runIf(Boolean(audioPython && hfPython))("embedding indexes over a transformers text embedder", () => {
  it("indexes text embeddings from the offline fixture and finds the identical sentence first", async () => {
    const { endpoint, dataset, embeddings } = await served(new TransformersBackend(hfPython!, modelRoot), [{ name: "x", type: "text" }, { name: "label", type: "string" }], TEXT_ROWS, "label", "text_embedder", { model: "tiny-bert-encoder" });
    const index = await embeddings.create(manager, { name: "sentences", endpointId: endpoint.id, datasetVersionId: dataset.id, payloadColumns: ["label"] });
    expect((await buildAll(embeddings, index.id)).status).toBe("READY");
    const hits = (await embeddings.search(manager, index.id, { rows: [{ x: TEXT_ROWS[7]!.x }], k: 3 }, signal())).results[0]!;
    expect(hits[0]!.score).toBeGreaterThan(0.9999);
    expect(TEXT_ROWS[hits[0]!.rowIndex]!.x).toBe(TEXT_ROWS[7]!.x);
  }, 900_000);
});
