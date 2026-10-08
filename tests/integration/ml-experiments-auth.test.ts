import pg from "pg";
import { randomUUID } from "node:crypto";
import { SignJWT } from "jose";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { createApp } from "../../src/api/app.js";
import { JwtAuthenticator } from "../../src/auth/authentication.js";
import { FakeBackend } from "./ml-fake-backend.js";

const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 8 });
const admin: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...admin, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE revoked_tokens,tenant_memberships,ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

describe("public authorization, tenancy and surface for ML experiments", () => {
  it("uses public JWT membership and RLS, ignores private ML roles, and omits private-only routes", async () => {
    const backend = new FakeBackend(), backends = new Map([[backend.id, backend]]); const training = new MlTrainingService(pool, store);
    const dataset = await training.ingest(admin, { name: "labels", schema: { id: randomUUID(), columns: [{ name: "x", type: "number" }, { name: "label", type: "string" }], target: "label" }, rows: Array.from({ length: 60 }, (_, i) => ({ x: i, label: i % 2 ? "odd" : "even" })) });
    const pipeline = await training.pipeline(admin, { name: "numeric", definition: { features: ["x"], steps: [{ operation: "standard_scale" }] } });
    const secret = "public-ml-test-only-secret-at-least-32-characters", issuer = "public-ml-experiments", audience = "public-ml-api";
    for (const p of [admin, outsider]) await pool.query("INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles) VALUES($1,$2,'user',$3)", [p.tenantId, p.userId, JSON.stringify(p.roles)]);
    const jwt = async (p: Principal, roles = p.roles) => new SignJWT({ tenant_id: p.tenantId, identity_type: "user", roles }).setProtectedHeader({ alg: "HS256" }).setSubject(p.userId).setIssuer(issuer).setAudience(audience).setJti(randomUUID()).setIssuedAt().setExpirationTime("5m").sign(new TextEncoder().encode(secret));
    const make = (enabled: boolean) => createApp(pool, new JwtAuthenticator(pool, secret, issuer, audience), null, null, null, undefined, null, { enabled, backends: enabled ? backends : new Map() });
    const app = make(true);
    const call = async (path: string, token: string, body?: unknown) => app.request(path, { method: body === undefined ? "GET" : "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    const tokens = { admin: await jwt(admin), member: await jwt(admin, []), privateRole: await jwt(admin, ["ml_operator"]), other: await jwt(outsider) };
    const benchmark = { name: "api benchmark", datasetVersionId: dataset.id, featurePipelineId: pipeline.id, taskType: "classification", seed: 1, evaluation: { primaryMetric: "f1", cv: { strategy: "stratified_kfold", folds: 3 } }, candidates: [{ algorithm: "logistic_regression" }, { algorithm: "svc" }] };
    const search = { name: "api search", datasetVersionId: dataset.id, featurePipelineId: pipeline.id, taskType: "classification", seed: 1, algorithm: "random_forest_classifier", strategy: "grid", maxCandidates: 4, space: { max_depth: { type: "choice", values: [3, 6] } }, evaluation: { primaryMetric: "f1" } };
    const automl = { name: "api auto", datasetVersionId: dataset.id, target: "label", taskType: "classification", metric: "f1", seed: 1, budget: { maxCandidateRuns: 4, maxSearchCandidatesPerModel: 1 } };

    expect((await app.request("/ml/benchmarks")).status).toBe(401); expect((await app.request("/ml/models")).status).toBe(401);
    for (const [path, body] of [["/ml/benchmarks", benchmark], ["/ml/searches", search], ["/ml/automl", automl]] as const) {
      expect((await call(path, tokens.member, body)).status, path).toBe(403);
      expect((await call(path, tokens.privateRole, body)).status, `${path} with a private-only role`).toBe(403);
    }
    expect((await call("/ml/models", tokens.member)).status).toBe(200); expect((await call("/ml/benchmarks", tokens.member)).status).toBe(200);
    expect((await pool.query("SELECT count(*)::int AS n FROM ml_orchestrations")).rows[0].n).toBe(0);

    const created = await call("/ml/benchmarks", tokens.admin, benchmark); expect(created.status).toBe(202); const id = (await created.json()).orchestration.id as string;
    expect((await call("/ml/searches", tokens.admin, search)).status).toBe(202); expect((await call("/ml/automl", tokens.admin, automl)).status).toBe(202);
    expect((await call(`/ml/benchmarks/${id}`, tokens.member)).status).toBe(200);
    // Tenant isolation: another tenant sees nothing, cannot cancel, and cannot build on this tenant's dataset.
    expect((await call(`/ml/benchmarks/${id}`, tokens.other)).status).toBe(404); expect((await (await call("/ml/benchmarks", tokens.other)).json())).toEqual([]);
    expect((await call(`/ml/experiments/${id}/ranking`, tokens.other)).status).toBe(404); expect((await call(`/ml/experiments/${id}/cancel`, tokens.other, {})).status).toBe(404);
    expect((await call("/ml/benchmarks", tokens.other, benchmark)).status).toBe(404);
    // Members can read but not cancel; only tenant_admin cancels.
    expect((await call(`/ml/experiments/${id}/cancel`, tokens.member, {})).status).toBe(403);
    // The worker finalizes when the last candidate ends; a read-only member only observes.
    const worker = new LifecycleWorker(pool, { workerId: "api-test", leaseSeconds: 30, kind: "ml" }, new TrainingProcessor(pool, store, backends)); while (await worker.tick());
    const done = await (await call(`/ml/benchmarks/${id}`, tokens.member)).json(); expect(done.orchestration.status).toBe("COMPLETED"); expect(done.final).toBe(true);
    expect((await call(`/ml/experiments/${id}/cancel`, tokens.admin, {})).status).toBe(409);
    // The private-only monitoring route is not part of the public surface; disabled ML stays 503.
    expect((await call(`/ml/endpoints/${randomUUID()}/monitor`, tokens.admin)).status).toBe(404);
    expect((await make(false).request("/ml/benchmarks", { headers: { authorization: `Bearer ${tokens.admin}` } })).status).toBe(503);
  }, 60_000);
});
