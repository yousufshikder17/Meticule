import pg from "pg";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { AudioBackend } from "../../src/ml/audio-backend.js";
import { AUDIO_ROWS, wav } from "../conformance/modality-rows.js";

// The real audio runner through the processor, registry and endpoints with WAV rows: a trained MFCC classifier serving
// labels, an MFCC embedding model whose vectors are returned but stored only as digest and shape, and tenant isolation.
const audioPython = process.env.ML_TEST_AUDIO_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let store: PostgresArtifactStore;
beforeAll(async () => { await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1"); store = new PostgresArtifactStore(pool); });
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); });

async function fixture() {
  const map = new Map([["audio", new AudioBackend(audioPython!)]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const dataset = await training.ingest(manager, { name: "tones", schema: { id: randomUUID(), columns: [{ name: "x", type: "audio" }, { name: "label", type: "string" }], target: "label" }, rows: AUDIO_ROWS });
  const pipeline = await training.pipeline(manager, { name: "tones", definition: { features: ["x"], steps: [] } });
  const experiment = await training.experiment(manager, "audio");
  const worker = new LifecycleWorker(pool, { workerId: "audio-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const train = async (algorithm: string, hyperparameters: Record<string, unknown>, metrics: string[]) => {
    const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "audio", algorithm, hyperparameters, seed: 3,
      split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics } } });
    while (await worker.tick());
    return training.getJob(manager, job.id);
  };
  const serve = async (attemptId: string, name: string) => {
    const registry = await models.registry(manager, { name }); const version = await models.register(manager, registry.id, { trainingRunId: attemptId });
    await models.transition(manager, version.id, "READY");
    return { version, endpoint: await models.endpoint(manager, { name, modelVersionId: version.id }) };
  };
  return { models, train, serve };
}

describe.runIf(Boolean(audioPython))("real audio backend through the durable lifecycle", () => {
  it("trains an MFCC classifier on WAV rows, registers it and serves labels; another tenant cannot reach it", async () => {
    const f = await fixture();
    const result = await f.train("audio_mfcc_classifier", { l2: 0.1 }, ["accuracy", "log_loss"]);
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].environment).toMatchObject({ backend: "audio", artifact_format: "audio-npz-v1", protocol: "meticule-audio-v1" });
    expect(result.metrics.find((m: { name: string; partition: string }) => m.name === "accuracy" && m.partition === "test")!.value).toBeGreaterThanOrEqual(0.9);
    const { endpoint, version } = await f.serve(result.attempts[0].id, "tones");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: wav(230) }, { x: wav(2600) }], new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toEqual(["low", "high"]); expect(prediction.model_version_id).toBe(version.id);
    await expect(f.models.predict(outsider, endpoint.id, [{ x: wav(230) }], new AbortController().signal)).rejects.toThrow(/not found/);
    // An undecodable clip is refused at the boundary, before any runner starts.
    await expect(f.models.predict(manager, endpoint.id, [{ x: Buffer.from("RIFF....WAVE").toString("base64") }], new AbortController().signal)).rejects.toThrow();
  }, 300_000);

  it("records the embedding dimension of MFCC features and returns vectors once while storing only digest and shape", async () => {
    const f = await fixture();
    const result = await f.train("audio_features", { nMfcc: 16 }, ["retrieval_precision_at_1"]);
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].environment.embedding_dimension).toBe("32");
    const { endpoint } = await f.serve(result.attempts[0].id, "tone-vectors");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: wav(300) }, { x: wav(3000) }, { x: wav(310) }], new AbortController().signal);
    expect(prediction.vectors).toHaveLength(3); expect(prediction.vectors[0]).toHaveLength(32);
    expect(prediction.output).toEqual({ digest: expect.stringMatching(/^[0-9a-f]{64}$/), shape: [3, 32] });
    expect((await pool.query("SELECT output FROM ml_predictions WHERE id=$1", [prediction.id])).rows[0].output).toEqual(prediction.output);
  }, 300_000);
});
