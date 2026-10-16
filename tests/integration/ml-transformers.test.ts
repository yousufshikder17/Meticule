import pg from "pg";
import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Principal } from "../../src/db/types.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { TransformersBackend } from "../../src/ml/transformers-backend.js";
import { AUDIO_ROWS, IMAGE_ROWS, SPEECH_ROWS, TEXT_ROWS, wav } from "../conformance/modality-rows.js";

// The real transformers runner through the processor, registry and endpoints, over tiny offline fixtures built into a
// temporary model root. The models are random: these tests prove the lifecycle, not semantic quality.
const hfPython = process.env.ML_TEST_TRANSFORMERS_PYTHON;
const pool = new pg.Pool({ connectionString: process.env.TEST_DATABASE_URL ?? process.env.DATABASE_URL ?? "postgres://agent:agent@localhost:5432/agent_platform", max: 10 });
const manager: Principal = { tenantId: "11111111-1111-4111-8111-111111111111", userId: "22222222-2222-4222-8222-222222222222", roles: ["tenant_admin"] };
const outsider: Principal = { ...manager, tenantId: "33333333-3333-4333-8333-333333333333" };
let modelRoot: string; let store: PostgresArtifactStore;
beforeAll(async () => {
  await pool.query("SELECT 1 FROM ml_orchestrations LIMIT 1");
  store = new PostgresArtifactStore(pool);
  modelRoot = await mkdtemp(join(tmpdir(), "meticule-ml-hf-models-"));
  if (hfPython) execFileSync(hfPython, ["-I", resolve("tests/python/hf_fixtures.py"), modelRoot], { encoding: "utf8", maxBuffer: 16_000_000 });
}, 300_000);
beforeEach(async () => { await pool.query("TRUNCATE ml_blobs,ml_dataset_schemas,ml_datasets,ml_feature_pipelines,ml_experiments,ml_registry_entries,runs,agents CASCADE"); });
afterAll(async () => { await pool.end(); if (modelRoot) await rm(modelRoot, { recursive: true, force: true }); });

async function fixture(modality: "text" | "image" | "audio", speech = false) {
  const map = new Map([["transformers", new TransformersBackend(hfPython!, modelRoot)]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, map);
  const rows = speech ? SPEECH_ROWS : modality === "text" ? TEXT_ROWS : modality === "image" ? IMAGE_ROWS : AUDIO_ROWS;
  const dataset = await training.ingest(manager, { name: modality, schema: { id: randomUUID(), columns: [{ name: "x", type: modality }, { name: "label", type: speech ? "text" : "string" }], target: "label" }, rows });
  const pipeline = await training.pipeline(manager, { name: modality, definition: { features: ["x"], steps: [] } });
  const experiment = await training.experiment(manager, `transformers-${modality}`);
  const worker = new LifecycleWorker(pool, { workerId: "hf-test", leaseSeconds: 120, kind: "ml" }, new TrainingProcessor(pool, store, map));
  const train = async (algorithm: string, hyperparameters: Record<string, unknown>, metrics: string[]) => {
    const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "transformers", algorithm, hyperparameters, seed: 3,
      split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: false }, evaluation: { metrics } } });
    while (await worker.tick());
    return training.getJob(manager, job.id);
  };
  const serve = async (attemptId: string, name: string) => {
    const registry = await models.registry(manager, { name }); const version = await models.register(manager, registry.id, { trainingRunId: attemptId });
    await models.transition(manager, version.id, "READY");
    return { version, endpoint: await models.endpoint(manager, { name, modelVersionId: version.id }) };
  };
  return { models, train, serve, rows };
}

describe.runIf(Boolean(hfPython))("real transformers backend through the durable lifecycle", () => {
  it("trains a softmax head over a frozen text encoder, registers it and serves labels; another tenant cannot reach it", async () => {
    const f = await fixture("text");
    const result = await f.train("text_embedding_classifier", { model: "tiny-bert-encoder", l2: 0.01 }, ["accuracy", "log_loss"]);
    expect(result.job.status).toBe("COMPLETED");
    const attempt = result.attempts[0];
    expect(attempt.environment).toMatchObject({ backend: "transformers", artifact_format: "transformers-reference-v1", model_name: "tiny-bert-encoder", model_family: "bert" });
    expect(JSON.stringify(attempt.environment)).not.toContain(modelRoot.slice(3)); // the model root path is never recorded
    expect(result.metrics.find((m: { name: string; partition: string }) => m.name === "accuracy" && m.partition === "train")!.value).toBeGreaterThanOrEqual(0.9);
    const { endpoint, version } = await f.serve(attempt.id, "text-head");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: "good great movie" }, { x: "awful sad movie" }], new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toEqual(["positive", "negative"]); expect(prediction.model_version_id).toBe(version.id);
    await expect(f.models.predict(outsider, endpoint.id, [{ x: "good" }], new AbortController().signal)).rejects.toThrow(/not found/);
  }, 600_000);

  it("records the embedding dimension and returns vectors once while storing only their digest and shape", async () => {
    const f = await fixture("text");
    const result = await f.train("text_embedder", { model: "tiny-distilbert-encoder" }, ["retrieval_precision_at_1"]);
    expect(result.job.status).toBe("COMPLETED");
    expect(result.attempts[0].environment.embedding_dimension).toBe("32");
    const { endpoint } = await f.serve(result.attempts[0].id, "text-vectors");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: "good movie" }, { x: "bad film" }], new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED");
    expect(prediction.vectors).toHaveLength(2); expect(prediction.vectors[0]).toHaveLength(32);
    expect(prediction.output).toEqual({ digest: expect.stringMatching(/^[0-9a-f]{64}$/), shape: [2, 32] });
    const stored = (await pool.query("SELECT output FROM ml_predictions WHERE id=$1", [prediction.id])).rows[0].output;
    expect(stored).toEqual(prediction.output); expect(JSON.stringify(stored)).not.toMatch(/\d\.\d{6}/);
  }, 600_000);

  it("binds a pretrained image classifier's labels and serves them; an encoder offered as a classifier fails without leaking paths", async () => {
    const f = await fixture("image");
    const result = await f.train("image_classifier", { model: "tiny-vit-colours" }, ["accuracy"]);
    expect(result.job.status).toBe("COMPLETED");
    const { endpoint } = await f.serve(result.attempts[0].id, "image-labels");
    const prediction = await f.models.predict(manager, endpoint.id, f.rows.slice(0, 3).map(r => ({ x: r.x })), new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output.every((label: string) => label === "red" || label === "blue")).toBe(true);
    const mismatch = await f.train("image_classifier", { model: "tiny-vit-encoder" }, ["accuracy"]); // an encoder offered as a classifier
    expect(mismatch.job.status).toBe("FAILED"); expect(JSON.stringify(mismatch.attempts[0].failure)).not.toContain(modelRoot.slice(3));
  }, 600_000);

  it("serves a pretrained speech recogniser: train records WER and CER, the endpoint returns transcripts", async () => {
    const f = await fixture("audio", true);
    const result = await f.train("speech_recognizer", { model: "tiny-wav2vec2-ctc" }, ["wer", "cer"]);
    expect(result.job.status).toBe("COMPLETED");
    expect(new Set(result.metrics.map((m: { name: string }) => m.name))).toEqual(new Set(["wer", "cer"]));
    const { endpoint } = await f.serve(result.attempts[0].id, "speech");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: wav(310) }, { x: wav(620) }], new AbortController().signal);
    expect(prediction.status).toBe("COMPLETED"); expect(prediction.output).toEqual(["h", "h"]); // the fixture's forced transcript
  }, 600_000);

  it("binds a pretrained audio classifier's labels and serves them", async () => {
    const f = await fixture("audio");
    const result = await f.train("audio_classifier", { model: "tiny-wav2vec2-tones" }, ["accuracy"]);
    expect(result.job.status).toBe("COMPLETED");
    const { endpoint } = await f.serve(result.attempts[0].id, "tones");
    const prediction = await f.models.predict(manager, endpoint.id, [{ x: wav(250) }, { x: wav(2600) }], new AbortController().signal);
    expect(prediction.output.every((label: string) => label === "low" || label === "high")).toBe(true);
  }, 600_000);
});
