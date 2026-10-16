import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { AudioBackend } from "../../src/ml/audio-backend.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { snapshotFor } from "./ml-fixtures.js";

const audioPython = process.env.ML_TEST_AUDIO_PYTHON;
const audio = (algorithm: string, hyperparameters: Record<string, unknown> = {}, opts: Parameters<typeof snapshotFor>[0] = {}) =>
  snapshotFor({ backend: "audio", algorithm, columns: [{ name: "clip", type: "audio" }, { name: "label", type: "string" }], target: "label", features: ["clip"], hyperparameters, ...opts });

describe("audio catalog and backend (Phase 4 M5)", () => {
  it("registers a fixed MFCC embedding transform and a trained softmax classifier over one audio column", () => {
    const models = Object.fromEntries(listModels("audio").map(m => [m.id, m]));
    expect(models.audio_features).toMatchObject({ taskType: "embedding", learning: "pretrained", inputs: { types: ["audio"], min: 1, max: 1 } });
    expect(models.audio_mfcc_classifier).toMatchObject({ taskType: "classification", learning: "supervised" });
    expect(models.audio_mfcc_classifier!.capabilities.scores).toBe("probability");
  });
  it("bounds the front end and refuses inconsistent settings", () => {
    const model = getModel("audio", "audio_features");
    for (const hp of [{ sampleRate: 4000 }, { nMfcc: 60 }, { frameMs: 10, hopMs: 20 }, { nMfcc: 40, nMels: 20 }, { window: "hamming" }]) expect(() => validateHyperparameters(model, hp)).toThrow();
    expect(() => validateHyperparameters(model, { sampleRate: 22050, nMfcc: 20, nMels: 64 })).not.toThrow();
  });
  it("declares CPU, one-shot, embedding and classification over audio, and refuses other modalities and steps", () => {
    const backend = new AudioBackend("unused");
    expect(backend.capabilities).toMatchObject({ artifactFormats: ["audio-npz-v1"], devices: ["cpu"], iterative: false, preprocessing: [], tasks: ["embedding", "classification"], inputs: ["audio"] });
    expect(() => backend.validate(audio("audio_mfcc_classifier"))).not.toThrow();
    expect(() => backend.validate(audio("audio_mfcc_classifier", {}, { columns: [{ name: "clip", type: "number" }, { name: "label", type: "string" }] }))).toThrow(ConflictError);
    expect(() => backend.validate(audio("audio_mfcc_classifier", {}, { steps: [{ operation: "standard_scale" }] }))).toThrow(ConflictError);
  });
  it("is disabled by default and enabled only with ML_ENABLED and its own flag", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("audio")).toBe(false);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_AUDIO_ENABLED: "true" }).backends.keys()].sort()).toEqual(["audio", "sklearn"]);
  });
});

// Every named behaviour is asserted by name: decoder known answers, the shared media vectors, MFCC invariances, learning, refusals.
it.runIf(Boolean(audioPython))("real audio runner: decoding, shared vectors, MFCC invariances, classifier, embeddings and refusals", () => {
  const out = JSON.parse(execFileSync(audioPython!, ["-I", resolve("tests/python/check_audio.py")], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).trim().split("\n").pop()!) as { results: Record<string, boolean> };
  expect(Object.entries(out.results).filter(([, value]) => value !== true).map(([name]) => name)).toEqual([]);
  expect(Object.keys(out.results)).toHaveLength(22); // a dropped check must fail the suite, not pass quietly
}, 600_000);

// CI's main test job must provide the audio interpreter (the scikit-learn venv), so these tests cannot pass by skipping.
// Only the main job runs this file (the framework jobs run explicit file lists).
it.runIf(process.env.CI === "true")("CI provides the audio interpreter in the main test job", () => {
  expect(audioPython, "ML_TEST_AUDIO_PYTHON must be set in the main test job").toBeTruthy();
});
