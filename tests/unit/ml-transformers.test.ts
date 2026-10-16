import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { mlConfiguration } from "../../src/ml/configuration.js";
import { contractFor, ML_BACKEND_CONTRACT } from "../../src/ml/contract.js";
import { getModel, listModels, validateHyperparameters } from "../../src/ml/model-catalog.js";
import { TransformersBackend } from "../../src/ml/transformers-backend.js";
import { snapshotFor } from "./ml-fixtures.js";

const root = mkdtempSync(join(tmpdir(), "meticule-model-root-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
const text = (algorithm: string, hyperparameters: Record<string, unknown> = { model: "tiny-bert-encoder" }, opts: Parameters<typeof snapshotFor>[0] = {}) =>
  snapshotFor({ backend: "transformers", algorithm, columns: [{ name: "x", type: "text" }, { name: "label", type: "string" }], target: "label", features: ["x"], hyperparameters, ...opts });

describe("transformers catalog (Phase 4 M4)", () => {
  it("registers six single-input text and image models with honest learning modes and scores", () => {
    const models = Object.fromEntries(listModels("transformers").map(m => [m.id, m]));
    expect(Object.keys(models).sort()).toEqual(["image_classifier", "image_embedder", "image_embedding_classifier", "text_classifier", "text_embedder", "text_embedding_classifier"]);
    expect(models.text_classifier).toMatchObject({ taskType: "classification", learning: "pretrained", inputs: { types: ["text"], min: 1, max: 1 } });
    expect(models.image_embedder).toMatchObject({ taskType: "embedding", learning: "pretrained", inputs: { types: ["image"], min: 1, max: 1 } });
    expect(models.text_embedding_classifier!.learning).toBe("supervised");
    expect(models.text_classifier!.capabilities.scores).toBe("probability");
    expect(models.text_embedder!.capabilities.scores).toBe("none");
  });
  it("bounds every parameter and refuses unknown ones (no trust_remote_code or revision passthrough)", () => {
    const embedder = getModel("transformers", "text_embedder");
    for (const hp of [{ maxLength: 4 }, { maxLength: 1024 }, { batchSize: 0 }, { pooling: "max" }, { trust_remote_code: true }, { revision: "main" }, { model: "../escape" }]) expect(() => validateHyperparameters(embedder, hp)).toThrow();
  });
});

describe("transformers backend validation", () => {
  const backend = new TransformersBackend("unused", root);
  it("declares CPU, fp32, one-shot, classification and embedding over text and image, no tabular preprocessing", () => {
    expect(backend.capabilities).toMatchObject({ artifactFormats: ["transformers-reference-v1"], devices: ["cpu"], precisions: ["fp32"], iterative: false, preprocessing: [], tasks: ["classification", "embedding"], inputs: ["text", "image"] });
    expect(() => backend.validate(text("text_embedder"))).not.toThrow();
  });
  it("requires a provisioned model name, a text feature, and no preprocessing", () => {
    expect(() => backend.validate(text("text_embedder", {}))).toThrow(/provisioned model/);
    expect(() => backend.validate(text("text_embedder", { model: "tiny" }, { columns: [{ name: "x", type: "number" }, { name: "label", type: "string" }] }))).toThrow(ConflictError);
    expect(() => backend.validate(text("text_embedder", { model: "tiny" }, { steps: [{ operation: "standard_scale" }] }))).toThrow(ConflictError);
  });
  it("refuses a tabular feature even when the snapshot alone would select contract v1", () => {
    // Classification over number/string columns with a random split is a v1 snapshot; the model's declared text input still holds.
    const tabular = text("text_classifier", { model: "tiny" }, { columns: [{ name: "x", type: "number" }, { name: "label", type: "string" }] });
    expect(contractFor(tabular, "classification")).toBe(ML_BACKEND_CONTRACT);
    expect(() => backend.validate(tabular)).toThrow(/feature column types/);
  });
  it("refuses cross-validation for pretrained models and allows it for trained heads", () => {
    const cv = { cv: { strategy: "kfold" as const, folds: 3 } };
    expect(() => backend.validate(text("text_classifier", { model: "tiny" }, { evaluation: cv }))).toThrow(/pretrained/);
    expect(() => backend.validate(text("text_embedding_classifier", { model: "tiny" }, { evaluation: cv }))).not.toThrow();
  });
});

describe("ML configuration for transformers", () => {
  it("stays disabled by default, needs ML_ENABLED, and requires an absolute existing model root", () => {
    expect(mlConfiguration({ ML_ENABLED: "true" }).backends.has("transformers")).toBe(false);
    expect(mlConfiguration({ ML_TRANSFORMERS_ENABLED: "true", ML_TRANSFORMERS_MODEL_ROOT: root }).backends.size).toBe(0);
    expect([...mlConfiguration({ ML_ENABLED: "true", ML_TRANSFORMERS_ENABLED: "true", ML_TRANSFORMERS_MODEL_ROOT: root }).backends.keys()].sort()).toEqual(["sklearn", "transformers"]);
    for (const bad of ["", "relative/models", join(root, "missing")]) expect(() => mlConfiguration({ ML_ENABLED: "true", ML_TRANSFORMERS_ENABLED: "true", ML_TRANSFORMERS_MODEL_ROOT: bad })).toThrow(/absolute path to an existing directory/);
  });
});
