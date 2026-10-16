import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ZodError } from "zod";
import { ConflictError } from "../../src/domain/errors.js";
import { COLUMN_TYPES, DatasetSchema, SplitDefinition, TABULAR_COLUMN_TYPES, TrainingSnapshot, TrainingSpec, TRAINING_STATES, MODEL_STATES, assertTrainingTransition, assertModelTransition, compareMetrics, datasetValidator, partitionSizes, validateRows } from "../../src/ml/domain.js";
const id = "11111111-1111-4111-8111-111111111111";
const schema = DatasetSchema.parse({ id, columns: [{ name: "x", type: "number", nullable: true }, { name: "y", type: "number" }], target: "y" });
// Cells from the shared media vectors (an input is the text, or {parts: [[text, count], ...]} to concatenate).
const vectors = (JSON.parse(readFileSync(resolve("tests/fixtures/media-vectors.json"), "utf8")) as { vectors: { id: string; input: string | { parts: [string, number][] } }[] }).vectors;
const cell = (vectorId: string) => { const { input } = vectors.find(v => v.id === vectorId)!; return typeof input === "string" ? input : input.parts.map(([text, count]) => text.repeat(count)).join(""); };
describe("ML domain", () => {
  it("validates schemas, splits, seeds and rows at the boundary", () => {
    expect(DatasetSchema.safeParse({ ...schema, columns: [schema.columns[0], schema.columns[0]] }).success).toBe(false);
    expect(SplitDefinition.safeParse({ id, strategy: "random", train: .8, validation: .2, test: .2 }).success).toBe(false);
    expect(TrainingSpec.safeParse({ datasetVersionId: id, featurePipelineId: id, backend: "future-backend", algorithm: "custom", seed: -1, split: { id, strategy: "random", train: .6, validation: .2, test: .2 } }).success).toBe(false);
    expect(validateRows(schema, [{ x: null, y: 1 }])).toHaveLength(1);
    for (const row of [{ x: "bad", y: 1 }, { x: 1 }, { x: 1, y: null }, { x: Infinity, y: 1 }, { x: 1, y: 1, extra: 2 }]) expect(() => validateRows(schema, [row])).toThrow();
    expect(validateRows(schema, [{ x: 1 }], true)).toHaveLength(1);
  });
  it("declares the Phase 4 column types and an optional target that must be a number, string, boolean or text column", () => {
    expect(TABULAR_COLUMN_TYPES).toEqual(["number", "string", "boolean"]);
    expect(COLUMN_TYPES).toEqual(["number", "string", "boolean", "text", "timestamp", "image", "audio"]);
    const corpus = DatasetSchema.parse({ id, columns: [{ name: "passage", type: "text" }] });
    expect(corpus).toEqual({ id, columns: [{ name: "passage", type: "text", nullable: false }] });
    for (const type of ["number", "string", "boolean", "text"]) expect(DatasetSchema.safeParse({ id, columns: [{ name: "a", type: "number" }, { name: "t", type }], target: "t" }).success, type).toBe(true);
    for (const type of ["timestamp", "image", "audio"]) expect(DatasetSchema.safeParse({ id, columns: [{ name: "a", type: "number" }, { name: "t", type }], target: "t" }).success, type).toBe(false);
    for (const bad of [{ id, columns: [{ name: "y", type: "number" }], target: "y" }, { id, columns: [{ name: "a", type: "number" }, { name: "b", type: "number" }], target: "c" },
      { id, columns: [{ name: "a", type: "video" }, { name: "y", type: "number" }], target: "y" }, { id, columns: [] }, { id, columns: [{ name: "a", type: "number" }], target: null }]) expect(DatasetSchema.safeParse(bad).success).toBe(false);
    expect(datasetValidator(schema)).toBe("meticule-tabular-v1");
    expect(datasetValidator(corpus)).toBe("meticule-dataset-v2");
  });
  it("validates each column type's cells at the row boundary, prediction rows leaving out the target unless asked", () => {
    const media = DatasetSchema.parse({ id, columns: [{ name: "note", type: "text" }, { name: "at", type: "timestamp", nullable: true }, { name: "photo", type: "image" }, { name: "clip", type: "audio" }, { name: "label", type: "boolean" }], target: "label" });
    const row = { note: "fine", at: "2024-03-01T12:34:56+01:00", photo: cell("png-rgb-3x2"), clip: cell("wav-pcm16-mono-100ms"), label: true };
    expect(validateRows(media, [row, { ...row, at: null, photo: cell("jpeg-baseline-8x8"), clip: cell("wav-float32-stereo-44k1"), note: "x".repeat(10_000) }])).toHaveLength(2);
    for (const bad of [{ note: 1 }, { at: "2024-02-30T00:00:00Z" }, { at: "2024-03-01T12:34:56" }, { photo: "not base64" }, { photo: cell("image-gif") }, { photo: cell("png-second-ihdr") },
      { clip: cell("png-gray-1x1") }, { clip: cell("wav-float-nan-sample") }, { label: "true" }, { label: null }]) expect(() => validateRows(media, [{ ...row, ...bad }]), JSON.stringify(Object.keys(bad))).toThrow(ConflictError);
    const { label: _, ...features } = row;
    expect(validateRows(media, [features], true)).toHaveLength(1);
    expect(() => validateRows(media, [row], true)).toThrow(/unexpected or missing columns/);
    // Forecasting history carries the target, which may not be null.
    expect(validateRows(media, [row], true, true)).toHaveLength(1);
    expect(() => validateRows(media, [{ ...row, label: null }], true, true)).toThrow(/invalid label/);
    // A dataset without a target has no column that must be non-null for that reason alone.
    const unlabelled = DatasetSchema.parse({ id, columns: [{ name: "a", type: "number", nullable: true }, { name: "b", type: "string", nullable: true }] });
    expect(validateRows(unlabelled, [{ a: null, b: null }])).toHaveLength(1);
  });
  it("refuses a non-media string over 10,000 characters with the Phase 1-3 validation error", () => {
    const media = DatasetSchema.parse({ id, columns: [{ name: "note", type: "text" }, { name: "photo", type: "image" }, { name: "n", type: "number" }] });
    const row = { note: "a", photo: cell("png-gray-1x1"), n: 1 };
    for (const [cells, path] of [[{ note: "x".repeat(10_001) }, [0, "note"]], [{ n: "x".repeat(10_001) }, [0, "n"]], [{ extra: "x".repeat(10_001) }, [0, "extra"]]] as const) {
      let error: unknown;
      try { validateRows(media, [{ ...row, ...cells }]); } catch (e) { error = e; }
      expect(error).toBeInstanceOf(ZodError);
      expect((error as ZodError).issues).toMatchObject([{ code: "too_big", maximum: 10_000, path }]);
    }
    // An image cell may be as long as the base64 bound; it is then refused by its own rules, not by length.
    expect(() => validateRows(media, [{ ...row, photo: "A".repeat(20_000) }])).toThrow(ConflictError);
  });
  it("accepts a random or a chronological split, each with fractions summing to one", () => {
    const chronological = { id, strategy: "chronological", timeColumn: "at", train: .7, validation: .15, test: .15 };
    expect(SplitDefinition.parse(chronological)).toEqual(chronological);
    expect(SplitDefinition.parse({ id, strategy: "random", train: .6, validation: .2, test: .2 })).toEqual({ id, strategy: "random", train: .6, validation: .2, test: .2, stratify: false });
    for (const bad of [{ ...chronological, stratify: false }, { ...chronological, timeColumn: undefined }, { ...chronological, timeColumn: "" }, { ...chronological, test: .2 }, { ...chronological, strategy: "rolling" }, { ...chronological, strategy: undefined }])
      expect(SplitDefinition.safeParse(bad).success, JSON.stringify(bad)).toBe(false);
  });
  it("computes partition sizes as the runners split", () => {
    const random = SplitDefinition.parse({ id, strategy: "random", train: .6, validation: .2, test: .2 });
    const chronological = SplitDefinition.parse({ id, strategy: "chronological", timeColumn: "at", train: .7, validation: .15, test: .15 });
    expect(partitionSizes(random, 100)).toEqual({ train: 60, validation: 20, test: 20 });
    expect(partitionSizes(random, 7)).toEqual({ train: 4, validation: 1, test: 2 }); // floor(4.2), then floor(3 * .5)
    expect(partitionSizes(chronological, 10)).toEqual({ train: 7, validation: 1, test: 2 });
    expect(partitionSizes(chronological, 3)).toEqual({ train: 2, validation: 0, test: 1 }); // refused by validation
  });
  it("parses stored Phase 1-3 documents unchanged", () => {
    const createdAt = "2026-01-01T00:00:00.000Z", content = { key: "a".repeat(64), sha256: "b".repeat(64), size: 10, mediaType: "application/json" };
    const stored = {
      datasetVersionId: id, featurePipelineId: id, backend: "sklearn", algorithm: "ridge", hyperparameters: { alpha: 1 }, seed: 7, training: null,
      split: { id, strategy: "random", train: .6, validation: .2, test: .2, stratify: true }, evaluation: { metrics: ["mae"], cv: { strategy: "kfold", folds: 3 } }, source: null,
      dataset: { id, tenantId: id, createdAt, datasetId: id, version: 1, schema: { id, columns: [{ name: "x", type: "number", nullable: true }, { name: "kind", type: "string", nullable: false }, { name: "flag", type: "boolean", nullable: false }], target: "flag" }, content, rowCount: 60, validation: { valid: true, validator: "meticule-tabular-v1" } },
      pipeline: { id, tenantId: id, createdAt, logicalId: id, version: 1, name: "p", definition: { features: ["x", "kind"], steps: [{ operation: "one_hot_encode", parameters: {} }] } },
    };
    expect(TrainingSnapshot.parse(stored)).toEqual(stored);
  });
  it("checks every training state pair including terminal immutability", () => {
    const allowed = new Set(["QUEUED:PREPARING", "PREPARING:TRAINING", "TRAINING:EVALUATING", "EVALUATING:SAVING", "SAVING:COMPLETED", ...TRAINING_STATES.slice(0, 5).flatMap(s => [`${s}:FAILED`, `${s}:CANCELLED`])]);
    for (const from of TRAINING_STATES) for (const to of TRAINING_STATES) {
      if (allowed.has(`${from}:${to}`)) expect(() => assertTrainingTransition(from, to)).not.toThrow();
      else expect(() => assertTrainingTransition(from, to)).toThrow();
    }
  });
  it("checks every registry transition", () => {
    for (const from of MODEL_STATES) for (const to of MODEL_STATES) {
      if (["REGISTERED:READY", "REGISTERED:REJECTED", "READY:RETIRED"].includes(`${from}:${to}`)) expect(() => assertModelTransition(from, to)).not.toThrow();
      else expect(() => assertModelTransition(from, to)).toThrow();
    }
  });
  it("compares matching metrics in their declared direction", () => {
    const b = [{ name: "mae", partition: "test" as const, value: 1, direction: "lower" as const }];
    expect(compareMetrics(b, [{ ...b[0]!, value: 2 }])[0]?.regressed).toBe(true);
    expect(() => compareMetrics(b, [])).toThrow();
  });
});
