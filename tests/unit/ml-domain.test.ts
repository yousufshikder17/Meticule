import { describe, expect, it } from "vitest";
import { DatasetSchema, SplitDefinition, TrainingSpec, TRAINING_STATES, MODEL_STATES, assertTrainingTransition, assertModelTransition, compareMetrics, validateRows } from "../../src/ml/domain.js";
const id = "11111111-1111-4111-8111-111111111111";
const schema = DatasetSchema.parse({ id, columns: [{ name: "x", type: "number", nullable: true }, { name: "y", type: "number" }], target: "y" });
describe("ML domain", () => {
  it("validates schemas, splits, seeds and rows at the boundary", () => {
    expect(DatasetSchema.safeParse({ ...schema, columns: [schema.columns[0], schema.columns[0]] }).success).toBe(false);
    expect(SplitDefinition.safeParse({ id, strategy: "random", train: .8, validation: .2, test: .2 }).success).toBe(false);
    expect(TrainingSpec.safeParse({ datasetVersionId: id, featurePipelineId: id, backend: "future-backend", algorithm: "custom", seed: -1, split: { id, strategy: "random", train: .6, validation: .2, test: .2 } }).success).toBe(false);
    expect(validateRows(schema, [{ x: null, y: 1 }])).toHaveLength(1);
    for (const row of [{ x: "bad", y: 1 }, { x: 1 }, { x: 1, y: null }, { x: Infinity, y: 1 }, { x: 1, y: 1, extra: 2 }]) expect(() => validateRows(schema, [row])).toThrow();
    expect(validateRows(schema, [{ x: 1 }], true)).toHaveLength(1);
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
