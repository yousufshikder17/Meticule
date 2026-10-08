import { describe, expect, it } from "vitest";
import { getModel } from "../../src/ml/model-catalog.js";
import { generateCandidates, SearchSpace } from "../../src/ml/search.js";

const forest = getModel("sklearn", "random_forest_classifier");
const space = (raw: unknown) => SearchSpace.parse(raw);
const generate = (model = forest, s: unknown = {}, fixed = {}, strategy = "grid", max = 50, seed = 1) => generateCandidates(model, space(s), fixed, strategy, max, seed);

describe("hyperparameter search", () => {
  it("enumerates a grid in a stable order: sorted parameters, declared values, last parameter fastest", () => {
    const points = generate(forest, { n_estimators: { type: "choice", values: [100, 300] }, max_depth: { type: "int", min: 4, max: 12, step: 4 } });
    expect(points).toEqual([[4, 100], [4, 300], [8, 100], [8, 300], [12, 100], [12, 300]].map(([max_depth, n_estimators]) => ({ max_depth, n_estimators })));
    expect(generate(forest, { n_estimators: { type: "choice", values: [100, 300] }, max_depth: { type: "int", min: 4, max: 12, step: 4 } })).toEqual(points);
  });
  it("expands float ranges on linear and log scales with exact endpoints", () => {
    const lr = getModel("sklearn", "logistic_regression");
    expect(generate(lr, { C: { type: "float", min: .01, max: 100, log: true, steps: 5 } }).map(p => p.C)).toEqual([.01, .1, 1, 10, 100]);
    expect(generate(lr, { C: { type: "float", min: 1, max: 3, steps: 3 } }).map(p => p.C)).toEqual([1, 2, 3]);
  });
  it("rejects grids above the candidate cap instead of silently truncating", () => {
    expect(() => generate(forest, { n_estimators: { type: "int", min: 1, max: 100 } }, {}, "grid", 50)).toThrow(/above maxCandidates/);
    expect(generate(forest, { n_estimators: { type: "int", min: 1, max: 100 } }, {}, "random", 7)).toHaveLength(7);
  });
  it("drops combinations the model cannot use and fails when none remain", () => {
    const svc = getModel("sklearn", "svc");
    const points = generate(svc, { kernel: { type: "choice", values: ["linear", "poly"] }, degree: { type: "choice", values: [2, 4] } });
    expect(points).toEqual([{ degree: 2, kernel: "poly" }, { degree: 4, kernel: "poly" }]);
    expect(() => generate(svc, { degree: { type: "choice", values: [2, 4] } }, { kernel: "linear" })).toThrow(/No valid/);
  });
  it("samples randomly but reproducibly, without duplicates, and stops when the space is exhausted", () => {
    const s = { n_estimators: { type: "int", min: 10, max: 500 }, max_depth: { type: "choice", values: [null, 3, 6, 12] }, max_features: { type: "choice", values: ["sqrt", "log2"] } };
    const a = generate(forest, s, {}, "random", 12, 99);
    expect(a).toHaveLength(12); expect(new Set(a.map(p => JSON.stringify(p))).size).toBe(12);
    expect(generate(forest, s, {}, "random", 12, 99)).toEqual(a); expect(generate(forest, s, {}, "random", 12, 100)).not.toEqual(a);
    expect(generate(forest, { max_depth: { type: "choice", values: [3, 6] } }, {}, "random", 30, 5)).toHaveLength(2);
    const floats = generate(getModel("sklearn", "ridge"), { alpha: { type: "float", min: .001, max: 1000, log: true } }, {}, "random", 20, 3);
    expect(floats).toHaveLength(20); for (const p of floats) { expect(p.alpha).toBeGreaterThanOrEqual(.001); expect(p.alpha).toBeLessThanOrEqual(1000); }
  });
  it("rejects unsupported parameters, invalid values, bad ranges and empty or oversized spaces", () => {
    const bad = (s: unknown, fixed = {}, strategy = "grid", model = forest) => () => generate(model, s, fixed, strategy);
    expect(() => SearchSpace.parse({})).toThrow(/empty/);
    expect(bad({ n_jobs: { type: "choice", values: [1] } })).toThrow(/unsupported search parameter/);
    expect(bad({ n_estimators: { type: "choice", values: [10, 0] } })).toThrow(/invalid value/);
    expect(bad({ n_estimators: { type: "int", min: 10, max: 5 } })).toThrow(/min exceeds max/);
    expect(bad({ n_estimators: { type: "int", min: 1, max: 5000, step: 1 } })).toThrow(/too many/);
    expect(bad({ max_depth: { type: "choice", values: [3, 3] } })).toThrow(/duplicate/);
    expect(bad({ max_depth: { type: "choice", values: ["deep"] } })).toThrow(/invalid value/);
    expect(bad({ n_estimators: { type: "float", min: 1.5, max: 4.5, steps: 2 } })).toThrow(/invalid value/);
    expect(bad({ n_estimators: { type: "choice", values: [10] } }, { n_estimators: 20 })).toThrow(/both fixed and searched/);
    expect(bad({ n_estimators: { type: "choice", values: [10] } }, { bogus: 1 })).toThrow();
    expect(bad({ C: { type: "float", min: 0, max: 10, log: true, steps: 3 } }, {}, "grid", getModel("sklearn", "svc"))).toThrow(/min > 0|invalid value/);
    expect(bad({ C: { type: "float", min: 1, max: 10 } }, {}, "grid", getModel("sklearn", "svc"))).toThrow(/needs steps/);
    expect(bad({ n_estimators: { type: "choice", values: [10] } }, {}, "bayesian")).toThrow(/Unsupported search strategy/);
    expect(() => SearchSpace.parse({ n_estimators: { type: "choice", values: [] } })).toThrow();
    expect(() => SearchSpace.parse(Object.fromEntries(Array.from({ length: 11 }, (_, i) => [`p${i}`, { type: "choice", values: [1] }])))).toThrow(/At most 10/);
    expect(() => SearchSpace.parse({ n_estimators: { type: "choice", values: [1], extra: true } })).toThrow();
  });
});
