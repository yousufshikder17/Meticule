import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { rank, similarity, similarityMatrix } from "../../src/ml/similarity.js";

describe("vector similarity", () => {
  it("computes known values for cosine, dot and euclidean", () => {
    expect(similarity("cosine", [1, 0], [1, 0])).toBe(1);
    expect(similarity("cosine", [1, 0], [0, 1])).toBe(0);
    expect(similarity("cosine", [1, 0], [-1, 0])).toBe(-1);
    expect(similarity("cosine", [1, 1], [1, 0])).toBeCloseTo(Math.SQRT1_2, 12);
    expect(similarity("dot", [1, 2, 3], [4, 5, 6])).toBe(32);
    expect(similarity("euclidean", [0, 0], [3, 4])).toBe(5);
  });
  it("makes cosine invariant to positive scaling, unlike dot and euclidean", () => {
    expect(similarity("cosine", [2, 3], [5, 1])).toBeCloseTo(similarity("cosine", [20, 30], [0.5, 0.1]), 12);
    expect(similarity("dot", [2, 3], [5, 1])).not.toBeCloseTo(similarity("dot", [20, 30], [5, 1]), 6);
  });
  it("refuses zero vectors for cosine, mismatched dimensions, empty and non-finite input, never returning NaN", () => {
    expect(() => similarity("cosine", [0, 0], [1, 0])).toThrow(ConflictError);
    expect(similarity("dot", [0, 0], [1, 0])).toBe(0);
    for (const [a, b] of [[[1, 2], [1, 2, 3]], [[], []], [[NaN, 1], [1, 1]], [[Infinity, 1], [1, 1]]] as [number[], number[]][]) {
      for (const metric of ["cosine", "dot", "euclidean"] as const) expect(() => similarity(metric, a, b)).toThrow(ConflictError);
    }
  });
  it("ranks closest first with a deterministic tie-break on the lower index", () => {
    const candidates = [[0, 1], [1, 0], [1, 0], [-1, 0]];
    expect(rank("cosine", [1, 0], candidates).map(r => r.index)).toEqual([1, 2, 0, 3]);
    expect(rank("euclidean", [1, 0], candidates).map(r => r.index)).toEqual([1, 2, 0, 3]);
    expect(rank("dot", [2, 0], candidates).map(r => r.score)).toEqual([2, 2, 0, -2]);
  });
  it("builds a left x right matrix", () => {
    expect(similarityMatrix("dot", [[1, 0], [0, 1]], [[1, 2], [3, 4], [5, 6]])).toEqual([[1, 3, 5], [2, 4, 6]]);
  });
});
