import { describe, expect, it } from "vitest";
import { binOf, countInto, driftStatus, jensenShannon, psi, referenceBins } from "../../src/ml/monitoring.js";

describe("drift statistics", () => {
  it("bins numeric values at interpolated training deciles, with a null bucket", () => {
    const b = referenceBins("x", "numeric", [...Array.from({ length: 100 }, (_, i) => i + 1), null]);
    if (b.kind !== "numeric") throw new Error("numeric");
    expect(b.edges[0]).toBeCloseTo(10.9); expect(b.edges[8]).toBeCloseTo(90.1); expect(b.edges).toHaveLength(9);
    expect(b.counts).toHaveLength(11); expect(b.counts.reduce((a, c) => a + c)).toBe(101); expect(b.counts.at(-1)).toBe(1);
    expect(binOf(b, 10.9)).toBe(0); expect(binOf(b, 11)).toBe(1); expect(binOf(b, 1e9)).toBe(9); expect(binOf(b, null)).toBe(10);
    const constant = referenceBins("c", "numeric", [5, 5, 5]);
    expect(constant.counts).toEqual([3, 0, 0]);
  });
  it("keeps the 20 most frequent categories, then other and null", () => {
    const values = [...Array.from({ length: 25 }, (_, i) => `c${i}`), "c0", "c0", true, null];
    const b = referenceBins("s", "categorical", values);
    if (b.kind !== "categorical") throw new Error("categorical");
    expect(b.categories).toHaveLength(20); expect(b.categories[0]).toBe("c0");
    // Ties sort by value, so "true" (one occurrence) loses to the c* values and counts as other, like "unseen".
    expect(b.categories).not.toContain("true");
    expect(countInto(b, ["c0", "unseen", null, true])).toEqual([1, ...Array(19).fill(0), 2, 1]);
  });
  it("computes PSI and Jensen-Shannon divergence against hand-computed values", () => {
    expect(psi([50, 50], [50, 50])).toBe(0);
    expect(psi([50, 50], [80, 20])).toBeCloseTo(.3 * Math.log(.8 / .5) + -.3 * Math.log(.2 / .5), 12); // 0.41589
    expect(psi([50, 50, 0], [50, 50, 0])).toBe(0);
    expect(psi([100, 0], [0, 100])).toBeGreaterThan(10); // empty buckets are floored, so disjoint mass is large but finite
    expect(jensenShannon([1, 0], [1, 0])).toBe(0);
    expect(jensenShannon([1, 0], [0, 1])).toBeCloseTo(1, 12);
    expect(jensenShannon([2, 0], [1, 1])).toBeCloseTo(0.3112781244591328, 12);
  });
  it("maps PSI to conventional bands", () => {
    const t = { psiModerate: .1, psiSignificant: .25 };
    expect([driftStatus(.05, t), driftStatus(.1, t), driftStatus(.24, t), driftStatus(.25, t)]).toEqual(["stable", "moderate", "moderate", "significant"]);
  });
});
