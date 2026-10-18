import { describe, expect, it } from "vitest";
import { confusionMatrix, precisionRecallCurve, rocCurve } from "../../src/ml/reports.js";

describe("report curves", () => {
  it("counts a confusion matrix with sorted classes (rows actual, columns predicted)", () => {
    expect(confusionMatrix(["b", "a", "a", "b"], ["b", "a", "b", "b"])).toEqual({ classes: ["a", "b"], matrix: [[1, 1], [0, 2]], accuracy: .75 });
  });
  it("matches scikit-learn's ROC AUC and average precision on a hand-checked example", () => {
    // sklearn: roc_auc_score([1,0,1,0],[.9,.8,.7,.6]) = 0.75; average_precision_score(...) = 0.8333...
    const scores = [.9, .8, .7, .6], labels = [true, false, true, false];
    const roc = rocCurve(scores, labels);
    expect(roc.points).toEqual([{ fpr: 0, tpr: 0 }, { fpr: 0, tpr: .5 }, { fpr: .5, tpr: .5 }, { fpr: .5, tpr: 1 }, { fpr: 1, tpr: 1 }]);
    expect(roc.auc).toBeCloseTo(.75, 12);
    expect(precisionRecallCurve(scores, labels).averagePrecision).toBeCloseTo(5 / 6, 12);
  });
  it("treats tied scores as one threshold and refuses one-class labels", () => {
    expect(rocCurve([.5, .5, .1], [true, false, false]).points).toEqual([{ fpr: 0, tpr: 0 }, { fpr: .5, tpr: 1 }, { fpr: 1, tpr: 1 }]);
    expect(rocCurve([.5, .5, .1], [true, false, false]).auc).toBeCloseTo(.75, 12);
    expect(() => rocCurve([.1, .2], [true, true])).toThrow(/both/);
    expect(() => precisionRecallCurve([.1, .2], [false, false])).toThrow(/positive/);
  });
});
