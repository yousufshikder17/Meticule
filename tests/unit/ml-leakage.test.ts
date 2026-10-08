import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { cohortKey, validateSnapshot } from "../../src/ml/domain.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { snapshotFor } from "./ml-fixtures.js";

const python = process.env.ML_TEST_PYTHON;
const columns = [{ name: "x", type: "number" as const }, { name: "y", type: "number" as const }];

describe("data leakage guarantees", () => {
  it.runIf(Boolean(python))("fits preprocessing on training rows only, in the final fit and inside every CV fold", () => {
    const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_leakage.py")], { encoding: "utf8", env: { ...process.env, OMP_NUM_THREADS: "1" } }).trim().split("\n").pop()!) as Record<string, boolean>;
    const expected = ["final_fit_uses_training_rows_only", "no_held_out_row_in_final_fit", "scaler_mean_is_training_mean", "imputer_median_is_training_median", "held_out_values_do_not_change_fitted_state",
      "folds_partition_training_rows", "cv_fits_once_per_fold", "cv_fold_fit_rows_exclude_own_validation_fold", "cv_fit_never_sees_validation_or_test_partition", "cv_fold_statistics_are_independent",
      "cv_scores_ignore_held_out_rows", "holdout_scores_do_see_held_out_rows", "design_matrix_excludes_target", "inference_imputes_with_training_median", "inference_is_independent_of_batch_composition",
      "same_seed_same_split_and_folds", "different_seed_different_split"];
    expect(Object.keys(out).sort()).toEqual([...expected].sort());
    for (const name of expected) expect(out[name], name).toBe(true);
  }, 120_000);
  it("never lets the target or an unknown column act as a feature", () => {
    const backend = new SklearnBackend("unused"); const cv = { cv: { strategy: "kfold", folds: 3 } };
    expect(() => validateSnapshot(snapshotFor({ columns, features: ["x"] }))).not.toThrow();
    expect(() => backend.validate(snapshotFor({ columns, features: ["x", "y"], evaluation: cv }))).toThrow(/exclude the target/);
    expect(() => backend.validate(snapshotFor({ columns, features: ["missing"], evaluation: cv }))).toThrow(/Features must exist/);
    expect(() => backend.validate(snapshotFor({ columns, features: ["y"] }))).toThrow(/exclude the target/);
  });
});

describe("reproducibility keys", () => {
  const key = (opts: Parameters<typeof snapshotFor>[0], indices: unknown = { train: [1] }) => cohortKey(snapshotFor(opts), indices);
  it("treats runs as one comparable cohort only when data, features, split, seed, folds and indices match", () => {
    const base = key({ columns, features: ["x"], evaluation: { cv: { strategy: "kfold", folds: 3 } } });
    // Same everything except estimator and hyperparameters: comparable (the point of a benchmark).
    expect(key({ columns, features: ["x"], algorithm: "lasso", hyperparameters: { alpha: 2 }, evaluation: { cv: { strategy: "kfold", folds: 3 } } })).toBe(base);
    for (const changed of [
      key({ columns, features: ["x"], evaluation: { cv: { strategy: "kfold", folds: 4 } } }), key({ columns, features: ["x"], evaluation: {} }),
      key({ columns, features: ["x"], steps: [{ operation: "standard_scale" }], evaluation: { cv: { strategy: "kfold", folds: 3 } } }),
      key({ columns, features: ["x"], rowCount: 100, evaluation: { cv: { strategy: "kfold", folds: 3 } } }, { train: [2] }),
    ]) expect(changed).not.toBe(base);
  });
});
