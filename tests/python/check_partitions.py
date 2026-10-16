"""Split semantics of ml_runtime_common.split_partitions, which every runner's prepare uses: the random path is the Phase 3
partitions() unchanged, an unlabelled dataset splits like zero targets, and the chronological split orders rows by UTC
instant with floor sizes. Prints one JSON line: {"results": {name: bool}}.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import UUID, classification_rows, regression_rows, run, snapshot  # noqa: E402
import ml_runtime_common as common  # noqa: E402  (support put ml/ on the path)

results = {}


def phase3_partitions(spec, y):
    """partitions() exactly as it was before Phase 4 (public 1b4118f), the reference the random path must keep reproducing."""
    split = spec["split"]
    indices = np.arange(len(y))
    train, holdout = train_test_split(indices, train_size=split["train"], random_state=spec["seed"],
                                      stratify=y if split["stratify"] else None)
    validation, test = train_test_split(holdout, train_size=split["validation"] / (split["validation"] + split["test"]),
                                        random_state=spec["seed"], stratify=y[holdout] if split["stratify"] else None)
    result = {"indices": {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}}
    cv = (spec.get("evaluation") or {}).get("cv")
    if cv:
        stratified = cv["strategy"] == "stratified_kfold"
        splitter = (StratifiedKFold if stratified else KFold)(n_splits=cv["folds"], shuffle=True, random_state=spec["seed"])
        result["folds"] = [train[val].tolist() for _, val in splitter.split(np.zeros(len(train)), y[train] if stratified else None)]
    return result


def raises(code_or_type, fn, *args):
    try:
        fn(*args)
    except common.RunnerError as error:
        return error.code == code_or_type
    except Exception as error:
        return isinstance(code_or_type, type) and isinstance(error, code_or_type)
    return False


# 1. Random path: byte-for-byte the Phase 3 split and folds, over seeds, fractions, stratification and both CV kinds.
class_rows, reg_rows = classification_rows(n=120), regression_rows(n=97)
class_y, reg_y = np.array([r["label"] for r in class_rows]), np.array([r["y"] for r in reg_rows])
CLASS = [("a", "number"), ("kind", "string"), ("label", "string")]
cases = [(snapshot("logistic_regression", CLASS, "label", ["a"], stratify=True, seed=7, evaluation={"cv": {"strategy": "stratified_kfold", "folds": 3}}), class_rows, class_y),
         (snapshot("logistic_regression", CLASS, "label", ["a"], stratify=False, seed=11), class_rows, class_y),
         (snapshot("ridge", [("a", "number"), ("y", "number")], "y", ["a"], seed=3, evaluation={"cv": {"strategy": "kfold", "folds": 4}}), reg_rows, reg_y),
         (snapshot("ridge", [("a", "number"), ("y", "number")], "y", ["a"], seed=42), reg_rows, reg_y)]
cases[3][0]["split"].update(train=0.7, validation=0.1, test=0.2)
results["random_path_matches_phase3_partitions"] = all(
    common.split_partitions(spec, rows, y) == phase3_partitions(spec, y) == common.partitions(spec, y) for spec, rows, y in cases)

# 2. Unlabelled random path: zeros stand in for the target, and nothing may stratify on a target that does not exist.
unlabelled = snapshot("ridge", [("a", "number"), ("b", "number")], None, ["a"], seed=5, evaluation={"cv": {"strategy": "kfold", "folds": 3}})
results["unlabelled_random_splits_like_zero_targets"] = common.split_partitions(unlabelled, reg_rows, None) == phase3_partitions(unlabelled, np.zeros(len(reg_rows)))
stratified = snapshot("ridge", [("a", "number"), ("b", "number")], None, ["a"], stratify=True)
stratified_cv = snapshot("ridge", [("a", "number"), ("b", "number")], None, ["a"], evaluation={"cv": {"strategy": "stratified_kfold", "folds": 3}})
results["unlabelled_random_refuses_stratification"] = raises(ValueError, common.split_partitions, stratified, reg_rows, None) and \
    raises(ValueError, common.split_partitions, stratified_cv, reg_rows, None)


# 3. Chronological path.
def chronological(train=0.6, validation=0.2, test=0.2, evaluation=None, seed=1):
    spec = snapshot("ridge", [("x", "number"), ("t", "timestamp"), ("y", "number")], "y", ["x"], seed=seed, evaluation=evaluation)
    spec["split"] = {"id": UUID, "strategy": "chronological", "timeColumn": "t", "train": train, "validation": validation, "test": test}
    return spec


def timed(stamps):
    return [{"x": float(i), "t": t, "y": float(i)} for i, t in enumerate(stamps)]


def sizes(spec, n):
    parts = common.split_partitions(spec, timed([f"2024-01-01T00:00:{i % 60:02d}Z" for i in range(n)]), None)["indices"]
    return len(parts["train"]), len(parts["validation"]), len(parts["test"])


# floor(n * fraction) on IEEE doubles, exactly as JavaScript computes it: 100 * 0.29 is 28.999999999999996, so 28.
results["chronological_sizes_floor_ieee_products"] = (sizes(chronological(), 10) == (6, 2, 2) and sizes(chronological(0.5, 0.25, 0.25), 7) == (3, 1, 3)
                                                      and sizes(chronological(0.29, 0.29, 0.42), 100) == (28, 28, 44) and math.floor(100 * 0.29) == 28)
rng = np.random.RandomState(0)
minutes = rng.permutation(40)
stamps = [f"2024-03-{1 + m // 24:02d}T{m % 24:02d}:00:00Z" for m in minutes]
parts = common.split_partitions(chronological(), timed(stamps), None)["indices"]
instant = lambda i: common.epoch_micros(stamps[i])  # noqa: E731
results["chronological_blocks_follow_time_order"] = (
    max(map(instant, parts["train"])) < min(map(instant, parts["validation"])) and max(map(instant, parts["validation"])) < min(map(instant, parts["test"]))
    and all(instant(a) < instant(b) for block in parts.values() for a, b in zip(block, block[1:])))
results["chronological_partitions_cover_every_row_once"] = sorted(parts["train"] + parts["validation"] + parts["test"]) == list(range(len(stamps)))
# Text order and UTC order differ here: 10:00+05:00 is 05:00Z, 06:00Z, 01:00-06:00 is 07:00Z.
offsets = ["2024-01-01T01:00:00-06:00", "2024-01-01T06:00:00Z", "2024-01-01T10:00:00+05:00", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]
results["chronological_orders_by_utc_instant_not_text"] = common.split_partitions(chronological(0.4, 0.2, 0.4), timed(offsets), None)["indices"] == \
    {"train": [2, 1], "validation": [0], "test": [3, 4]}
# One instant written three ways is a tie: stable sort keeps row order.
ties = ["2024-01-01T00:00:00Z", "2023-12-31T19:00:00-05:00", "2024-01-01T05:30:00+05:30", "2023-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
results["chronological_ties_keep_row_order"] = common.split_partitions(chronological(0.4, 0.2, 0.4), timed(ties), None)["indices"] == \
    {"train": [3, 0], "validation": [1], "test": [2, 4]}
results["chronological_split_ignores_seed"] = common.split_partitions(chronological(seed=1), timed(stamps), None) == \
    common.split_partitions(chronological(seed=2), timed(stamps), None)
results["chronological_refuses_empty_partition"] = raises("invalid_time_series", common.split_partitions, chronological(0.5, 0.25, 0.25),
                                                          timed(["2024-01-01T00:00:00Z"] * 3), None)
results["chronological_refuses_invalid_timestamps"] = all(
    raises("invalid_time_series", common.split_partitions, chronological(), timed(stamps[:9] + [bad]), None)
    for bad in ("2024-01-01T00:00:00", "2024-02-30T00:00:00Z", None, 1704067200))
results["chronological_refuses_cross_validation"] = raises(ValueError, common.split_partitions, chronological(evaluation={"cv": {"strategy": "kfold", "folds": 3}}),
                                                           timed(stamps), None)
unknown = chronological()
unknown["split"]["strategy"] = "shuffled"
results["unknown_strategy_refused"] = raises(ValueError, common.split_partitions, unknown, timed(stamps), None)

# 4. The sklearn runner's prepare goes through split_partitions (the neural runners share the same line).
rows = timed(stamps)
prepared = run("prepare", chronological(), rows)
random_spec = cases[2][0]
results["sklearn_prepare_uses_split_partitions"] = (prepared["indices"] == common.split_partitions(chronological(), rows, None)["indices"]
                                                    and {k: v for k, v in run("prepare", random_spec, reg_rows).items() if k != "environment"} == phase3_partitions(random_spec, reg_y))

print(json.dumps({"results": results}))
