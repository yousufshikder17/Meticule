"""The Python metric registry and every Phase 4 metric against hand-computed values and the D1/D2 edge cases.
Prints one JSON line: {"results": {name: bool}, "registry": {...}}. `registry` is the mirror that the TypeScript registry in
src/ml/metrics.ts must equal: directions, task metrics (sorted), label-free and diagnostic metrics (sorted), and the
defaults per task for labelled and unlabelled data (null where the task always has a target).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import sklearn_runner  # noqa: E402
import ml_runtime_common as common  # noqa: E402

results = {}
close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)  # noqa: E731


def refuses(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except common.RunnerError as error:
        return error.code == "undefined_metric"
    return False


def value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


# ---- registry: the spec's table (sec. 7 without transcription), and Phase 3 tables kept exactly -----------------------
C, R, F, A, E = "classification", "regression", "forecasting", "anomaly_detection", "embedding"
SPEC = {"accuracy": ({C}, "higher", True, False), "precision": ({C}, "higher", True, False), "recall": ({C}, "higher", True, False),
        "f1": ({C}, "higher", True, False), "roc_auc": ({C, A}, "higher", True, False), "log_loss": ({C}, "lower", True, False),
        "mae": ({R, F}, "lower", True, False), "mse": ({R}, "lower", True, False), "rmse": ({R, F}, "lower", True, False),
        "r2": ({R}, "higher", True, False), "smape": ({F}, "lower", True, False), "mase": ({F}, "lower", True, False),
        "average_precision": ({A}, "higher", True, False), "anomaly_precision": ({A}, "higher", True, False),
        "anomaly_recall": ({A}, "higher", True, False), "anomaly_f1": ({A}, "higher", True, False),
        "threshold_calibration_error": ({A}, "lower", False, True), "retrieval_precision_at_1": ({E}, "higher", True, False),
        "retrieval_mrr": ({E}, "higher", True, False), "mean_pairwise_cosine": ({E}, "lower", False, True)}
results["registry_matches_phase4_spec"] = {n: (set(m[0]), m[1], m[2], m[3]) for n, m in common.METRICS.items()} == SPEC
PHASE3 = {"accuracy": "higher", "precision": "higher", "recall": "higher", "f1": "higher", "roc_auc": "higher",
          "log_loss": "lower", "mae": "lower", "mse": "lower", "rmse": "lower", "r2": "higher"}
results["phase3_tables_unchanged"] = (common.TASK_METRICS[C] == ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
                                      and common.TASK_METRICS[R] == ("mae", "mse", "rmse", "r2")
                                      and list(sklearn_runner.DIRECTIONS.items()) == list(PHASE3.items())
                                      and list(common.DIRECTIONS.items())[:10] == list(PHASE3.items()))
results["derived_tables_consistent"] = (set(common.DIRECTIONS) == set(SPEC) and set(common.TASK_METRICS) == {C, R, F, A, E}
                                        and all(n in common.TASK_METRICS[t] for n, m in SPEC.items() for t in m[0])
                                        and common.LABEL_FREE == common.DIAGNOSTIC == {"threshold_calibration_error", "mean_pairwise_cosine"})
DEFAULTS = {C: (["accuracy"], None), R: (["mae", "mse"], None), F: (["mae", "rmse"], None),
            A: (["average_precision"], ["threshold_calibration_error"]), E: (["retrieval_precision_at_1"], ["mean_pairwise_cosine"])}
results["default_metrics_per_task"] = all(
    common.default_metrics(t, True) == labelled and (value_error(common.default_metrics, t, False) if unlabelled is None else common.default_metrics(t, False) == unlabelled)
    for t, (labelled, unlabelled) in DEFAULTS.items())
spec = lambda *names: {"evaluation": {"metrics": list(names)}}  # noqa: E731
results["requested_metrics_rules"] = (common.requested_metrics({}, C) == ["accuracy"] and common.requested_metrics(spec("rmse", "smape"), F) == ["rmse", "smape"]
                                      and common.requested_metrics({}, A, labelled=False) == ["threshold_calibration_error"]
                                      and common.requested_metrics(spec("mean_pairwise_cosine"), E, False) == ["mean_pairwise_cosine"]
                                      and value_error(common.requested_metrics, spec("mse"), F) and value_error(common.requested_metrics, spec("smape"), R)
                                      and value_error(common.requested_metrics, spec("retrieval_mrr"), E, False)
                                      and value_error(common.requested_metrics, spec("roc_auc", "threshold_calibration_error"), A, False))

# measure_predictions keeps the Phase 3 values (hand-computed): weighted precision 13/15, log loss of the true-class probabilities.
labels, guesses = np.array(["a", "b", "b", "a", "b"]), np.array(["a", "b", "a", "a", "b"])
proba = np.array([[.8, .2], [.1, .9], [.4, .6], [.7, .3], [.3, .7]])
measured = common.measure_predictions(["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"], labels, guesses, proba=proba, scores=proba, classes=np.array(["a", "b"]))
regression = common.measure_predictions(["mae", "mse", "rmse", "r2"], np.array([1.0, 2, 3, 4]), np.array([1.5, 2, 2.5, 5]))
results["existing_metric_values_unchanged"] = (
    close(measured["accuracy"], 0.8) and close(measured["precision"], 13 / 15) and close(measured["recall"], 0.8) and close(measured["f1"], 0.8)
    and close(measured["roc_auc"], 1.0) and close(measured["log_loss"], -(math.log(.8) + math.log(.9) + math.log(.6) + 2 * math.log(.7)) / 5)
    and close(regression["mae"], 0.5) and close(regression["mse"], 0.375) and close(regression["rmse"], math.sqrt(0.375)) and close(regression["r2"], 0.7)
    and all(type(v) is float for v in [*measured.values(), *regression.values()]))

# ---- forecasting ---------------------------------------------------------------------------------------------------
# terms: 20/210, 0/0 -> 0, 20/90, 40/20; mean * 100 = 3650/63.
results["smape_hand_computed"] = close(common.smape([100, 0, 50, -10], [110, 0, 40, 10]), 3650 / 63)
results["smape_zero_terms_and_range"] = common.smape([0, 0], [0, 0]) == 0.0 and common.smape([1, -2], [-1, 2]) == 200.0
# history 1,3,2,5 -> naive errors 2,1,3 (scale 2); forecast errors 1,2 (mean 1.5). Period 2: 0,0,1,4 -> 1,4 (scale 2.5).
results["mase_hand_computed"] = (close(common.mase([4, 6], [5, 4], [1, 3, 2, 5]), 0.75) and close(common.mase([4, 6], [5, 4], [1, 3, 2, 5], None), 0.75)
                                 and close(common.mase([4, 6], [5, 4], [0, 0, 1, 4], 2), 0.6))
results["mase_refuses_zero_or_missing_scale"] = (refuses(common.mase, [1], [2], [3, 3, 3]) and refuses(common.mase, [1], [2], [5])
                                                 and refuses(common.mase, [1], [2], [1, 2], 2) and refuses(common.mase, [1], [2], [1, 5, 1, 5], 2))

# ---- anomaly detection ---------------------------------------------------------------------------------------------
truth, scores = [True, False, True, False, False], [0.9, 0.8, 0.7, 0.2, 0.1]
# Ranked T F T F F: precision 1 at recall .5, 2/3 at recall 1 -> AP 5/6. Pairs (anomaly above normal): 3 + 2 of 6.
results["average_precision_hand_computed"] = close(common.average_precision(truth, scores), 5 / 6)
results["anomaly_roc_auc_hand_computed"] = close(common.anomaly_roc_auc(truth, scores), 5 / 6)
flagged = common.anomaly_scores(truth, [True, True, True, True, False])  # tp 2, fp 2, fn 0
results["anomaly_scores_hand_computed"] = flagged == {"anomaly_precision": 0.5, "anomaly_recall": 1.0, "anomaly_f1": 2 * 2 / (2 * 2 + 2 + 0)}
results["anomaly_zero_division_is_zero"] = common.anomaly_scores(truth, [False] * 5) == {"anomaly_precision": 0.0, "anomaly_recall": 0.0, "anomaly_f1": 0.0}


def single_class_refused(fn, *args):
    try:
        fn(*args)
    except common.RunnerError as error:
        return error.code == "undefined_metric" and "stratify" in error.message
    return False


results["binary_metrics_refuse_single_class"] = all(
    single_class_refused(common.anomaly_roc_auc, y, scores) and single_class_refused(common.average_precision, y, scores)
    and single_class_refused(common.anomaly_scores, y, [True, False, True, False, True]) for y in ([False] * 5, [True] * 5, []))
results["threshold_calibration_error_hand_computed"] = close(common.threshold_calibration_error([True, False, False, False], 0.1), 0.15) and \
    close(common.threshold_calibration_error(np.array([0, 0, 0, 0]), 0.05), 0.05)

# ---- embedding -----------------------------------------------------------------------------------------------------
# Unit vectors at 0 (a), 10 (b), 20 (a), 90 (b), 45 (c), 95 (b) degrees. Ranks of the first same-label row: 2, 4, 2, 1, -, 1
# (c has no peer and is excluded) -> P@1 2/5, MRR (1/2 + 1/4 + 1/2 + 1 + 1) / 5.
angles, groups = [0, 10, 20, 90, 45, 95], ["a", "b", "a", "b", "c", "b"]
unit = lambda degrees: [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]  # noqa: E731
vectors = [unit(d) for d in angles]
expected = {"retrieval_precision_at_1": 0.4, "retrieval_mrr": 0.65}
same = lambda got: all(close(got[k], v) for k, v in expected.items())  # noqa: E731
results["retrieval_hand_computed"] = same(common.retrieval_scores(vectors, groups))
results["retrieval_is_scale_invariant"] = same(common.retrieval_scores([[3 * x for x in v] if i % 2 else v for i, v in enumerate(vectors)], groups))
results["retrieval_excludes_rows_without_a_peer"] = same(common.retrieval_scores(vectors + [unit(180)], groups + ["d"]))
# Identical vectors: a non-relevant row tied with the best relevant one ranks first. Ranks 2, 3, 2, 3.
tied = common.retrieval_scores([[1, 0], [1, 0], [1, 0], [0, 1]], ["a", "b", "a", "b"])
results["retrieval_ties_rank_non_relevant_first"] = tied["retrieval_precision_at_1"] == 0.0 and close(tied["retrieval_mrr"], 5 / 12) and \
    common.retrieval_scores([[2, 2]] * 6, list("aabbcc"))["retrieval_precision_at_1"] == 0.0
rng = np.random.RandomState(4)
random_vectors, random_labels = rng.randn(53, 6), rng.randint(0, 5, 53).tolist()
reference = common.retrieval_scores(random_vectors, random_labels, block=1000)
results["retrieval_blockwise_equals_single_block"] = all(common.retrieval_scores(random_vectors, random_labels, block=b) == reference for b in (1, 7, 52))
results["retrieval_refuses_undefined"] = (refuses(common.retrieval_scores, [[1, 0]], ["a"]) and refuses(common.retrieval_scores, [[1, 0], [0, 1]], ["a", "b"])
                                          and refuses(common.retrieval_scores, [[1, 0], [0, 0], [0, 1]], ["a", "a", "b"])
                                          and refuses(common.retrieval_scores, [[1, 0], [np.nan, 1]], ["a", "a"]))
results["mean_pairwise_cosine_hand_computed"] = close(common.mean_pairwise_cosine([[1, 0], [0, 1], [1, 1]]), math.sqrt(2) / 3) and \
    close(common.mean_pairwise_cosine([[1, 0], [-2, 0]]), -1.0)
pairs = rng.randn(31, 8)
units = pairs / np.linalg.norm(pairs, axis=1, keepdims=True)
naive = np.mean([units[i] @ units[j] for i in range(31) for j in range(31) if i != j])
results["mean_pairwise_cosine_matches_all_pairs"] = close(common.mean_pairwise_cosine(pairs), float(naive))
results["mean_pairwise_cosine_refuses_undefined"] = refuses(common.mean_pairwise_cosine, [[1, 2]]) and refuses(common.mean_pairwise_cosine, [[1, 2], [0, 0]]) \
    and refuses(common.mean_pairwise_cosine, [])

registry = {"directions": common.DIRECTIONS,
            "taskMetrics": {task: sorted(names) for task, names in common.TASK_METRICS.items()},
            "labelFree": sorted(common.LABEL_FREE), "diagnostic": sorted(common.DIAGNOSTIC),
            "defaults": {task: {"labelled": common.default_metrics(task, True),
                                "unlabelled": None if common.DEFAULTS[task][1] is None else common.default_metrics(task, False)}
                         for task in common.TASKS}}
print(json.dumps({"results": results, "registry": registry}))
