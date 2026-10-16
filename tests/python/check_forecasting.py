"""Forecasting backend checks (pandas/NumPy/scikit-learn, the scikit-learn interpreter). Prints one JSON object of named results.

Known-answer series, leakage proofs, refusals, outputs and artifact safety. LightGBM's forecaster is checked in check_gbdt.py;
processor integration is in tests/integration/ml-forecasting.test.ts.
"""
import base64
import io
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import forecasting_runner as fb  # noqa: E402
import forecasting_common as fc  # noqa: E402
import ml_runtime_common as common  # noqa: E402

results, details = {}, {}
UUID = "00000000-0000-4000-8000-000000000000"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def stamp(i, step=timedelta(hours=1)):
    return (START + i * step).strftime("%Y-%m-%dT%H:%M:%SZ")


def rows_of(values, step=timedelta(hours=1)):
    return [{"at": stamp(i, step), "y": float(v)} for i, v in enumerate(values)]


def spec(algorithm, hyperparameters=None, metrics=None, split=(0.6, 0.2, 0.2)):
    return {"algorithm": algorithm, "backend": "forecasting", "seed": 3, "hyperparameters": hyperparameters or {},
            "evaluation": {"metrics": metrics or ["mae", "rmse", "smape", "mase"]},
            "split": {"id": UUID, "strategy": "chronological", "timeColumn": "at", "train": split[0], "validation": split[1], "test": split[2]},
            "dataset": {"schema": {"columns": [{"name": "at", "type": "timestamp"}, {"name": "y", "type": "number"}], "target": "y"}},
            "pipeline": {"definition": {"features": ["at"], "steps": []}}}


def run(command, s, rows, **extra):
    return fb.execute({"command": command, "snapshot": s, "rows": rows, **extra})


def lifecycle(s, rows):
    prepared = run("prepare", s, rows)
    trained = run("train", s, rows, prepared=prepared)
    return prepared, trained, run("evaluate", s, rows, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])


def metric(evaluated, name, partition):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)


def refusal(fn):
    try:
        fn()
    except common.RunnerError as error:
        return error.code
    except common.IncompatibleArtifact:
        return "incompatible_artifact"
    except ValueError:
        return "value_error"
    return None


# ---- known answers ------------------------------------------------------------------------------------------------------
trend = 2.0 * np.arange(120) + 5.0
ridge_trend = spec("ridge_forecaster", {"lags": 2, "alpha": 1e-9, "horizon": 5})
_, trained, evaluated = lifecycle(ridge_trend, rows_of(trend))
forecast = run("predict", ridge_trend, rows_of(trend)[-10:], artifact=trained["artifact"])["predictions"]
results["ridge_extrapolates_a_linear_trend"] = np.allclose([p["value"] for p in forecast], 2.0 * np.arange(10, 15) + trend[-10], atol=1e-4) and metric(evaluated, "mae", "test") < 1e-4
season = np.tile([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], 20)
naive = spec("seasonal_naive_forecaster", {"seasonal_period": 7, "horizon": 7}, ["mae", "rmse", "smape"])
_, naive_trained, naive_eval = lifecycle(naive, rows_of(season))
naive_forecast = run("predict", naive, rows_of(season)[-14:], artifact=naive_trained["artifact"])["predictions"]
results["seasonal_naive_repeats_a_pure_season"] = [p["value"] for p in naive_forecast] == list(season[-7:]) and metric(naive_eval, "mae", "test") == 0.0
# A perfectly seasonal training block has a zero seasonal-naive error, so MASE has no scale: refused, never infinite (D1).
results["mase_without_a_scale_is_refused"] = refusal(lambda: lifecycle(spec("seasonal_naive_forecaster", {"seasonal_period": 7}, ["mase"]), rows_of(season))) == "undefined_metric"
rng = np.random.RandomState(0)
ar = [0.0]
for _ in range(399):
    ar.append(0.8 * ar[-1] + rng.randn())
ar_spec = spec("ridge_forecaster", {"lags": 1, "alpha": 1e-6, "horizon": 1})
_, ar_trained, ar_eval = lifecycle(ar_spec, rows_of(ar))
coefficient = float(np.load(io.BytesIO(base64.b64decode(ar_trained["artifact"])), allow_pickle=False)["coef"][0])
results["ridge_recovers_an_ar1_coefficient"] = abs(coefficient - 0.8) < 0.1
details["ar1_coefficient"] = coefficient
results["ar1_one_step_error_near_the_noise"] = 0.6 < metric(ar_eval, "rmse", "test") < 1.4
windowed = spec("ridge_forecaster", {"lags": 3, "windows": [4, 12], "horizon": 3})
results["rolling_windows_train_and_forecast"] = len(run("predict", windowed, rows_of(ar)[-20:], artifact=lifecycle(windowed, rows_of(ar))[1]["artifact"])["predictions"]) == 3

# ---- leakage -------------------------------------------------------------------------------------------------------------
base_rows = rows_of(ar)
prepared, trained, evaluated = lifecycle(windowed, base_rows)
test_positions = set(prepared["indices"]["test"])
changed = [dict(r, y=r["y"] + 1000.0) if i in test_positions else r for i, r in enumerate(base_rows)]
_, changed_trained, changed_eval = lifecycle(windowed, changed)
results["test_block_values_never_change_the_fitted_model"] = changed_trained["artifact"] == trained["artifact"] or \
    np.array_equal(np.load(io.BytesIO(base64.b64decode(changed_trained["artifact"])), allow_pickle=False)["coef"], np.load(io.BytesIO(base64.b64decode(trained["artifact"])), allow_pickle=False)["coef"])
results["origins_never_see_later_actuals"] = all(metric(changed_eval, n, "validation") == metric(evaluated, n, "validation") for n in ("mae", "rmse", "smape"))
shuffled = list(base_rows)
random.Random(5).shuffle(shuffled)
_, shuffled_trained, shuffled_eval = lifecycle(windowed, shuffled)
results["shuffled_rows_give_identical_results"] = [m["value"] for m in shuffled_eval["metrics"]] == [m["value"] for m in evaluated["metrics"]]
results["same_series_same_artifact_arrays"] = np.array_equal(np.load(io.BytesIO(base64.b64decode(shuffled_trained["artifact"])), allow_pickle=False)["coef"],
                                                             np.load(io.BytesIO(base64.b64decode(trained["artifact"])), allow_pickle=False)["coef"])

# ---- refusals ------------------------------------------------------------------------------------------------------------
duplicate = rows_of(ar[:50]) + [{"at": stamp(10), "y": 1.0}]
irregular = [r for i, r in enumerate(rows_of(ar[:60])) if i != 30]
months = [{"at": (datetime(2020, 1 + i % 12, 1, tzinfo=timezone.utc).replace(year=2020 + i // 12)).strftime("%Y-%m-%dT%H:%M:%SZ"), "y": float(i)} for i in range(48)]
missing = [dict(r, y=None) if i == 20 else r for i, r in enumerate(rows_of(ar[:60]))]
results["duplicate_instants_refused"] = refusal(lambda: run("prepare", windowed, duplicate)) == "invalid_time_series"
results["irregular_spacing_refused"] = refusal(lambda: run("prepare", windowed, irregular)) == "invalid_time_series"
results["calendar_months_refused"] = refusal(lambda: run("prepare", windowed, months)) == "invalid_time_series"
results["missing_target_refused"] = refusal(lambda: run("prepare", windowed, missing)) == "invalid_time_series"
results["naive_timestamp_refused"] = refusal(lambda: run("prepare", windowed, [{"at": "2025-01-01T00:00:00", "y": 1.0}] + rows_of(ar[:30])[1:])) == "invalid_time_series"
short = spec("ridge_forecaster", {"lags": 30})
results["history_shorter_than_the_model_refused"] = refusal(lambda: lifecycle(short, rows_of(ar[:40]))) == "invalid_time_series"
results["prediction_history_must_continue_the_frequency"] = refusal(lambda: run("predict", windowed, rows_of(ar[:20], timedelta(minutes=30)), artifact=trained["artifact"])) == "invalid_time_series"
results["prediction_history_shorter_than_the_context_refused"] = refusal(lambda: run("predict", windowed, rows_of(ar[:5]), artifact=trained["artifact"])) == "invalid_time_series"
random_split = {**windowed, "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False}}
results["random_split_refused"] = refusal(lambda: run("prepare", random_split, base_rows)) == "value_error"
results["cuda_refused"] = refusal(lambda: run("prepare", {**windowed, "training": {"device": "cuda", "precision": "fp32"}}, base_rows)) == "cuda_unsupported"

# ---- outputs and artifacts -----------------------------------------------------------------------------------------------
history = base_rows[-30:]
out = run("predict", windowed, history, artifact=trained["artifact"])["predictions"]
last = datetime.strptime(history[-1]["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
results["outputs_are_horizon_steps_with_continuing_utc_timestamps"] = (
    [p["step"] for p in out] == [1, 2, 3] and all(set(p) == {"step", "timestamp", "value"} and np.isfinite(p["value"]) for p in out)
    and [datetime.strptime(p["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc) for p in out] == [last + timedelta(hours=k) for k in (1, 2, 3)])
results["forecasts_are_deterministic"] = run("predict", windowed, history, artifact=trained["artifact"])["predictions"] == out
buffer = io.BytesIO()
np.savez(buffer, metadata=np.array(json.dumps({"schema": 1})), evil=np.array([{"x": 1}], dtype=object))
results["object_array_artifact_refused"] = refusal(lambda: run("predict", windowed, history, artifact=base64.b64encode(buffer.getvalue()).decode())) == "incompatible_artifact"
results["foreign_runtime_refused"] = refusal(lambda: run("predict", windowed, history, artifact=trained["artifact"], expectedEnvironment={**trained["environment"], "pandas": "0.0.1"})) == "incompatible_artifact"
results["artifact_of_another_configuration_refused"] = refusal(lambda: run("predict", spec("ridge_forecaster", {"lags": 3, "windows": [4, 12], "horizon": 4}), history, artifact=trained["artifact"])) == "incompatible_artifact"

print(json.dumps({"results": {k: bool(v) for k, v in results.items()}, "details": details}))
