"""Univariate time-series forecasting shared by the `forecasting` backend and LightGBM's `lightgbm_forecaster` (pandas + NumPy).

Series. The time column is ordered by its exact UTC instant (ml_runtime_common.epoch_micros, the C4 rules). Duplicate instants,
irregular spacing (every step must be the same number of microseconds, so calendar months are refused), and missing or
non-finite targets are refused as invalid_time_series, as is a history shorter than the model needs.

Features. Target lags 1..L and rolling means over declared windows, built with pandas shift/rolling so that every feature of a
row uses strictly earlier targets only; rows without full history are dropped from fitting. Models are fitted on the training
block only and forecast recursively: each step's prediction is appended to the history, never a future actual.

Evaluation. The training block scores the one-step in-sample fit. Each later block uses rolling origins (at most MAX_ORIGINS,
evenly spaced, deterministic): an origin sees only actuals strictly before it, forecasts `horizon` steps, and only targets inside
the block are scored. Metrics: mae, rmse, smape, mase (scaled by the seasonal-naive in-sample MAE of the training block).
"""
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas

import ml_runtime_common as common

RunnerError = common.RunnerError
MAX_ORIGINS = 200
MAX_WINDOWS = 8


def series(rows, time_column, target):
    """(order, epoch microseconds, values, step) of rows sorted by time. `order` maps series position -> row index."""
    if len(rows) < 2:
        raise RunnerError("invalid_time_series", "A time series needs at least two rows")
    micros = [common.epoch_micros(row.get(time_column)) for row in rows]
    order = sorted(range(len(rows)), key=micros.__getitem__)
    times = np.array([micros[i] for i in order], dtype=np.int64)
    deltas = np.diff(times)
    if (deltas == 0).any():
        raise RunnerError("invalid_time_series", "The time column has duplicate instants")
    if (deltas != deltas[0]).any():
        raise RunnerError("invalid_time_series", "The time series is not regularly spaced")
    values = [rows[i].get(target) for i in order]
    if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise RunnerError("invalid_time_series", "The time series has a missing or non-finite target")
    return order, times, np.array(values, dtype=float), int(deltas[0])


def frequency_name(times):
    """pandas' name for the spacing when it has one (provenance only)."""
    try:
        return pandas.infer_freq(pandas.to_datetime(times[:50], unit="us", utc=True)) or "none"
    except (TypeError, ValueError):
        return "none"


def context_length(settings):
    """Past values a model needs before it can forecast one step."""
    if settings["kind"] == "seasonal_naive":
        return settings["seasonal_period"]
    return max([settings["lags"], *settings["windows"]])


def design(values, settings):
    """Feature matrix (lags, then rolling means of earlier values) and the rows with full history."""
    frame = pandas.DataFrame({"y": values})
    columns = [frame["y"].shift(k) for k in range(1, settings["lags"] + 1)]
    columns += [frame["y"].shift(1).rolling(w).mean() for w in settings["windows"]]
    matrix = pandas.concat(columns, axis=1).to_numpy(dtype=float)
    complete = ~np.isnan(matrix).any(axis=1)
    return matrix, complete


def features_from_histories(histories, settings):
    """The next step's features for each history (all at least the context length long)."""
    lags = np.array([[h[-k] for k in range(1, settings["lags"] + 1)] for h in histories], dtype=float)
    windows = np.array([[np.mean(h[-w:]) for w in settings["windows"]] for h in histories], dtype=float).reshape(len(histories), len(settings["windows"]))
    return np.hstack([lags, windows])


def forecast(model, histories, horizon, settings):
    """Recursive multi-step forecasts for many origins at once: (origins, horizon)."""
    histories = [list(h) for h in histories]
    out = np.zeros((len(histories), horizon))
    for step in range(horizon):
        if settings["kind"] == "seasonal_naive":
            period = settings["seasonal_period"]
            # The value one season back from the step being forecast: a pure repetition of the last observed season.
            values = np.array([h[-period] for h in histories], dtype=float)
        else:
            values = np.asarray(model["predict"](features_from_histories(histories, settings)), dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Forecast is not finite")
        out[:, step] = values
        for h, v in zip(histories, values):
            h.append(float(v))
    return out


def fit(settings, values, fit_model):
    """Fits on the training block's values only. Seasonal naive fits nothing."""
    need = context_length(settings)
    if len(values) < need + 2:
        raise RunnerError("invalid_time_series", "The training block is shorter than the model's history")
    if settings["kind"] == "seasonal_naive":
        return {}
    matrix, complete = design(values, settings)
    return fit_model(matrix[complete], values[complete])


def measure(names, actual, predicted, train_values, period):
    values = {}
    for name in names:
        if name == "mae":
            values[name] = float(np.mean(np.abs(actual - predicted)))
        elif name == "rmse":
            values[name] = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        elif name == "smape":
            values[name] = common.smape(actual, predicted)
        elif name == "mase":
            values[name] = common.mase(actual, predicted, train_values, period)
        else:
            raise ValueError("Metric does not apply to forecasting")
    return values


def evaluate(spec, rows, prepared, settings, model, names):
    """Per-partition metrics by the rules in the module docstring."""
    target, time_column = spec["dataset"]["schema"]["target"], spec["split"]["timeColumn"]
    order, _, values, _ = series(rows, time_column, target)
    position = {row: k for k, row in enumerate(order)}
    blocks = {name: sorted(position[i] for i in prepared[name]) for name in ("train", "validation", "test")}
    train_end = blocks["train"][-1] + 1
    if blocks["train"] != list(range(train_end)):
        raise RunnerError("invalid_time_series", "The training block is not the earliest part of the series")
    train_values, need, horizon, period = values[:train_end], context_length(settings), settings["horizon"], settings["seasonal_period"]
    metrics = []
    # Training block: one-step forecasts from actual history, for rows that have the full context.
    starts = list(range(need, train_end))
    if not starts:
        raise RunnerError("invalid_time_series", "The training block is shorter than the model's history")
    in_sample = forecast(model, [values[:t] for t in starts], 1, settings)[:, 0]
    metrics.append(("train", measure(names, values[starts], in_sample, train_values, period)))
    for name in ("validation", "test"):
        block = blocks[name]
        first, last = block[0], block[-1]
        candidates = np.unique(np.linspace(first, last, num=min(MAX_ORIGINS, last - first + 1)).round().astype(int))
        origins = [int(o) for o in candidates if o >= need]
        if not origins:
            raise RunnerError("invalid_time_series", "The series is too short to evaluate this block")
        predicted = forecast(model, [values[:o] for o in origins], horizon, settings)
        actual, guess = [], []
        for row, origin in enumerate(origins):
            steps = min(horizon, last + 1 - origin)
            actual.extend(values[origin:origin + steps])
            guess.extend(predicted[row, :steps])
        metrics.append((name, measure(names, np.array(actual), np.array(guess), train_values, period)))
    return metrics


def predict(rows, time_column, target, settings, model, step_micros):
    """`horizon` forecasts after the most recent history row, continuing the training frequency."""
    _, times, values, step = series(rows, time_column, target)
    if step != step_micros:
        raise RunnerError("invalid_time_series", "Prediction history does not continue the training frequency")
    if len(values) < context_length(settings):
        raise RunnerError("invalid_time_series", "Prediction history is shorter than the model's context")
    forecasts = forecast(model, [values], settings["horizon"], settings)[0]
    out = []
    for k, value in enumerate(forecasts, start=1):
        instant = int(times[-1]) + k * step_micros
        stamp = datetime.fromtimestamp(instant // 1_000_000, tz=timezone.utc).replace(microsecond=instant % 1_000_000)
        out.append({"step": k, "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "value": float(value)})
    return out


def resolve_settings(supplied, kind, extra_names=()):
    """Bounded forecasting settings common to every forecaster; `extra_names` are the model's own parameters, resolved
    by its backend."""
    allowed = {"horizon", "seasonal_period"} | ({"lags", "windows"} if kind != "seasonal_naive" else set()) | set(extra_names)
    if set(supplied) - allowed:
        raise ValueError("Unsupported forecasting hyperparameter")

    def integer(name, low, high, default):
        value = supplied.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"Hyperparameter {name} out of range")
        return value

    settings = {"kind": kind, "horizon": integer("horizon", 1, 100, 1), "seasonal_period": integer("seasonal_period", 1, 366, 1)}
    if kind != "seasonal_naive":
        settings["lags"] = integer("lags", 1, 64, 7)
        windows = supplied.get("windows", [])
        if (not isinstance(windows, list) or len(windows) > MAX_WINDOWS or len(set(windows)) != len(windows)
                or any(not isinstance(w, int) or isinstance(w, bool) or not 2 <= w <= 366 for w in windows)):
            raise ValueError("Hyperparameter windows out of range")
        settings["windows"] = sorted(windows)
    return settings


def resolve_device(spec):
    """CPU only. A CUDA request is refused, never run on CPU instead."""
    training = spec.get("training") or {}
    if training.get("device", "cpu") == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported for forecasting")
    if training.get("device", "cpu") != "cpu" or training.get("precision", "fp32") != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")


def make_execute(adapter):
    """The four commands for a forecasting runner. `adapter` supplies environment(), contract, settings(spec),
    fit(settings, matrix, values, seed) -> model with a "predict" callable, serialize(metadata, model) and load(artifact, env)."""
    def execute(request):
        env = adapter.environment()
        spec, rows, command = request["snapshot"], request["rows"], request["command"]
        if request.get("expectedEnvironment") is not None:
            common.check_compatible(request["expectedEnvironment"], env, adapter.contract)
        split = spec["split"]
        if split["strategy"] != "chronological" or (spec.get("evaluation") or {}).get("cv"):
            raise ValueError("Forecasting needs a chronological split and no cross-validation")
        time_column, target = split["timeColumn"], spec["dataset"]["schema"].get("target")
        if spec["pipeline"]["definition"]["features"] != [time_column] or spec["pipeline"]["definition"]["steps"] or target is None:
            raise ValueError("Forecasting reads only the time column as its feature and needs a numeric target")
        resolve_device(spec)
        settings = adapter.settings(spec)
        if command == "prepare":
            series(rows, time_column, target)  # refuse an irregular or incomplete series before any work
            return {**common.split_partitions(spec, rows, None), "environment": env}
        if command == "train":
            order, times, values, step = series(rows, time_column, target)
            position = {row: k for k, row in enumerate(order)}
            train = sorted(position[i] for i in request["prepared"]["indices"]["train"])
            if train != list(range(len(train))):
                raise RunnerError("invalid_time_series", "The training block is not the earliest part of the series")
            started = time.perf_counter()
            model = fit(settings, values[:len(train)], lambda matrix, y: adapter.fit(settings, matrix, y, spec["seed"]))
            metadata = {"schema": 1, "task": "forecasting", "algorithm": spec["algorithm"], "settings": settings, "stepMicros": step,
                        "frequency": frequency_name(times), "environment": env}
            return {"artifact": adapter.serialize(metadata, model), "format": adapter.contract["artifact_format"], "environment": env,
                    "resolvedHyperparameters": {k: v for k, v in settings.items() if k != "kind"} | adapter.model_parameters(spec),
                    "performance": {"fitSeconds": time.perf_counter() - started}}
        metadata, model = adapter.load(request["artifact"], env)
        if metadata["algorithm"] != spec["algorithm"] or metadata["settings"] != settings:
            raise common.IncompatibleArtifact("artifact belongs to another forecasting configuration")
        if command == "predict":
            return {"predictions": predict(rows, time_column, target, settings, model, metadata["stepMicros"])}
        if command == "evaluate":
            names = common.requested_metrics(spec, "forecasting")
            metrics = []
            for partition, values in evaluate(spec, rows, request["prepared"]["indices"], settings, model, names):
                metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n]} for n, v in values.items())
            return {"metrics": metrics, "performance": {}}
        raise ValueError("Unknown backend command")
    return execute
