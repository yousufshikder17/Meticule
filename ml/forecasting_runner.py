"""Forecasting backend over the Meticule runner protocol (pandas + NumPy + scikit-learn; the scikit-learn interpreter).

`seasonal_naive_forecaster` repeats the last observed season and fits nothing. `ridge_forecaster` fits scikit-learn's Ridge on
lag and rolling-mean features of the training block and is exported as its coefficients (no pickle). Series rules, leakage
policy, evaluation and prediction are in forecasting_common. Artifacts are NumPy archives (`forecast-npz-v1`) read with
allow_pickle=False.
"""
import base64
import io
import json
import sys

import joblib
import numpy as np
import pandas
import scipy
import sklearn
import threadpoolctl
from sklearn.linear_model import Ridge

import forecasting_common
import ml_runtime_common as common

BACKEND = "forecasting"
ARTIFACT_FORMAT = "forecast-npz-v1"
PROTOCOL = "meticule-forecasting-v1"
RUNTIME_KEYS = ("pandas", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": {PROTOCOL: 1}, "schemas": {1}, "runtime_keys": RUNTIME_KEYS}
MODELS = {"seasonal_naive_forecaster": "forecasting", "ridge_forecaster": "forecasting"}
KINDS = {"seasonal_naive_forecaster": "seasonal_naive", "ridge_forecaster": "features"}
MAX_METADATA_BYTES = 1024 * 1024


class Adapter:
    contract = CONTRACT

    @staticmethod
    def environment():
        return common.base_environment([__file__, common.__file__, forecasting_common.__file__], BACKEND, ARTIFACT_FORMAT, 1, PROTOCOL,
                                       {"pandas": pandas.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                        "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})

    @staticmethod
    def model_parameters(spec):
        return {}  # every parameter, alpha included, is part of the settings

    @staticmethod
    def settings(spec):
        kind = KINDS[spec["algorithm"]]
        supplied = spec.get("hyperparameters") or {}
        settings = forecasting_common.resolve_settings(supplied, kind, ("alpha",) if kind == "features" else ())
        if kind == "features":
            alpha = supplied.get("alpha", 1.0)
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1e6:
                raise ValueError("Hyperparameter alpha out of range")
            settings["alpha"] = float(alpha)
        return settings

    @staticmethod
    def fit(settings, matrix, values, seed):
        model = Ridge(alpha=settings["alpha"]).fit(matrix, values)
        return Adapter.linear(np.asarray(model.coef_, dtype=float), float(model.intercept_))

    @staticmethod
    def linear(coef, intercept):
        return {"coef": coef, "intercept": intercept, "predict": lambda x: x @ coef + intercept}

    @staticmethod
    def serialize(metadata, model):
        arrays = {} if not model else {"coef": np.asarray(model["coef"], dtype=np.float64), "intercept": np.array([model["intercept"]], dtype=np.float64)}
        buffer = io.BytesIO()
        np.savez(buffer, metadata=np.array(json.dumps({**metadata, "backend": BACKEND}, allow_nan=False)), **arrays)
        return base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def load(artifact, env):
        try:
            archive = np.load(io.BytesIO(base64.b64decode(artifact, validate=True)), allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile) or "metadata" not in archive.files or archive.zip.getinfo("metadata.npy").file_size > MAX_METADATA_BYTES:
                raise ValueError
            raw = archive["metadata"]
            if raw.dtype.kind != "U" or raw.shape != ():
                raise ValueError
            metadata = json.loads(str(raw))
            if not isinstance(metadata, dict):
                raise ValueError
        except Exception:
            raise common.IncompatibleArtifact("unreadable artifact") from None
        common.check_compatible(metadata.get("environment"), env, CONTRACT)
        if metadata.get("backend") != BACKEND or metadata.get("schema") != 1 or metadata.get("algorithm") not in MODELS or not isinstance(metadata.get("settings"), dict):
            raise common.IncompatibleArtifact("malformed artifact")
        settings = metadata["settings"]
        if settings.get("kind") == "seasonal_naive":
            if set(archive.files) != {"metadata"}:
                raise common.IncompatibleArtifact("malformed artifact")
            return metadata, {}
        width = settings["lags"] + len(settings["windows"])
        if set(archive.files) != {"metadata", "coef", "intercept"}:
            raise common.IncompatibleArtifact("malformed artifact")
        coef, intercept = archive["coef"], archive["intercept"]
        if coef.dtype != np.float64 or coef.shape != (width,) or intercept.shape != (1,) or not (np.isfinite(coef).all() and np.isfinite(intercept).all()):
            raise common.IncompatibleArtifact("malformed artifact")
        return metadata, Adapter.linear(coef, float(intercept[0]))


execute = forecasting_common.make_execute(Adapter)


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
