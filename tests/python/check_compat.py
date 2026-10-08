"""Artifact compatibility contract against the real adapter, including a real Phase 1 artifact.

Prints one JSON object of named boolean results (plus details). Provenance (adapter checksum, platform, python
patch) must never decide compatibility; backend, format, protocol, artifact schema and runtime versions must."""
import base64
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import emit, full_run, ml_backend, run, snapshot  # noqa: E402

LEGACY = Path(__file__).resolve().parent / "legacy" / "sklearn_runner_phase1.py"
ADAPTER = Path(ml_backend.__file__)
rows = [{"x": float(i), "y": 2.0 * i + 1.0} for i in range(60)]
spec = snapshot("ridge", [("x", "number"), ("y", "number")], "y", ["x"], [("impute_median", {}), ("standard_scale", {})], hyperparameters={"alpha": 0.01}, seed=42)
results, details = {}, {}
current = ml_backend.environment()


def predicts(artifact, recorded):
    return run("predict", spec, [{"x": 10.0}], artifact=artifact, expectedEnvironment=recorded)["predictions"][0]


def rejected(artifact, recorded):
    try:
        predicts(artifact, recorded)
    except ml_backend.IncompatibleArtifact:
        return True
    return False


def repickle(artifact, **changes):
    """Same fitted pipeline, altered embedded environment (as another build of the adapter would have written)."""
    payload = pickle.loads(base64.b64decode(artifact))
    payload["environment"] = {**payload["environment"], **changes}
    return base64.b64encode(pickle.dumps(payload, protocol=5)).decode()


def without(env, *keys):
    return {k: v for k, v in env.items() if k not in keys}


# New (Phase 2) artifact: provenance is recorded exactly, and the artifact works.
prepared, model, evaluation = full_run(spec, rows)
env = model["environment"]
baseline = predicts(model["artifact"], env)
results["new_artifact_records_provenance"] = env["adapter_sha256"] == hashlib.sha256(ADAPTER.read_bytes()).hexdigest() and env["artifact_schema"] == "1" and env["protocol"] == ml_backend.PROTOCOL and env["backend"] == "sklearn" and env["artifact_format"] == "sklearn-pickle-v1"
results["new_artifact_embeds_same_environment"] = pickle.loads(base64.b64decode(model["artifact"]))["environment"] == env

# Provenance-only differences (checksum, platform, python patch) are tolerated, at request level and inside the artifact.
python_minor = ".".join(current["python"].split(".")[:2])
drift = {"adapter_sha256": "0" * 64, "platform": "Another-OS-1.0", "python": python_minor + ".0"}
results["adapter_checksum_drift_tolerated_request"] = abs(predicts(model["artifact"], {**env, **drift}) - baseline) < 1e-12
results["adapter_checksum_drift_tolerated_embedded"] = abs(predicts(repickle(model["artifact"], **drift), env) - baseline) < 1e-12
results["checksum_drift_keeps_provenance_untouched"] = env["adapter_sha256"] != drift["adapter_sha256"] and pickle.loads(base64.b64decode(repickle(model["artifact"], **drift)))["environment"]["adapter_sha256"] == "0" * 64

# Contract violations are refused at both layers.
violations = {
    "different_backend": {"backend": "xgboost"},
    "unsupported_artifact_format": {"artifact_format": "joblib-v9"},
    "unsupported_protocol": {"protocol": "meticule-sklearn-v9"},
    "unsupported_artifact_schema": {"artifact_schema": "2"},
    "unreadable_artifact_schema": {"artifact_schema": "two"},
    "sklearn_version_differs": {"sklearn": "0.0.1"},
    "numpy_version_differs": {"numpy": "0.0.1"},
    "scipy_version_differs": {"scipy": "0.0.1"},
    "joblib_version_differs": {"joblib": "0.0.1"},
    "python_minor_differs": {"python": "2.7.18"},
}
for name, change in violations.items():
    results[f"rejects_{name}_request"] = rejected(model["artifact"], {**env, **change})
    results[f"rejects_{name}_embedded"] = rejected(repickle(model["artifact"], **change), env)
results["rejects_missing_runtime_versions"] = rejected(model["artifact"], without(env, "sklearn"))
results["rejects_non_object_environment"] = rejected(model["artifact"], "sklearn")

# The recorded environment is checked before the artifact is unpickled: garbage bytes never reach pickle.
try:
    run("predict", spec, [{"x": 1.0}], artifact=base64.b64encode(b"not a pickle").decode(), expectedEnvironment={**env, "sklearn": "0.0.1"})
    results["incompatible_environment_checked_before_unpickling"] = False
except ml_backend.IncompatibleArtifact:
    results["incompatible_environment_checked_before_unpickling"] = True

# A genuine Phase 1 artifact, produced by the unmodified Phase 1 adapter in its own process.
def legacy(command, **extra):
    done = subprocess.run([sys.executable, str(LEGACY)], input=json.dumps({"command": command, "snapshot": spec, "rows": rows, **extra}), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


old_prepared = legacy("prepare")
old_model = legacy("train", prepared=old_prepared, expectedEnvironment=old_prepared["environment"])
old_env = old_model["environment"]
results["phase1_artifact_is_really_phase1"] = old_env["protocol"] == "meticule-sklearn-v1" and "artifact_schema" not in old_env and old_env["adapter_sha256"] != current["adapter_sha256"]
old_predict = legacy("predict", rows=[{"x": 10.0}], artifact=old_model["artifact"], expectedEnvironment=old_env)["predictions"][0]
results["phase1_artifact_infers_under_phase2"] = abs(predicts(old_model["artifact"], old_env) - old_predict) < 1e-12
old_metrics = legacy("evaluate", prepared=old_prepared, artifact=old_model["artifact"], expectedEnvironment=old_env)["metrics"]
new_metrics = run("evaluate", spec, rows, prepared=old_prepared, artifact=old_model["artifact"], expectedEnvironment=old_env)["metrics"]
results["phase1_artifact_evaluates_identically_under_phase2"] = len(new_metrics) == len(old_metrics) == 6 and all(a["name"] == b["name"] and a["partition"] == b["partition"] and abs(a["value"] - b["value"]) < 1e-12 for a, b in zip(old_metrics, new_metrics))
results["phase1_provenance_is_untouched"] = pickle.loads(base64.b64decode(old_model["artifact"]))["environment"] == old_env
results["phase1_artifact_still_refused_on_runtime_mismatch"] = rejected(old_model["artifact"], {**old_env, "sklearn": "0.0.1"}) and rejected(old_model["artifact"], {**old_env, "python": "2.7.18"})
results["phase1_artifact_refused_if_schema_claimed_unsupported"] = rejected(old_model["artifact"], {**old_env, "artifact_schema": "2"})
results["phase1_artifact_with_misleading_protocol_refused"] = rejected(old_model["artifact"], {**old_env, "protocol": "meticule-sklearn-v0"})
emit(results)
