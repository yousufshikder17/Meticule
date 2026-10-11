"""Runs the frozen Phase 1 adapter behind the v1 runner envelope.

The adapter file is loaded unchanged, so its checksum (adapter_sha256) and the artifacts it wrote remain the original
evidence. Only the transport is shared: the contract check and response stamp come from the current ml_runtime_common.serve,
which is what every runner uses. Test-only; this is not a deployable runner.
"""
import importlib.util
import pathlib
import sys

ADAPTER = pathlib.Path(__file__).with_name("sklearn_runner_phase1.py")
sys.path.insert(0, str(ADAPTER.parents[3] / "ml"))

import ml_runtime_common as common  # noqa: E402

spec = importlib.util.spec_from_file_location("sklearn_runner_phase1", ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

if __name__ == "__main__":
    sys.exit(common.serve(adapter.execute))
