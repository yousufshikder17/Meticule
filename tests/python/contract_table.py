"""Decides what tests/unit/ml-tasks.test.ts sends on stdin with the runtime's own functions, so the TypeScript side can
compare row for row (spec amendment A5): uses_extensions per snapshot; require_contract per raw envelope (null when
accepted, else the refusal code); the partition sizes prepare produces per split (null when it refuses the split).
Prints one JSON line.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import ml_runtime_common as common  # noqa: E402


def refusal(request, task):
    try:
        common.require_contract(request, task)
    except common.RunnerError as error:
        return error.code
    return None


def sizes(split, n):
    # Every row at one instant: chronological ties keep row order, so only the sizes matter here.
    rows = [{split.get("timeColumn", "t"): "2024-01-01T00:00:00Z"}] * n
    try:
        indices = common.split_partitions({"split": split, "seed": 0}, rows, None)["indices"]
    except (common.RunnerError, ValueError):
        return None
    return [len(indices[name]) for name in ("train", "validation", "test")]


request = json.load(sys.stdin)
print(json.dumps({
    "usesExtensions": {row["id"]: common.uses_extensions(row["snapshot"], row["task"]) for row in request["snapshots"]},
    "envelopes": {envelope["id"]: refusal(envelope["request"], envelope["runnerTask"]) for envelope in request["envelopes"]},
    "sizes": {split["id"]: sizes(split["split"], split["n"]) for split in request["splits"]},
}))
