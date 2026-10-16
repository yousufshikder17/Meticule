"""Contract envelope semantics of ml_runtime_common.serve (amendments A1 and A3) and the contract table shared with
contractFor in src/ml/contract.ts (A5). Prints one JSON line: {"results": {name: bool}}.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import ml_runtime_common as common  # noqa: E402

V0, V1, V2 = "meticule-ml-backend/v0", common.ML_BACKEND_CONTRACT, common.ML_BACKEND_CONTRACT_V2
UUID = "00000000-0000-4000-8000-000000000000"
TASKS = {"ridge": "regression", "logistic_regression": "classification", "embedder": "embedding"}
TABLE = json.loads((ROOT / "tests" / "fixtures" / "contract-table.json").read_text(encoding="utf-8"))["rows"]
results = {}


def snapshot(algorithm="ridge", strategy="random", target="y", kind="number"):
    split = {"id": UUID, "strategy": strategy, "train": 0.6, "validation": 0.2, "test": 0.2}
    split.update({"stratify": False} if strategy == "random" else {"timeColumn": "at"})
    columns = [{"name": "a", "type": kind}, {"name": "at", "type": "timestamp" if strategy != "random" else "number"}]
    schema = {"id": UUID, "columns": columns + ([{"name": target, "type": "number"}] if target else [])}
    if target:
        schema["target"] = target
    return {"algorithm": algorithm, "seed": 1, "hyperparameters": {}, "split": split, "dataset": {"schema": schema},
            "pipeline": {"definition": {"features": ["a"], "steps": []}}}


def call(request, task_of=TASKS.get, execute=lambda request: {"ok": True}):
    """One request through serve with in-memory streams: (status, response, refusal code, execute ran, stderr lines)."""
    ran = []

    def tracked(req):
        ran.append(True)
        return execute(req)

    stdout, stderr = io.StringIO(), io.StringIO()
    status = common.serve(tracked, stdin=io.StringIO(json.dumps(request)), stdout=stdout, stderr=stderr, task_of=task_of)
    lines = stderr.getvalue().strip().splitlines()
    code = json.loads(lines[-1]).get("code") if lines and lines[-1].startswith("{") else None
    out = stdout.getvalue().strip()
    return status, json.loads(out) if out else None, code, bool(ran), lines


def refused(request, code, **kw):
    status, response, got, ran, _ = call(request, **kw)
    return status == 1 and response is None and got == code and not ran


def accepted(request, contract, **kw):
    status, response, code, ran, _ = call(request, **kw)
    return status == 0 and ran and code is None and response == {"ok": True, "contract": contract}


# A1: contract membership first, then the semantic check before execute for every command carrying a snapshot.
results["v0_refused_before_any_work"] = refused({"command": "prepare", "contract": V0, "snapshot": snapshot()}, "unsupported_contract")
results["missing_or_malformed_envelope_refused"] = (refused({"command": "prepare", "snapshot": snapshot()}, "unsupported_contract")
                                                     and refused([V1], "unsupported_contract"))
results["v1_tabular_request_runs_and_echoes_v1"] = accepted({"command": "prepare", "contract": V1, "snapshot": snapshot()}, V1)
results["v2_request_runs_and_echoes_v2"] = (accepted({"command": "train", "contract": V2, "task": "regression", "snapshot": snapshot()}, V2)
                                            and accepted({"command": "prepare", "contract": V2, "task": "regression", "snapshot": snapshot(strategy="chronological")}, V2)
                                            and accepted({"command": "prepare", "contract": V2, "task": "embedding", "snapshot": snapshot("embedder", target=None, kind="text")}, V2))
results["v1_extended_prepare_refused"] = (refused({"command": "prepare", "contract": V1, "snapshot": snapshot(strategy="chronological")}, "unsupported_contract")
                                          and refused({"command": "prepare", "contract": V1, "snapshot": snapshot(target=None)}, "unsupported_contract"))
results["v1_extended_predict_refused"] = (refused({"command": "predict", "contract": V1, "snapshot": snapshot(kind="text"), "artifact": "x"}, "unsupported_contract")
                                          and refused({"command": "predict", "contract": V1, "snapshot": snapshot(kind="image"), "artifact": "x"}, "unsupported_contract"))
results["v1_phase4_task_refused"] = refused({"command": "prepare", "contract": V1, "snapshot": snapshot("embedder")}, "unsupported_contract")
results["v1_unknown_algorithm_refused"] = refused({"command": "prepare", "contract": V1, "snapshot": snapshot("nonexistent")}, "unsupported_contract")
results["v2_task_mismatch_refused"] = refused({"command": "prepare", "contract": V2, "task": "classification", "snapshot": snapshot()}, "unsupported_task")
results["v2_unknown_algorithm_refused"] = refused({"command": "predict", "contract": V2, "task": "regression", "snapshot": snapshot("nonexistent")}, "unsupported_task")
results["v2_missing_task_refused"] = refused({"command": "prepare", "contract": V2, "snapshot": snapshot()}, "unsupported_task")
results["catalog_skips_semantic_check"] = (accepted({"command": "catalog", "contract": V1}, V1)
                                           and accepted({"command": "catalog", "contract": V1, "snapshot": snapshot(target=None)}, V1))
# The conformance suite's raw check: a current-contract request without a snapshot reaches the runner, never a contract refusal.
status, _, code, _, _ = call({"command": "unknown_command", "contract": V1}, execute=lambda request: request["snapshot"])
results["request_without_snapshot_reaches_execute"] = status == 1 and code is None
results["served_without_task_of_speaks_v1_only"] = (accepted({"command": "prepare", "contract": V1, "snapshot": snapshot(strategy="chronological")}, V1, task_of=None)
                                                    and refused({"command": "prepare", "contract": V2, "task": "regression", "snapshot": snapshot()}, "unsupported_contract", task_of=None))
results["refusal_messages_are_fixed"] = all(
    json.loads(call(request)[4][-1])["message"] in ("Unsupported ML backend contract", "Contract v1 does not cover this request",
                                                    "The runner does not implement this task for the algorithm")
    for request in ({"contract": V0}, {"command": "prepare", "contract": V1, "snapshot": snapshot("secret_algorithm_name")},
                    {"command": "prepare", "contract": V2, "task": "secret_task", "snapshot": snapshot()}))

# A5: the table shared with contractFor; a v1 request is served exactly when the table says v1, a v2 one always.
table_ok = len(TABLE) >= 12 and {r["contract"] for r in TABLE} == {V1, V2}
for row in TABLE:
    spec = {"algorithm": "probe", "split": row["split"], "dataset": {"schema": row["schema"]}}
    task_of = {"probe": row["task"]}.get
    table_ok = table_ok and common.uses_extensions(spec, row["task"]) == (row["contract"] == V2)
    v1 = {"command": "prepare", "contract": V1, "snapshot": spec}
    table_ok = table_ok and (accepted(v1, V1, task_of=task_of) if row["contract"] == V1 else refused(v1, "unsupported_contract", task_of=task_of))
    table_ok = table_ok and accepted({"command": "prepare", "contract": V2, "task": row["task"], "snapshot": spec}, V2, task_of=task_of)
results["uses_extensions_matches_contract_table"] = table_ok


# A3: nothing printed while execute runs reaches the protocol channel.
def noisy(request):
    print("python noise")
    sys.stdout.write("more python noise\n")
    return {"ok": True}


status, response, _, _, lines = call({"command": "train", "contract": V1, "snapshot": snapshot()}, execute=noisy)
results["python_stdout_goes_to_stderr_with_custom_streams"] = status == 0 and response == {"ok": True, "contract": V1} and "python noise" in lines

# The real process: Python prints, raw writes to fd 1 and buffered C stdio (as native libraries do) all end up on stderr.
RUNNER = r'''
import ctypes, os, sys
sys.path.insert(0, sys.argv[1])
import ml_runtime_common as common
libc = ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)

def execute(request):
    print("python noise")
    os.write(1, b"native fd noise\n")
    libc.puts(b"buffered native noise")
    if request["command"] == "fail":
        raise common.RunnerError("unsupported_task", "Refused on purpose")
    return {"ok": True}

sys.exit(common.serve(execute, task_of={"ridge": "regression"}.get))
'''


def process(command):
    request = {"command": command, "contract": V1, "snapshot": snapshot()}
    return subprocess.run([sys.executable, "-I", "-c", RUNNER, str(ROOT / "ml")], input=json.dumps(request).encode(),
                          capture_output=True, timeout=120)


def parsed(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return None


done, failed = process("train"), process("fail")
noise = (b"python noise", b"native fd noise", b"buffered native noise")
results["process_stdout_carries_only_the_response"] = (done.returncode == 0 and parsed(done.stdout) == {"ok": True, "contract": V1}
                                                       and done.stdout.strip().count(b"\n") == 0 and all(n in done.stderr for n in noise))
last = (failed.stderr.decode().strip().splitlines() or [""])[-1]
results["refusal_is_last_stderr_line_after_native_noise"] = (failed.returncode == 1 and failed.stdout == b"" and all(n in failed.stderr for n in noise)
                                                             and parsed(last) == {"error": "RunnerError", "code": "unsupported_task", "message": "Refused on purpose"})

print(json.dumps({"results": results}))
