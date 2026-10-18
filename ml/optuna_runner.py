"""Optuna ask runner (Phase 5). Meticule owns trial execution and persistence; this process only asks Optuna's sampler for
the next parameters.

Each call rebuilds an in-memory study from the durable trial history Meticule sends (completed trials with their objective
value, failed or cancelled trials as FAIL, unfinished ones as RUNNING) and asks `count` new trials. The sampler seed is the
study seed plus the number of known trials, so a rebuild is deterministic for a given history and never repeats the draws
of an earlier rebuild. The search space arrives as validated bounds and choices; nothing in it is evaluated as code.
"""
import math
import sys

import ml_runtime_common as common
from ml_runtime_common import RunnerError

SAMPLERS = ("tpe", "random")
STATES = ("COMPLETE", "FAIL", "RUNNING")


def distributions(space):
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    out = {}
    for name, d in sorted(space.items()):
        kind = d.get("type")
        if kind == "choice":
            out[name] = CategoricalDistribution(tuple(d["values"]))
        elif kind == "int":
            out[name] = IntDistribution(int(d["min"]), int(d["max"]), step=int(d.get("step") or 1))
        elif kind == "float" and math.isfinite(d["min"]) and math.isfinite(d["max"]):
            out[name] = FloatDistribution(float(d["min"]), float(d["max"]), log=bool(d.get("log")))
        else:
            raise RunnerError("invalid_search_space", "Unsupported search dimension")
    return out


def ask(request):
    import optuna
    from optuna.trial import TrialState, create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_type, direction = request["sampler"], request["direction"]
    if sampler_type not in SAMPLERS or direction not in ("minimize", "maximize"):
        raise RunnerError("invalid_search_space", "Unsupported sampler or direction")
    dists = distributions(request["space"])
    seed = (int(request["seed"]) + len(request["trials"])) % 2**32
    sampler = optuna.samplers.TPESampler(seed=seed, constant_liar=True) if sampler_type == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    for t in request["trials"]:
        if t["state"] not in STATES:
            raise RunnerError("invalid_search_space", "Unknown trial state")
        params = {k: v for k, v in t["params"].items() if k in dists}
        if set(params) != set(dists):
            continue  # a trial from a different space cannot inform this one
        try:
            study.add_trial(create_trial(params=params, distributions=dists, state=TrialState[t["state"]], value=t["value"] if t["state"] == "COMPLETE" else None))
        except ValueError:
            raise RunnerError("invalid_search_space", "A recorded trial lies outside the search space") from None
    trials = [study.ask(dists).params for _ in range(int(request["count"]))]
    return {"trials": trials, "sampler": type(sampler).__name__, "seed": seed, "optuna": optuna.__version__}


def execute(request):
    if request["command"] == "ask":
        return ask(request)
    raise ValueError("Unknown Optuna command")


if __name__ == "__main__":
    sys.exit(common.serve(execute))
