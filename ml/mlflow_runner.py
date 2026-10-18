"""MLflow interoperability runner (Phase 5). Exports one Meticule training attempt to an operator-configured MLflow tracking
server, or reads one MLflow run's metadata for import. It never downloads, loads or uploads model artifacts.

Idempotency: every exported run carries the tag meticule.training_run_id. A run already tagged complete is reused; a run
left incomplete by an interrupted export is deleted and written again, so retries never leave duplicates behind.
"""
import json
import re
import sys

import ml_runtime_common as common
from ml_runtime_common import RunnerError

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_IMPORTED = 200


def client(uri):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=uri)


def export(request):
    from mlflow.entities import Metric, Param, RunTag
    run = request["run"]
    run_id = run["tags"]["meticule.training_run_id"]
    if not UUID.match(run_id):
        raise RunnerError("external_service_unavailable", "Invalid training run identity")
    c = client(request["trackingUri"])
    experiment = c.get_experiment_by_name(request["experimentName"])
    experiment_id = experiment.experiment_id if experiment else c.create_experiment(request["experimentName"], tags=request.get("experimentTags") or {})
    found = c.search_runs([experiment_id], filter_string=f"tags.`meticule.training_run_id` = '{run_id}'", max_results=10)
    for existing in found:
        if existing.data.tags.get("meticule.export_complete") == "true":
            return {"runId": existing.info.run_id, "experimentId": experiment_id, "reused": True}
        c.delete_run(existing.info.run_id)
    created = c.create_run(experiment_id, start_time=run["startTime"], tags=run["tags"], run_name=run["name"])
    rid = created.info.run_id
    params = [Param(k, v) for k, v in sorted(run["params"].items())]
    metrics = [Metric(m["key"], m["value"], run["endTime"], m["step"]) for m in run["metrics"]]
    for i in range(0, max(len(params), 1), 100):
        c.log_batch(rid, params=params[i:i + 100])
    for i in range(0, len(metrics), 1000):
        c.log_batch(rid, metrics=metrics[i:i + 1000])
    c.log_text(rid, json.dumps(run["lineage"], indent=2, sort_keys=True), "meticule/lineage.json")
    c.log_batch(rid, tags=[RunTag("meticule.export_complete", "true")])
    c.set_terminated(rid, status="FINISHED", end_time=run["endTime"])
    return {"runId": rid, "experimentId": experiment_id, "reused": False}


def read(request):
    """Metadata only: parameters, latest metric values, user tags, status and times. Artifacts are never fetched."""
    if not RUN_ID.match(request["mlflowRunId"]):
        raise RunnerError("external_service_unavailable", "Invalid MLflow run id")
    r = client(request["trackingUri"]).get_run(request["mlflowRunId"])
    text = lambda v: str(v)[:500]
    params = dict(sorted((text(k), text(v)) for k, v in r.data.params.items())[:MAX_IMPORTED])
    metrics = dict(sorted((text(k), float(v)) for k, v in r.data.metrics.items() if v == v and abs(v) != float("inf"))[:MAX_IMPORTED])
    tags = dict(sorted((text(k), text(v)) for k, v in r.data.tags.items() if not k.startswith("mlflow."))[:MAX_IMPORTED])
    return {"run": {"runId": r.info.run_id, "experimentId": r.info.experiment_id, "name": text(r.info.run_name or ""), "status": r.info.status,
                    "startTime": r.info.start_time, "endTime": r.info.end_time, "params": params, "metrics": metrics, "tags": tags}}


def execute(request):
    command = request["command"]
    try:
        if command == "export":
            return export(request)
        if command == "read":
            return read(request)
    except RunnerError:
        raise
    except Exception:
        # Connection errors, authentication failures and server refusals: the message may carry the URI or a token prefix.
        raise RunnerError("external_service_unavailable", "MLflow tracking request failed") from None
    raise ValueError("Unknown MLflow command")


if __name__ == "__main__":
    sys.exit(common.serve(execute))
