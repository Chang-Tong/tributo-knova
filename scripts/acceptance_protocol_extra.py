"""Additional protocol cases against an existing isolated acceptance stack.

Run after acceptance_workloads.py using its /artifacts results and shared models.
Uses a fresh Redis namespace and appends only new execution IDs to its output table.
"""

import copy
import json
import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import clickhouse_connect
import redis
from local_smoke import _inference_request
from ray.job_submission import JobSubmissionClient

from tributo_knova.broker import KnovaBrokerPlugin

out = Path("/artifacts")
prior = json.loads((out / "workload-results.json").read_text())
large = next(
    item["detail"]
    for item in prior
    if item["case"] == "train_300000_rows_116_features" and item["passed"]
)
old_run = large["job_id"].split("-")[1]
run = uuid.uuid4().hex[:8]
config = json.loads(Path("/etc/tributo-knova/config.json").read_text())
config["transport"] = {
    "url": "redis://repro-redis:6379/0",
    "password_ref": "env:KNOVA_REDIS_PASSWORD",
    "block_ms": 100,
    "claim_idle_ms": 500,
    "max_driver_input_bytes": 262144,
}
config["execution"]["ray_dashboard_url"] = "http://repro-ray:8265"
for name, channel in config["channels"].items():
    for key in (
        "task_stream_key",
        "event_stream_prefix",
        "cancel_key_prefix",
        "consumer_group",
    ):
        channel[key] = f"accept-extra:{run}:{name}:{key}"
    channel["group_start_id"] = "0-0"
r = redis.Redis(host="repro-redis", password="local-repro-only", decode_responses=True)
ch = clickhouse_connect.get_client(
    host="clickhouse", username="knova", password="local-test-only"
)
jobs = JobSubmissionClient("http://repro-ray:8265")
runtime = KnovaBrokerPlugin().create_runtime(config)
runtime.start()
channel = config["channels"]["batch_inference"]
args = SimpleNamespace(
    clickhouse_host="clickhouse", clickhouse_port=8123, clickhouse_user="knova"
)
output = f"accept_{old_run}_output"
source = f"accept_{old_run}_small"
results = []


def request(name, manifest=None):
    req = _inference_request(
        args,
        execution_id=f"extra-{run}-{name}",
        input_table=source,
        output_table=output,
        artifact_manifest=copy.deepcopy(manifest or large["artifact_manifest"]),
        password="local-test-only",
        shap_mode="none",
    )
    req["input"]["features"] = [
        {
            "feature_id": f"f{i}",
            "display_name": f"Feature {i}",
            "feature_role": "REGULAR",
            "origin": {
                "table_alias": "t0",
                "column_name": f"f{i}",
                "column_type": "Float32",
            },
            "result_column": f"t0__f{i}",
        }
        for i in range(1, 117)
    ]
    projection = ", ".join(f"t0.f{i} AS t0__f{i}" for i in range(1, 117))
    req["input"]["query"] = {
        "sql": (
            f"SELECT t0.entity_id AS entity_id, {projection} "
            f"FROM knova.{source} t0 WHERE t0.stat_month = {{month:String}}"
        ),
        "params": {"month": "202601"},
    }
    req["execution"]["progress_report_interval_rows"] = 50
    return req


def submit(req, expected="COMPLETED"):
    ident = req["execution_id"]
    stream = channel["event_stream_prefix"] + ":" + ident
    r.xadd(
        channel["task_stream_key"], {"execution_id": ident, "payload": json.dumps(req)}
    )
    assert runtime.run_once(timeout_ms=1000)
    active = runtime.active_submissions.get(ident)
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            events = [json.loads(f["payload"]) for _, f in r.xrange(stream)]
            terminal = [
                e
                for e in events
                if e["event_type"] in {"COMPLETED", "FAILED", "CANCELLED"}
            ]
            if terminal:
                assert len(terminal) == 1 and terminal[0]["event_type"] == expected, (
                    terminal
                )
                assert not r.xpending(
                    channel["task_stream_key"], channel["consumer_group"]
                )["pending"]
                return terminal[0]
            time.sleep(0.5)
        raise TimeoutError(ident)
    finally:
        (out / (ident + ".events.json")).write_text(
            json.dumps(
                [json.loads(f["payload"]) for _, f in r.xrange(stream)], indent=2
            )
        )
        if active:
            (out / (ident + ".ray.log")).write_text(
                jobs.get_job_logs(active.submission.submission_id)
            )


def rows(ident):
    return ch.query(
        "SELECT entity_id,pred_label,pred_probability,pred_probabilities,pred_extra "
        f"FROM knova.{output} WHERE execution_id={{id:String}} ORDER BY entity_id",
        parameters={"id": ident},
    ).result_rows


def record(name, fn):
    try:
        detail = fn()
        results.append({"case": name, "passed": True, "detail": detail})
        print("PASS", name, flush=True)
    except Exception as exc:
        results.append({"case": name, "passed": False, "error": str(exc)})
        print("FAIL", name, type(exc).__name__, str(exc), flush=True)
    (out / "extra-protocol-results.json").write_text(json.dumps(results, indent=2))


def invalid_utf8():
    ident = f"extra-{run}-utf8"
    r.xadd(channel["task_stream_key"], {"execution_id": ident, "payload": b"\xff"})
    assert runtime.run_once(timeout_ms=1000)
    event = json.loads(
        r.xrange(channel["event_stream_prefix"] + ":" + ident)[-1][1]["payload"]
    )
    assert (
        event["event_type"] == "FAILED" and event["error_code"] == "INVALID_PAYLOAD"
    ), event
    assert not r.xpending(channel["task_stream_key"], channel["consumer_group"])[
        "pending"
    ]
    return event["error_code"]


def native_only():
    req = request("native")
    req["model"]["model_artifacts"]["model_weights"]["alternatives"] = [
        alt
        for alt in req["model"]["model_artifacts"]["model_weights"]["alternatives"]
        if alt["format"] == "xgboost"
    ]
    req["input"]["features"].reverse()
    submit(req)
    actual = rows(req["execution_id"])
    expected = rows(f"accept-{old_run}-threshold-0.5")
    assert len(actual) == len(expected) == 300
    for a, b in zip(actual, expected, strict=True):
        assert a[:2] == b[:2] and abs(a[2] - b[2]) < 1e-6, (a[:3], b[:3])
    return {"rows": len(actual), "matches_onnx": True}


def shap_credentials():
    req = request("shap")
    req["extensions"] = {
        "explanation": {"enabled": True, "method": "TREE_SHAP", "approximate": False}
    }
    req["input"]["features"].reverse()
    for ds in [req["input"]["datasource"], req["output"]["datasource"]]:
        ds.update(
            password="wrong-shadowed-password",
            credential_ref="env://KNOVA_OUTPUT_CLICKHOUSE_PASSWORD",
        )
    submit(req)
    actual = rows(req["execution_id"])
    assert len(actual) == 300
    for row in actual:
        exp = json.loads(row[4])["explanation"]
        margin = exp["base_value"] + sum(
            item["shap_value"] for item in exp["feature_contributions"]
        )
        assert abs(1 / (1 + math.exp(-margin)) - row[2]) < 1e-5
        assert [item["feature_name"] for item in exp["feature_contributions"]] == [
            f"t0__f{i}" for i in range(1, 117)
        ]
    return {
        "rows": 300,
        "shap_matches_probability": True,
        "credential_ref_precedence": True,
    }


def output_failure():
    req = request("writer-fail")
    req["output"]["datasource"]["password"] = "wrong-writer-password"
    req["output"]["datasource"].pop("credential_ref", None)
    terminal = submit(req, "FAILED")
    assert (
        terminal["error_code"] == "RESULT_WRITE_FAILED"
        and terminal["phase"] == "WRITE_RESULT"
    ), terminal
    assert terminal["processed_rows"] == 300, terminal
    assert "wrong-writer-password" not in json.dumps(terminal)
    return {
        "code": terminal["error_code"],
        "processed_rows": terminal["processed_rows"],
        "phase": terminal["phase"],
    }


try:
    record("invalid_utf8_then_continue", invalid_utf8)
    record("native_model_schema_sidecar", native_only)
    record("tree_shap_and_credential_ref", shap_credentials)
    record("writer_failure_snapshot", output_failure)
finally:
    runtime.close()
    ch.close()
    r.close()
if any(not item["passed"] for item in results):
    raise SystemExit(1)
