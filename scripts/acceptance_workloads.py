"""Real Redis -> packaged Consumer -> Ray -> ClickHouse/S3 acceptance workloads.
Run ONLY on the isolated acceptance network; writes under acceptance prefixes.
"""

import copy
import json
import logging
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import boto3
import clickhouse_connect
import redis
from botocore.config import Config
from local_smoke import _inference_request, _training_request
from ray.job_submission import JobSubmissionClient

from tributo_knova.broker import KnovaBrokerPlugin

logging.basicConfig(level=logging.INFO)
OUT = Path(os.environ.get("ACCEPT_OUTPUT", "/artifacts"))
OUT.mkdir(exist_ok=True, parents=True)
RUN = uuid.uuid4().hex[:8]
results = []
for attempt in range(120):
    try:
        ch = clickhouse_connect.get_client(
            host="clickhouse",
            username="knova",
            password="local-test-only",
            connect_timeout=3,
        )
        break
    except Exception:
        if attempt == 119:
            raise
        time.sleep(1)
r = redis.Redis(host="repro-redis", password="local-repro-only", decode_responses=True)
for attempt in range(120):
    try:
        jobs = JobSubmissionClient("http://repro-ray:8265")
        break
    except ConnectionError:
        if attempt == 119:
            raise
        time.sleep(1)
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
        channel[key] = f"accept:{RUN}:{name}:{key}"
    channel["group_start_id"] = "0-0"
runtime = KnovaBrokerPlugin().create_runtime(config)
runtime.start()
args = SimpleNamespace(
    storage_mode="nfs",
    s3_bucket="knova-acceptance",
    clickhouse_host="clickhouse",
    clickhouse_port=8123,
    clickhouse_user="knova",
)


def save():
    (OUT / "workload-results.json").write_text(json.dumps(results, indent=2))


def submit(request, name="training", expected="COMPLETED", timeout=300, cancel=False):
    channel = config["channels"][name]
    ident = request[channel["outer_identity_field"]]
    stream = channel["event_stream_prefix"] + ":" + ident
    r.xadd(
        channel["task_stream_key"],
        {channel["outer_identity_field"]: ident, "payload": json.dumps(request)},
    )
    start = time.monotonic()
    assert runtime.run_once(timeout_ms=1000), "consumer did not receive task"
    active = runtime.active_submissions.get(ident)
    if cancel:
        r.set(channel["cancel_key_prefix"] + ":" + ident, "1")
    try:
        while time.monotonic() - start < timeout:
            events = [json.loads(fields["payload"]) for _, fields in r.xrange(stream)]
            terminal = [
                e
                for e in events
                if e["event_type"] in {"COMPLETED", "FAILED", "CANCELLED"}
            ]
            if terminal:
                assert len(terminal) == 1, terminal
                assert terminal[0]["event_type"] == expected, terminal[0]
                return terminal[0]
            time.sleep(0.5)
        raise TimeoutError(ident)
    finally:
        if active:
            info = jobs.get_job_info(active.submission.submission_id)
            (OUT / (ident + ".status.json")).write_text(
                json.dumps(
                    {
                        "status": str(info.status),
                        "error_type": info.error_type,
                        "message": info.message,
                    },
                    indent=2,
                )
            )
            logs = jobs.get_job_logs(active.submission.submission_id)
            (OUT / (ident + ".ray.log")).write_text(logs)
        (OUT / (ident + ".events.json")).write_text(
            json.dumps(
                [json.loads(f["payload"]) for _, f in r.xrange(stream)], indent=2
            )
        )


def record(name, fn):
    start = time.monotonic()
    try:
        detail = fn()
        results.append(
            {
                "case": name,
                "passed": True,
                "seconds": round(time.monotonic() - start, 2),
                "detail": detail,
            }
        )
        print("PASS", name, flush=True)
        return detail
    except Exception as exc:
        results.append(
            {
                "case": name,
                "passed": False,
                "seconds": round(time.monotonic() - start, 2),
                "error": str(exc),
            }
        )
        logging.exception("FAIL %s", name)
        return None
    finally:
        save()


def create_table(name, rows, width=116, classes=2):
    columns = ", ".join(f"f{i} Float32" for i in range(1, width + 1))
    values = ", ".join(
        f"toFloat32((number*{i * 2 + 1})%{31 + i})/{31 + i}"
        for i in range(1, width + 1)
    )
    ch.command(
        f"CREATE TABLE knova.{name} (entity_id UInt64, {columns}, "
        "label UInt8, stat_month String) ENGINE=MergeTree ORDER BY entity_id"
    )
    ch.command(
        f"INSERT INTO knova.{name} SELECT number, {values}, "
        f"toUInt8(number%{classes}), '202601' FROM numbers({rows})"
    )


def training(name, table, width=116, mode="nfs", multiclass=False):
    job = f"accept-{RUN}-{name}"
    local_args = copy.copy(args)
    local_args.storage_mode = mode
    request = _training_request(
        local_args,
        job_id=job,
        table=table,
        storage_root="/accept-shared",
        storage_prefix=job,
        password="local-test-only",
    )
    request["features"] = [
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
        for i in range(1, width + 1)
    ]
    projection = ", ".join(f"t0.f{i} AS t0__f{i}" for i in range(1, width + 1))
    request["data_query"]["query"]["sql"] = (
        f"SELECT {projection}, t0.label AS t0__label FROM knova.{table} t0"
    )
    request["algorithm"]["hyper_params"] = {"n_estimators": 5, "max_depth": 3}
    if multiclass:
        request["target"]["task_type"] = "MULTICLASS_CLASSIFICATION"
        request["target"]["label_mapping"] = {"a": 0, "b": 1, "c": 2}
        request["algorithm"]["hyper_params"]["num_class"] = 3
    terminal = submit(request, timeout=420)
    return {
        "job_id": job,
        "rows": terminal["result_summary"]["sample_rows"],
        "artifact_manifest": terminal["artifact_manifest"],
    }


try:
    ch.command("CREATE DATABASE IF NOT EXISTS knova")
    big = f"accept_{RUN}_big"
    small = f"accept_{RUN}_small"
    multi = f"accept_{RUN}_multi"
    output = f"accept_{RUN}_output"
    create_table(big, 300000)
    create_table(small, 300)
    create_table(multi, 3000, classes=3)
    ch.command(
        f"CREATE TABLE knova.{output} (entity_id String, "
        "pred_label Nullable(String), pred_probability Nullable(Float64), "
        "pred_probabilities Nullable(String), pred_value Nullable(Float64), "
        "pred_extra String, execution_id String, model_id String, "
        "version_id String, tenant_id String, inferred_at DateTime64(6)) "
        "ENGINE=MergeTree ORDER BY (execution_id,entity_id)"
    )
    large = record("train_300000_rows_116_features", lambda: training("large", big))
    if large:
        assert large["rows"]["total"] == 300000, large["rows"]
        counts = []
        inference_templates = []
        for threshold in (0.0, 0.5, 1.0):

            def infer(threshold=threshold):
                ident = f"accept-{RUN}-threshold-{threshold}"
                request = _inference_request(
                    args,
                    execution_id=ident,
                    input_table=small,
                    output_table=output,
                    artifact_manifest=large["artifact_manifest"],
                    password="local-test-only",
                    shap_mode="none",
                )
                request["input"]["features"] = [
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
                request["input"]["query"]["sql"] = (
                    f"SELECT t0.entity_id AS entity_id, {projection} "
                    f"FROM knova.{small} t0 WHERE t0.stat_month = {{month:String}}"
                )
                request["extensions"] = {"decision_threshold": threshold}
                request["execution"]["progress_report_interval_rows"] = 50
                inference_templates.append(copy.deepcopy(request))
                terminal = submit(request, "batch_inference")
                events = json.loads((OUT / (ident + ".events.json")).read_text())
                assert sum(event["event_type"] == "STARTED" for event in events) == 1, (
                    events
                )
                progress = [
                    event for event in events if event["event_type"] == "PROGRESS"
                ]
                assert progress[-1]["percent"] == 100.0, progress
                assert any(0 < event["processed_rows"] < 300 for event in progress), (
                    progress
                )
                rows = ch.query(
                    f"SELECT pred_label,pred_probability FROM knova.{output} "
                    "WHERE execution_id={id:String} ORDER BY entity_id",
                    parameters={"id": ident},
                ).result_rows
                assert len(rows) == 300 == terminal["result_rows"], (
                    len(rows),
                    terminal,
                )
                for label, probability in rows:
                    assert label == (
                        "churn" if probability >= threshold else "active"
                    ), (label, probability, threshold)
                count = sum(label == "churn" for label, _ in rows)
                counts.append(count)
                return {
                    "rows": len(rows),
                    "threshold": threshold,
                    "positive_predictions": count,
                }

            record("infer_threshold_" + str(threshold), infer)
        if len(counts) == 3:
            record(
                "threshold_monotonicity",
                lambda: (
                    {"counts": counts}
                    if counts[0] >= counts[1] >= counts[2]
                    else (_ for _ in ()).throw(AssertionError(counts))
                ),
            )
        if inference_templates:
            template = (
                inference_templates[1]
                if len(inference_templates) > 1
                else inference_templates[0]
            )

            def reordered_features():
                req = copy.deepcopy(template)
                req["execution_id"] = f"accept-{RUN}-reordered"
                req["input"]["features"].reverse()
                terminal = submit(req, "batch_inference")
                sql = (
                    f"SELECT entity_id,pred_label,pred_probability FROM knova.{output} "
                    "WHERE execution_id={id:String} ORDER BY entity_id"
                )
                original = ch.query(
                    sql, parameters={"id": template["execution_id"]}
                ).result_rows
                reordered = ch.query(
                    sql, parameters={"id": req["execution_id"]}
                ).result_rows
                assert reordered == original, "declaration order changed predictions"
                return {"rows": terminal["result_rows"], "identical_predictions": True}

            record("infer_reordered_features", reordered_features)
            for case, change, expected_code in [
                (
                    "infer_hash_mismatch",
                    lambda q: q["model"]["model_artifacts"]["model_weights"][
                        "alternatives"
                    ][0]["files"][0].update(hash="sha256:" + "0" * 64),
                    "MODEL_ARTIFACT_CORRUPTED",
                ),
                (
                    "infer_required_preprocessor",
                    lambda q: q["model"]["model_artifacts"].update(
                        preprocessor={
                            "required_for_inference": True,
                            "alternatives": [
                                {
                                    "format": "sklearn",
                                    "files": [
                                        {
                                            "path": "missing-preprocessor.joblib",
                                            "size": 1,
                                            "hash": "sha256:" + "a" * 64,
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                    "MODEL_FORMAT_UNSUPPORTED",
                ),
                (
                    "infer_sql_expression",
                    lambda q: q["input"]["query"].update(
                        sql=q["input"]["query"]["sql"].replace(
                            "t0.f1 AS", "t0.f1 * 100 AS"
                        )
                    ),
                    "INVALID_PAYLOAD",
                ),
                (
                    "infer_input_column_missing",
                    lambda q: (
                        q["input"]["features"][0]["origin"].update(
                            column_name="missing_feature"
                        ),
                        q["input"]["query"].update(
                            sql=q["input"]["query"]["sql"].replace(
                                "t0.f1 AS", "t0.missing_feature AS"
                            )
                        ),
                    ),
                    "INPUT_COLUMN_MISSING",
                ),
                (
                    "infer_timeout",
                    lambda q: q["execution"].update(max_execution_time_seconds=1),
                    "EXECUTION_TIMEOUT",
                ),
            ]:

                def inference_failure(
                    case=case, change=change, expected_code=expected_code
                ):
                    req = copy.deepcopy(template)
                    req["execution_id"] = f"accept-{RUN}-{case}"
                    change(req)
                    terminal = submit(req, "batch_inference", expected="FAILED")
                    assert terminal["error_code"] == expected_code, terminal
                    count = ch.query(
                        f"SELECT count() FROM knova.{output} "
                        "WHERE execution_id={id:String}",
                        parameters={"id": req["execution_id"]},
                    ).first_row[0]
                    assert count == 0, count
                    return terminal["error_code"]

                record(case, inference_failure)
    record(
        "train_multiclass_116_features",
        lambda: training("multi", multi, multiclass=True),
    )
    s3 = boto3.client(
        "s3",
        endpoint_url="http://rustfs:9000",
        aws_access_key_id="localtestkey",
        aws_secret_access_key="local-test-secret",
        region_name="us-east-1",
        config=Config(connect_timeout=3, read_timeout=15, retries={"max_attempts": 2}),
    )
    if "knova-acceptance" not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
        s3.create_bucket(Bucket="knova-acceptance")
    record("train_publish_s3", lambda: training("s3", small, mode="s3"))
    base = _training_request(
        args,
        job_id=f"accept-{RUN}-bad",
        table=small,
        storage_root="/accept-shared",
        storage_prefix="errors",
        password="local-test-only",
    )
    for name, change in [
        ("unknown_algorithm", lambda q: q["algorithm"].update(algorithm_key="missing")),
        (
            "missing_feature",
            lambda q: q.update(
                features=[{"origin": {"column_name": "does_not_exist"}}]
            ),
        ),
        ("empty_features", lambda q: q.update(features=[])),
        (
            "clickhouse_bad_password",
            lambda q: q["datasource"].update(password="wrong-test-password"),
        ),
        ("clickhouse_unreachable", lambda q: q["datasource"].update(port=1)),
        (
            "storage_unwritable_prefix",
            lambda q: q.update(
                storage_context={
                    "type": "nfs",
                    "bucket": "/proc",
                    "prefix": "cannot-write/",
                    "properties": {},
                }
            ),
        ),
    ]:

        def negative(name=name, change=change):
            req = copy.deepcopy(base)
            req["job_id"] = f"accept-{RUN}-{name}"
            change(req)
            result = submit(req, expected="FAILED")
            allowed = {
                "unknown_algorithm": {"INVALID_PAYLOAD"},
                "missing_feature": {"INVALID_PAYLOAD"},
                "empty_features": {"INVALID_PAYLOAD"},
                "clickhouse_bad_password": {
                    "DATA_QUERY_FAILED",
                    "DATA_SOURCE_UNREACHABLE",
                },
                "clickhouse_unreachable": {
                    "DATA_QUERY_FAILED",
                    "DATA_SOURCE_UNREACHABLE",
                },
                "storage_unwritable_prefix": {"TRAINING_FAILED", "MODEL_EXPORT_FAILED"},
            }
            assert result["error_code"] in allowed[name], result
            assert "wrong-test-password" not in json.dumps(result), result
            return result["error_code"]

        record(name, negative)

    def oversized():
        req = copy.deepcopy(base)
        req["job_id"] = f"accept-{RUN}-oversized"
        req["extensions"]["padding"] = "x" * 210000
        result = submit(req, expected="FAILED")
        assert result["error_code"] == "INVALID_PAYLOAD", result
        return result["error_code"]

    record("payload_base64_over_256k", oversized)

    def within_limit():
        req = copy.deepcopy(base)
        req["job_id"] = f"accept-{RUN}-large-input"
        req["algorithm"]["algorithm_key"] = "missing"
        req["extensions"]["padding"] = "x" * 150000
        result = submit(req, expected="FAILED")
        logs = (OUT / (req["job_id"] + ".ray.log")).read_text()
        assert "Driver input loaded:" in logs, logs[-1500:]
        assert "Argument list too long" not in logs
        assert "Traceback" in logs
        return {
            "code": result["error_code"],
            "input_padding_bytes": 150000,
            "driver_started": True,
        }

    record("payload_200k_reaches_driver", within_limit)

    def queued_cancel():
        req = copy.deepcopy(base)
        req["job_id"] = f"accept-{RUN}-cancelled"
        r.set(
            config["channels"]["training"]["cancel_key_prefix"] + ":" + req["job_id"],
            "1",
        )
        return submit(req, expected="CANCELLED")["event_type"]

    record("cancel_before_admission", queued_cancel)

    def active_cancel():
        req = copy.deepcopy(base)
        req["job_id"] = f"accept-{RUN}-running-cancel"
        req["algorithm"]["hyper_params"]["n_estimators"] = 100000
        return submit(req, expected="CANCELLED", cancel=True)["event_type"]

    record("cancel_after_admission", active_cancel)

    def training_timeout():
        req = copy.deepcopy(base)
        req["job_id"] = f"accept-{RUN}-training-timeout"
        req["resource_limits"] = {"max_training_time_seconds": 1}
        result = submit(req, expected="FAILED")
        assert result["error_code"] == "TRAINING_TIMEOUT", result
        return result["error_code"]

    record("train_timeout", training_timeout)
finally:
    runtime.close()
    ch.close()
    r.close()
    save()
if any(not result["passed"] for result in results):
    raise SystemExit(1)
