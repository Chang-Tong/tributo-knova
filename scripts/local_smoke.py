#!/usr/bin/env python3
"""Run a local KnoVa Redis -> Ray -> ClickHouse training/inference smoke test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import clickhouse_connect
import redis

from tributo_knova.broker import KnovaBrokerPlugin

_TERMINAL_EVENTS = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-root",
        type=Path,
        default=Path(
            os.environ.get("TRIBUTO_CORE_ROOT", repository.parent / "tributo")
        ),
    )
    parser.add_argument(
        "--broker-root",
        type=Path,
        default=repository.parent / "tributo-broker-redis",
    )
    parser.add_argument(
        "--algorithms-root",
        type=Path,
        default=repository.parent / "tributo-algorithms",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("TRIBUTO_KNOVA_REDIS_URL", "redis://127.0.0.1:6380/3"),
    )
    parser.add_argument(
        "--ray-dashboard-url",
        default=os.environ.get(
            "TRIBUTO_KNOVA_RAY_DASHBOARD_URL", "http://127.0.0.1:8265"
        ),
    )
    parser.add_argument(
        "--clickhouse-host",
        default=os.environ.get("TRIBUTO_KNOVA_CLICKHOUSE_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--clickhouse-port",
        type=int,
        default=int(os.environ.get("TRIBUTO_KNOVA_CLICKHOUSE_PORT", "8123")),
    )
    parser.add_argument(
        "--clickhouse-user",
        default=os.environ.get("TRIBUTO_KNOVA_CLICKHOUSE_USER", "knova"),
    )
    parser.add_argument(
        "--clickhouse-password-env",
        default="TRIBUTO_KNOVA_CLICKHOUSE_PASSWORD",
    )
    parser.add_argument(
        "--shap-mode",
        choices=("none", "exact", "approximate"),
        default="none",
        help="Enable XGBoost TreeSHAP and verify its ClickHouse result payload.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=("nfs", "s3"),
        default="nfs",
        help="Publish the final Bundle locally or to an S3-compatible store.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("TRIBUTO_KNOVA_SMOKE_S3_BUCKET", "tributo-knova-smoke"),
    )
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument(
        "--exercise-recovery",
        action="store_true",
        help="verify pending recovery and cancellation after Consumer restart",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _required_directory(path: Path, marker: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not (resolved / marker).is_dir():
        raise ValueError(f"required package directory is missing below {resolved}")
    return resolved


def _runtime_config(args: argparse.Namespace, repository: Path) -> dict[str, Any]:
    core = _required_directory(args.core_root, Path("src/tributo"))
    broker = _required_directory(args.broker_root, Path("src/tributo_broker_redis"))
    algorithms = _required_directory(
        args.algorithms_root,
        Path("packages/boosting/src/tributo_algorithms_boosting"),
    )
    python_bin = str(Path(sys.executable).absolute().parent)
    current_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    runtime_path = os.pathsep.join(
        dict.fromkeys((python_bin, *current_path.split(os.pathsep)))
    )
    return {
        "broker_id": "tributo-knova",
        "api_version": 1,
        "transport": {
            "mode": "standalone",
            "url": args.redis_url,
            "driver_url": args.redis_url,
            "block_ms": 100,
            "claim_idle_ms": 1_000,
        },
        "channels": {
            "training": {
                "task_stream_key": "knova:aimodel:training:distributed:tasks",
                "event_stream_prefix": "knova:aimodel:training:distributed:events",
                "cancel_key_prefix": "knova:aimodel:training:distributed:cancel",
                "consumer_group": "knova-trainers",
                "consumer_name": f"local-smoke-training-{os.getpid()}",
                "group_start_id": "0-0",
                "outer_identity_field": "job_id",
            },
            "batch_inference": {
                "task_stream_key": "knova:aimodel:inference:distributed:tasks",
                "event_stream_prefix": "knova:aimodel:inference:distributed:events",
                "cancel_key_prefix": "knova:aimodel:inference:distributed:cancel",
                "consumer_group": "knova-backend-inference",
                "consumer_name": f"local-smoke-inference-{os.getpid()}",
                "group_start_id": "0-0",
                "outer_identity_field": "execution_id",
            },
        },
        "operations": {
            "training": {"execution_profiles": ["distributed"]},
            "batch_inference": {"execution_profiles": ["distributed"]},
        },
        "execution": {
            "ray_dashboard_url": args.ray_dashboard_url,
            "project_root": str(core),
            "extra_py_modules": [
                str(repository / "src/tributo_knova"),
                str(broker / "src/tributo_broker_redis"),
                str(algorithms / "packages/boosting/src/tributo_algorithms_boosting"),
            ],
            "env_vars": {"PATH": runtime_path},
            "entrypoint_num_cpus": 0,
        },
    }


def _events(
    client: redis.Redis,
    *,
    stream: str,
) -> list[dict[str, Any]]:
    return [
        json.loads(fields["payload"]) for _event_id, fields in client.xrange(stream)
    ]


def _wait_terminal(
    client: redis.Redis,
    *,
    stream: str,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = _events(client, stream=stream)
        terminal = [
            event for event in events if event.get("event_type") in _TERMINAL_EVENTS
        ]
        if terminal:
            if len(terminal) != 1:
                raise RuntimeError(
                    f"expected one terminal event, received {len(terminal)}"
                )
            return terminal[0], events
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for terminal event on {stream}")


def _submit(
    runtime: Any,
    client: redis.Redis,
    *,
    task_stream: str,
    identity_field: str,
    operation_id: str,
    request: dict[str, Any],
) -> None:
    client.xadd(
        task_stream,
        {identity_field: operation_id, "payload": json.dumps(request)},
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if runtime.run_once(timeout_ms=100):
            return
    raise TimeoutError(f"broker did not admit {operation_id}")


def _require_success(terminal: dict[str, Any], operation_id: str) -> None:
    if terminal.get("event_type") != "COMPLETED":
        error = terminal.get("error_code") or terminal.get("event_type")
        raise RuntimeError(f"{operation_id} ended with {error}")


def _training_request(
    args: argparse.Namespace,
    *,
    job_id: str,
    table: str,
    storage_root: str,
    storage_prefix: str,
    password: str,
) -> dict[str, Any]:
    storage_context = {
        "type": args.storage_mode,
        "bucket": storage_root if args.storage_mode == "nfs" else args.s3_bucket,
        "prefix": storage_prefix,
        "properties": (
            {} if args.storage_mode == "nfs" else {"ray_storage_path": storage_root}
        ),
    }
    return {
        "protocol_version": "2.0",
        "job_id": job_id,
        "model_id": "local-xgboost",
        "version_id": "v1",
        "tenant_id": "local-smoke",
        "algorithm": {
            "algorithm_key": "xgboost",
            "hyper_params": {"n_estimators": 3, "max_depth": 3},
        },
        "datasource": {
            "type": "CLICKHOUSE",
            "host": args.clickhouse_host,
            "port": args.clickhouse_port,
            "database_name": "knova",
            "username": args.clickhouse_user,
            "password": password,
            "properties": {"native_table": f"knova.{table}"},
        },
        "features": [
            {"origin": {"column_name": "f1"}},
            {"origin": {"column_name": "f2"}},
        ],
        "target": {
            "origin": {"column_name": "label"},
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"active": 0, "churn": 1},
        },
        "storage_context": storage_context,
        "extensions": {
            "tributo": {
                "training_runtime": {"ray": {"num_workers": 2, "cpus_per_worker": 1}}
            }
        },
    }


def _inference_request(
    args: argparse.Namespace,
    *,
    execution_id: str,
    input_table: str,
    output_table: str,
    artifact_manifest: dict[str, Any],
    password: str,
    shap_mode: str,
) -> dict[str, Any]:
    request = {
        "protocol_version": "2.0",
        "execution_id": execution_id,
        "task_id": "local-smoke-task",
        "tenant_id": "local-smoke",
        "model": {
            "model_id": "local-xgboost",
            "version_id": "v1",
            "algorithm_key": "xgboost",
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"active": 0, "churn": 1},
            "positive_label_value": "churn",
            "storage": artifact_manifest["storage"],
            "model_artifacts": artifact_manifest["model_artifacts"],
        },
        "input": {
            "datasource": {
                "type": "CLICKHOUSE",
                "host": args.clickhouse_host,
                "port": args.clickhouse_port,
                "database_name": "knova",
                "username": args.clickhouse_user,
                "password": password,
                "properties": {},
            },
            "tables": [
                {
                    "table_alias": "t0",
                    "database_name": "knova",
                    "table_name": input_table,
                    "role": "PRIMARY",
                }
            ],
            "entity_key": {
                "origin": {"table_alias": "t0", "column_name": "entity_id"},
                "result_column": "entity_id",
            },
            "query": {
                "sql": (
                    f"SELECT * FROM knova.{input_table} t0 "
                    "WHERE t0.stat_month = {month:String}"
                ),
                "params": {"month": "202601"},
            },
            "features": [
                {
                    "origin": {"table_alias": "t0", "column_name": "f1"},
                    "result_column": "t0__f1",
                },
                {
                    "origin": {"table_alias": "t0", "column_name": "f2"},
                    "result_column": "t0__f2",
                },
            ],
        },
        "output": {
            "datasource": {
                "type": "CLICKHOUSE",
                "host": args.clickhouse_host,
                "port": args.clickhouse_port,
                "database_name": "knova",
                "username": args.clickhouse_user,
                "password": password,
                "properties": {},
            },
            "table_name": output_table,
            "result_filter": None,
        },
        "execution": {"batch_size": 64, "concurrency": 2},
    }
    if shap_mode != "none":
        request["extensions"] = {
            "explanation": {
                "enabled": True,
                "method": "TREE_SHAP",
                "approximate": shap_mode == "approximate",
            }
        }
    return request


def _bundle_uri(artifact_manifest: dict[str, Any]) -> str:
    storage = artifact_manifest["storage"]
    prefix = storage["prefix"].rstrip("/")
    if storage["type"] == "s3":
        return f"s3://{storage['bucket']}/{prefix}"
    return str(Path(storage["bucket"]) / prefix)


def _s3_client() -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_REGION"),
        config=Config(s3={"addressing_style": "path"}),
    )


def _ensure_s3_bucket(client: Any, bucket: str) -> None:
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client.create_bucket(Bucket=bucket)


def _s3_bundle_files(client: Any, bundle_uri: str) -> list[str]:
    parsed = urlsplit(bundle_uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    response = client.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix)
    files = [
        item["Key"].removeprefix(prefix)
        for item in response.get("Contents", ())
        if item["Key"] != prefix
    ]
    if "manifest.json" not in files:
        raise RuntimeError("S3 Bundle manifest was not published")
    return sorted(files)


def _delete_s3_prefix(client: Any, bucket: str, prefix: str) -> None:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", ())]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects})


def _checkpoint_artifacts(storage_root: str) -> list[str]:
    names = sorted(
        {
            path.name
            for path in Path(storage_root).rglob("*")
            if path.is_file() and path.name in {"model.ubj", "feature_names.json"}
        }
    )
    if names != ["feature_names.json", "model.ubj"]:
        raise RuntimeError("Ray shared storage does not contain an XGBoost checkpoint")
    return names


def _exercise_recovery(
    args: argparse.Namespace,
    *,
    runtime_config: dict[str, Any],
    client: redis.Redis,
    table: str,
    storage_root: str,
    storage_prefix: str,
    password: str,
    suffix: str,
) -> dict[str, Any]:
    task_stream = "knova:aimodel:training:distributed:tasks"
    event_prefix = "knova:aimodel:training:distributed:events"
    cancel_prefix = "knova:aimodel:training:distributed:cancel"
    group = "knova-trainers"

    pending_id = f"knova-pending-recovery-{suffix}"
    pending_request = _training_request(
        args,
        job_id=pending_id,
        table=table,
        storage_root=storage_root,
        storage_prefix=f"{storage_prefix}/pending",
        password=password,
    )
    client.xadd(
        task_stream,
        {"job_id": pending_id, "payload": json.dumps(pending_request)},
    )
    claimed_by_dead_consumer = client.xreadgroup(
        groupname=group,
        consumername=f"dead-consumer-{suffix}",
        streams={task_stream: ">"},
        count=1,
    )
    if not claimed_by_dead_consumer:
        raise RuntimeError("could not stage a pending Redis delivery")
    client.set(f"{cancel_prefix}:{pending_id}", "1")
    time.sleep(1.1)

    recovered_runtime = KnovaBrokerPlugin().create_runtime(runtime_config)
    try:
        recovered_runtime.start()
        if not recovered_runtime.run_once(timeout_ms=0):
            raise RuntimeError("recovered Consumer did not process pending delivery")
        pending_terminal, _ = _wait_terminal(
            client,
            stream=f"{event_prefix}:{pending_id}",
            timeout=30,
        )
        if pending_terminal.get("event_type") != "CANCELLED":
            raise RuntimeError("recovered pending task was not cancelled")
        if client.xpending(task_stream, group)["pending"] != 0:
            raise RuntimeError("recovered pending Redis delivery was not acknowledged")

        active_id = f"knova-active-recovery-{suffix}"
        active_request = _training_request(
            args,
            job_id=active_id,
            table=table,
            storage_root=storage_root,
            storage_prefix=f"{storage_prefix}/active",
            password=password,
        )
        active_request["algorithm"]["hyper_params"]["n_estimators"] = 100_000
        _submit(
            recovered_runtime,
            client,
            task_stream=task_stream,
            identity_field="job_id",
            operation_id=active_id,
            request=active_request,
        )
    finally:
        recovered_runtime.close()

    restarted_runtime = KnovaBrokerPlugin().create_runtime(runtime_config)
    try:
        restarted_runtime.start()
        recovered = restarted_runtime.active_submissions.get(active_id)
        if recovered is None:
            raise RuntimeError("running Ray job was not recovered after restart")
        client.set(f"{cancel_prefix}:{active_id}", "1")
        active_terminal, _ = _wait_terminal(
            client,
            stream=f"{event_prefix}:{active_id}",
            timeout=120,
        )
        if active_terminal.get("event_type") != "CANCELLED":
            raise RuntimeError("recovered active task was not cancelled")
    finally:
        restarted_runtime.close()
        client.delete(
            f"{cancel_prefix}:{pending_id}",
            f"{cancel_prefix}:{active_id}",
        )

    return {
        "pending_delivery": "recovered_and_cancelled",
        "active_ray_job": "recovered_and_cancelled",
    }


def main() -> int:
    args = _arguments()
    password = os.environ.get(args.clickhouse_password_env)
    if password is None:
        raise ValueError(f"{args.clickhouse_password_env} is required")
    repository = Path(__file__).resolve().parents[1]
    suffix = uuid.uuid4().hex[:10]
    input_table = f"tributo_knova_input_{suffix}"
    output_table = f"tributo_knova_output_{suffix}"
    job_id = f"knova-train-smoke-{suffix}"
    execution_id = f"knova-infer-smoke-{suffix}"
    storage_prefix = "bundle" if args.storage_mode == "nfs" else f"smoke/{suffix}"
    redis_client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    clickhouse = clickhouse_connect.get_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        username=args.clickhouse_user,
        password=password,
    )
    storage_root = tempfile.mkdtemp(prefix="tributo-knova-smoke-")
    runtime = None
    s3 = None
    try:
        if args.storage_mode == "s3":
            s3 = _s3_client()
            _ensure_s3_bucket(s3, args.s3_bucket)
        clickhouse.command("CREATE DATABASE IF NOT EXISTS knova")
        clickhouse.command(
            f"CREATE TABLE knova.{input_table} ("
            "entity_id UInt64, f1 Float32, f2 Float32, label UInt8, "
            "stat_month String) ENGINE=MergeTree ORDER BY entity_id"
        )
        clickhouse.command(
            f"INSERT INTO knova.{input_table} SELECT number, "
            "toFloat32(number % 17), toFloat32((number * 3) % 23), "
            "toUInt8((number % 17) + ((number * 3) % 23) > 18), "
            "if(number % 2 = 0, '202601', '202602') FROM numbers(300)"
        )
        clickhouse.command(
            f"CREATE TABLE knova.{output_table} ("
            "entity_id String, pred_label Nullable(String), "
            "pred_probability Nullable(Float64), "
            "pred_probabilities Nullable(String), pred_value Nullable(Float64), "
            "pred_extra String, execution_id String, model_id String, "
            "version_id String, tenant_id String, inferred_at DateTime64(6, 'UTC')) "
            "ENGINE=MergeTree ORDER BY (execution_id, entity_id)"
        )

        runtime_config = _runtime_config(args, repository)
        runtime = KnovaBrokerPlugin().create_runtime(runtime_config)
        runtime.start()
        training = _training_request(
            args,
            job_id=job_id,
            table=input_table,
            storage_root=storage_root,
            storage_prefix=storage_prefix,
            password=password,
        )
        _submit(
            runtime,
            redis_client,
            task_stream="knova:aimodel:training:distributed:tasks",
            identity_field="job_id",
            operation_id=job_id,
            request=training,
        )
        training_terminal, training_events = _wait_terminal(
            redis_client,
            stream=f"knova:aimodel:training:distributed:events:{job_id}",
            timeout=args.timeout,
        )
        _require_success(training_terminal, job_id)
        artifact_manifest = training_terminal["artifact_manifest"]
        bundle_uri = _bundle_uri(artifact_manifest)
        if training_terminal["result_summary"]["sample_rows"]["train"] != 300:
            raise RuntimeError("training terminal event has an invalid row count")
        formats = {
            alternative["format"]
            for alternative in artifact_manifest["model_artifacts"][
                "model_weights"
            ]["alternatives"]
        }
        if formats != {"onnx", "xgboost"}:
            raise RuntimeError("training artifact manifest is incomplete")
        checkpoint_artifacts = _checkpoint_artifacts(storage_root)
        if args.storage_mode == "s3":
            if not bundle_uri.startswith(f"s3://{args.s3_bucket}/"):
                raise RuntimeError("training did not publish the Bundle to S3")
            bundle_files = _s3_bundle_files(s3, bundle_uri)
        else:
            bundle_files = sorted(path.name for path in Path(bundle_uri).iterdir())

        inference = _inference_request(
            args,
            execution_id=execution_id,
            input_table=input_table,
            output_table=output_table,
            artifact_manifest=artifact_manifest,
            password=password,
            shap_mode=args.shap_mode,
        )
        _submit(
            runtime,
            redis_client,
            task_stream="knova:aimodel:inference:distributed:tasks",
            identity_field="execution_id",
            operation_id=execution_id,
            request=inference,
        )
        inference_terminal, inference_events = _wait_terminal(
            redis_client,
            stream=f"knova:aimodel:inference:distributed:events:{execution_id}",
            timeout=args.timeout,
        )
        _require_success(inference_terminal, execution_id)

        result_rows = clickhouse.query(
            f"SELECT count() FROM knova.{output_table} "
            "WHERE execution_id = {execution_id:String}",
            parameters={"execution_id": execution_id},
        ).first_row[0]
        expected_rows = 150
        if result_rows != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} ClickHouse rows, received {result_rows}"
            )
        if inference_terminal.get("result_rows") != expected_rows:
            raise RuntimeError(
                "terminal inference event does not contain the ClickHouse row count"
            )
        explanation_exactness = None
        if args.shap_mode != "none":
            pred_extra = clickhouse.query(
                f"SELECT pred_extra FROM knova.{output_table} "
                "WHERE execution_id = {execution_id:String} LIMIT 1",
                parameters={"execution_id": execution_id},
            ).first_row[0]
            explanation = json.loads(pred_extra).get("explanation")
            if not isinstance(explanation, dict):
                raise RuntimeError("ClickHouse result does not contain TreeSHAP")
            explanation_exactness = explanation.get("exactness")
            if explanation_exactness != args.shap_mode:
                raise RuntimeError(
                    "ClickHouse TreeSHAP exactness does not match the request"
                )
            if len(explanation.get("feature_contributions", ())) != 2:
                raise RuntimeError("ClickHouse TreeSHAP feature width is invalid")
        recovery = None
        if args.exercise_recovery:
            runtime.close()
            runtime = None
            recovery = _exercise_recovery(
                args,
                runtime_config=runtime_config,
                client=redis_client,
                table=input_table,
                storage_root=storage_root,
                storage_prefix=storage_prefix,
                password=password,
                suffix=suffix,
            )
        print(
            json.dumps(
                {
                    "status": "succeeded",
                    "training": {
                        "job_id": job_id,
                        "events": [event["event_type"] for event in training_events],
                        "bundle_files": bundle_files,
                        "checkpoint_artifacts": checkpoint_artifacts,
                        "storage_mode": args.storage_mode,
                    },
                    "inference": {
                        "execution_id": execution_id,
                        "events": [event["event_type"] for event in inference_events],
                        "clickhouse_rows": result_rows,
                        "reported_rows": inference_terminal["result_rows"],
                        "shap_exactness": explanation_exactness,
                    },
                    "recovery": recovery,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if runtime is not None:
            runtime.close()
        if not args.keep_tables:
            clickhouse.command(f"DROP TABLE IF EXISTS knova.{output_table}")
            clickhouse.command(f"DROP TABLE IF EXISTS knova.{input_table}")
        clickhouse.close()
        redis_client.close()
        if s3 is not None:
            _delete_s3_prefix(s3, args.s3_bucket, storage_prefix)
            s3.close()
        shutil.rmtree(storage_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
