"""Execute via stdin in the disposable Kind fixture; never a production command."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
from pathlib import Path


def main():
    if os.environ.get("KNOVA_LOCAL_ACCEPTANCE") != "1":
        raise RuntimeError("Only the explicit local fixture may execute this helper")
    import boto3
    import clickhouse_connect
    from botocore.config import Config
    from redis.sentinel import Sentinel

    client = clickhouse_connect.get_client(
        host="clickhouse",
        username="knova",
        password=os.environ["KNOVA_CLICKHOUSE_PASSWORD"],
    )
    s3 = boto3.client("s3", config=Config(s3={"addressing_style": "path"}))
    if sys.argv[1] == "seed":
        client.command("CREATE DATABASE IF NOT EXISTS knova")
        client.command(
            "CREATE TABLE IF NOT EXISTS knova.accept_input "
            "(entity_id UInt64, f1 Float32, f2 Float32, label UInt8, "
            "stat_month String) ENGINE=MergeTree ORDER BY entity_id"
        )
        if client.command("SELECT count() FROM knova.accept_input") == 0:
            client.command(
                "INSERT INTO knova.accept_input SELECT number, "
                "toFloat32(number % 17), toFloat32((number * 3) % 23), "
                "toUInt8((number % 17) + ((number * 3) % 23) > 18), "
                "if(number % 2 = 0, '202601', '202602') FROM numbers(1000)"
            )
        client.command(
            "CREATE TABLE IF NOT EXISTS knova.accept_output "
            "(entity_id String, pred_label Nullable(String), "
            "pred_probability Nullable(Float64), "
            "pred_probabilities Nullable(String), pred_value Nullable(Float64), "
            "pred_extra String, execution_id String, model_id String, "
            "version_id String, tenant_id String, inferred_at DateTime64(6)) "
            "ENGINE=MergeTree ORDER BY (execution_id, entity_id)"
        )
        buckets = [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]
        if "knova-acceptance" not in buckets:
            s3.create_bucket(Bucket="knova-acceptance")
        print(json.dumps({"seeded": True, "input_rows": 1000}))
        return

    config = json.loads(Path("/etc/tributo-knova/config.json").read_text())
    transport = config["transport"]
    sentinel = Sentinel(
        [
            (node.rsplit(":", 1)[0], int(node.rsplit(":", 1)[1]))
            for node in transport["sentinel_nodes"]
        ],
        sentinel_kwargs={"password": os.environ["KNOVA_REDIS_SENTINEL_PASSWORD"]},
        password=os.environ["KNOVA_REDIS_PASSWORD"],
        db=transport["database"],
        decode_responses=True,
    )
    redis = sentinel.master_for(transport["sentinel_master"])
    run_id = uuid.uuid4().hex[:10]
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "port": 8123,
        "database_name": "knova",
        "username": "knova",
        "credential_ref": "env://KNOVA_CLICKHOUSE_PASSWORD",
        "properties": {},
    }

    def submit(kind, request):
        channel = config["channels"][kind]
        identity = request[channel["outer_identity_field"]]
        redis.xadd(
            channel["task_stream_key"],
            {
                channel["outer_identity_field"]: identity,
                "payload": json.dumps(request),
            },
        )
        started = time.monotonic()
        stream = channel["event_stream_prefix"] + ":" + identity
        while time.monotonic() - started < 300:
            events = [
                json.loads(fields["payload"]) for _, fields in redis.xrange(stream)
            ]
            terminal = [
                event
                for event in events
                if event["event_type"]
                in {
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                }
            ]
            if terminal:
                if len(terminal) != 1 or terminal[0]["event_type"] != "COMPLETED":
                    # Only disposable fixture requests and events are used here.
                    print(json.dumps({"operation_id": identity, "events": events}))
                    raise RuntimeError("business operation did not complete")
                print(
                    json.dumps(
                        {
                            "operation_id": identity,
                            "seconds": round(time.monotonic() - started, 2),
                            "events": events,
                        }
                    ),
                    flush=True,
                )
                return terminal[0]
            time.sleep(0.5)
        raise TimeoutError(identity)

    job_id = "kind-training-" + run_id
    training = {
        "protocol_version": "2.0",
        "job_id": job_id,
        "model_id": "kind-existing-xgboost",
        "version_id": "v1",
        "tenant_id": "local-kind-only",
        "algorithm": {
            "algorithm_key": "xgboost",
            "hyper_params": {"n_estimators": 3, "max_depth": 3},
        },
        "datasource": {
            **datasource,
            "properties": {"native_table": "knova.accept_input"},
        },
        "features": [{"origin": {"column_name": name}} for name in ("f1", "f2")],
        "target": {
            "origin": {"column_name": "label"},
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"active": 0, "churn": 1},
        },
        "storage_context": {
            "type": "s3",
            "bucket": "knova-acceptance",
            "prefix": "kind-business/" + run_id,
            "properties": {},
        },
        "extensions": {
            "tributo": {
                "training_runtime": {"ray": {"num_workers": 2, "cpus_per_worker": 1}}
            }
        },
    }
    trained = submit("training", training)
    artifact = trained["artifact_manifest"]
    execution_id = "kind-inference-" + run_id
    inference = {
        "protocol_version": "2.0",
        "execution_id": execution_id,
        "task_id": "kind-business-task",
        "tenant_id": "local-kind-only",
        "model": {
            "model_id": "kind-existing-xgboost",
            "version_id": "v1",
            "algorithm_key": "xgboost",
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"active": 0, "churn": 1},
            "positive_label_value": "churn",
            "storage": artifact["storage"],
            "model_artifacts": artifact["model_artifacts"],
        },
        "input": {
            "datasource": datasource,
            "tables": [
                {
                    "table_alias": "t0",
                    "database_name": "knova",
                    "table_name": "accept_input",
                    "role": "PRIMARY",
                }
            ],
            "entity_key": {
                "origin": {"table_alias": "t0", "column_name": "entity_id"},
                "result_column": "entity_id",
            },
            "query": {
                "sql": (
                    "SELECT * FROM knova.accept_input t0 "
                    "WHERE t0.stat_month = {month:String}"
                ),
                "params": {"month": "202601"},
            },
            "features": [
                {
                    "origin": {"table_alias": "t0", "column_name": name},
                    "result_column": "t0__" + name,
                }
                for name in ("f1", "f2")
            ],
        },
        "output": {
            "datasource": datasource,
            "table_name": "accept_output",
            "result_filter": None,
        },
        "execution": {"batch_size": 64, "concurrency": 2},
        "extensions": {
            "explanation": {
                "enabled": True,
                "method": "TREE_SHAP",
                "approximate": False,
            }
        },
    }
    submit("batch_inference", inference)
    rows = client.query(
        "SELECT entity_id,pred_probability,pred_extra FROM knova.accept_output "
        "WHERE execution_id={id:String}",
        parameters={"id": execution_id},
    ).result_rows
    assert len(rows) == len({row[0] for row in rows}) == 500
    assert all(math.isfinite(row[1]) and 0 <= row[1] <= 1 for row in rows)
    errors = []
    for _, probability, extra in rows:
        explanation = json.loads(extra)["explanation"]
        assert explanation["exactness"] == "exact"
        assert explanation["output_space"] == "RAW_MARGIN"
        assert explanation["explained_class_label"] == "churn"
        contributions = explanation["feature_contributions"]
        assert len(contributions) == 2
        values = [item["shap_value"] for item in contributions]
        assert all(math.isfinite(value) for value in values)
        margin = explanation["base_value"] + sum(values)
        errors.append(abs(1 / (1 + math.exp(-margin)) - probability))
    assert max(errors) < 1e-5
    print(
        json.dumps(
            {
                "business_passed": True,
                "training_rows": 1000,
                "inference_rows": len(rows),
                "unique_entities": 500,
                "max_shap_probability_error": max(errors),
                "shap_example": json.loads(rows[0][2]),
                "job_id": job_id,
                "execution_id": execution_id,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
