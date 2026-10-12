"""Run inside the image against isolated real Ray/Redis; no mocked execution.

Use tests' isolated services only. REDIS_TEST_PASSWORD must be a disposable
test password. Artifacts are written to REPRO_OUTPUT (default /artifacts).
Modes: old-schema, new-schema, consumer-live, consumer-restart.
"""

import argparse
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path

import redis
from ray.job_submission import JobSubmissionClient
from tributo_broker_redis.protocol import DriverInput

from tributo_knova.broker import KnovaBrokerPlugin


def wait_for(predicate, description, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.5)
    raise AssertionError(f"Timed out: {description}")


def run(mode):
    logging.basicConfig(level=logging.INFO)
    operation = f"repro-{mode}-{uuid.uuid4().hex[:8]}"
    prefix = f"repro:{operation}"
    output = Path(os.environ.get("REPRO_OUTPUT", "/artifacts"))
    output.mkdir(parents=True, exist_ok=True)
    password = os.environ["REDIS_TEST_PASSWORD"]
    database = redis.Redis(
        host="repro-redis",
        port=6379,
        db=0,
        password=password,
        decode_responses=True,
    )
    jobs = JobSubmissionClient("http://repro-ray:8265")
    modules = [
        "/opt/tributo-runtime/core/tributo",
        "/opt/tributo-runtime/extensions/tributo_knova",
        "/opt/tributo-runtime/extensions/tributo_broker_redis",
        "/opt/tributo-runtime/extensions/tributo_algorithms_boosting",
    ]
    event_prefix = prefix + ":training:events"
    stream = event_prefix + ":" + operation

    def events():
        return [json.loads(fields["payload"]) for _, fields in database.xrange(stream)]

    def failures():
        return [event for event in events() if event["event_type"] == "FAILED"]

    runtime = None
    try:
        if mode.endswith("schema"):
            value = DriverInput(
                operation_id=operation,
                operation_type="training",
                execution_profile="distributed",
                run_id=operation,
                attempt_id="attempt-1",
                operation_payload={"knova_request": {}},
                redis_transport={
                    "mode": "standalone",
                    "url": "redis://repro-redis:6379/0",
                    "password_ref": "env:KNOVA_REDIS_PASSWORD",
                },
                event_stream_prefix=event_prefix,
                outer_identity_field="job_id",
                max_event_bytes=1048576,
                max_stream_length=100,
            )
            submission = operation
            jobs.submit_job(
                entrypoint="python -m tributo_knova.execution_driver",
                submission_id=submission,
                entrypoint_num_cpus=0,
                runtime_env={
                    "py_modules": modules,
                    "env_vars": {
                        "TRIBUTO_REDIS_DRIVER_INPUT_B64": base64.urlsafe_b64encode(
                            value.model_dump_json().encode()
                        ).decode(),
                        "TRIBUTO_RUN_ID": operation,
                        "TRIBUTO_ATTEMPT_ID": "attempt-1",
                        "TRIBUTO_SUBMISSION_ID": submission,
                    },
                },
            )
        else:
            # The Consumer has this reference; the Ray cluster deliberately does not.
            os.environ["KNOVA_TEST_DRIVER_REDIS_PASSWORD"] = password
            config = {
                "broker_id": "tributo-knova",
                "transport": {
                    "url": "redis://repro-redis:6379/0",
                    "password_ref": "env:KNOVA_TEST_DRIVER_REDIS_PASSWORD",
                    "block_ms": 100,
                },
                "channels": {
                    name: {
                        "task_stream_key": prefix + ":" + name + ":tasks",
                        "event_stream_prefix": prefix + ":" + name + ":events",
                        "cancel_key_prefix": prefix + ":" + name + ":cancel",
                        "consumer_group": operation + "-" + name,
                        "outer_identity_field": field,
                    }
                    for name, field in [
                        ("training", "job_id"),
                        ("batch_inference", "execution_id"),
                    ]
                },
                "execution": {
                    "ray_dashboard_url": "http://repro-ray:8265",
                    "project_root": "/opt/tributo-runtime/core",
                    "extra_py_modules": modules[1:],
                    "entrypoint_num_cpus": 0,
                    "cancel_poll_interval_seconds": 0.2,
                },
            }
            runtime = KnovaBrokerPlugin().create_runtime(config)
            if mode == "consumer-live":
                runtime.start()
            database.xadd(
                config["channels"]["training"]["task_stream_key"],
                {
                    "job_id": operation,
                    "payload": json.dumps(
                        {
                            "protocol_version": "2.0",
                            "job_id": operation,
                            "tenant_id": "local-repro",
                            "model_id": "model-test",
                            "version_id": "version-test",
                            "algorithm": {"algorithm_key": "xgboost"},
                        }
                    ),
                },
            )
            assert runtime.run_once(timeout_ms=100)
            active = runtime.active_submissions.get(operation)
            assert active is not None, events()
            submission = active.submission.submission_id

        def terminal_status():
            status = str(jobs.get_job_status(submission))
            return status if status in {"FAILED", "SUCCEEDED", "STOPPED"} else None

        status = wait_for(terminal_status, "Ray terminal status")
        assert status == "FAILED", status
        logs = jobs.get_job_logs(submission)
        (output / f"{mode}.ray.log").write_text(logs)
        if mode == "consumer-restart":
            assert failures() == [], "Driver must not have published without its secret"
            runtime.close()
            runtime = KnovaBrokerPlugin().create_runtime(config)
            runtime.start()

        terminal = wait_for(failures, "Redis FAILED event")
        assert len(terminal) == 1, terminal
        if mode == "old-schema":
            assert "Connected to Ray cluster" in logs
            assert "Driver input loaded:" not in logs
            assert "Traceback (most recent call last)" not in logs
        else:
            assert "INFO tributo_knova.execution_driver" in logs, logs
            assert "Driver input loaded:" in logs
            assert "Traceback (most recent call last)" in logs
            assert "runtime_resources/py_modules_files/" in logs, logs
        if mode.startswith("consumer"):
            assert terminal[0]["error_code"] == "RAY_JOB_FAILED", terminal
            assert "KNOVA_TEST_DRIVER_REDIS_PASSWORD" in logs
            wait_for(
                lambda: runtime.active_submissions.get(operation) is None,
                "consumer drops terminal job",
            )
            runtime.close()
            runtime = KnovaBrokerPlugin().create_runtime(config)
            runtime.start()
            assert runtime.active_submissions.get(operation) is None
            assert len(failures()) == 1, "Restart duplicated terminal event"
        else:
            assert terminal[0]["error_code"] == "EXECUTION_FAILED"
        result = {
            "mode": mode,
            "operation_id": operation,
            "submission_id": submission,
            "ray_status": status,
            "redis_failed_count": len(failures()),
            "error_code": terminal[0]["error_code"],
            "passed": True,
        }
        (output / f"{mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        if runtime is not None:
            runtime.close()
        database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "old-schema",
            "new-schema",
            "consumer-live",
            "consumer-restart",
        ],
    )
    run(parser.parse_args().mode)
