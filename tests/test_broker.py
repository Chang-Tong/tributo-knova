import base64
import json
from pathlib import Path
from typing import Any

import pytest
from tributo.ray_jobs import RayJobSubmission
from tributo_broker_redis.config import RedisBrokerConfig
from tributo_broker_redis.operations import MappingFailure
from tributo_broker_redis.protocol import DriverInput, GenericRequest
from tributo_broker_redis.runtime import RedisBrokerRuntime

from tributo_knova.broker import (
    DRIVER_ENTRYPOINT,
    parse_broker_request,
    prepare_broker_operation,
)
from tributo_knova.reporter import KnovaRedisEventReporter


class _FakeRedis:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.delivered = False
        self.acked: list[tuple[str, str, str]] = []
        self.events: list[tuple[str, dict[str, str]]] = []

    def xgroup_create(self, **_kwargs: Any) -> bool:
        return True

    def xreadgroup(self, *, streams: dict[str, str], **_kwargs: Any) -> list[Any]:
        if self.delivered or next(iter(streams)) != "knova:training:tasks":
            return []
        self.delivered = True
        return [
            (
                "knova:training:tasks",
                [("1-0", {"job_id": "job-1", "payload": self.payload})],
            )
        ]

    def xack(self, stream: str, group: str, delivery: str) -> int:
        self.acked.append((stream, group, delivery))
        return 1

    def xadd(self, stream: str, fields: dict[str, str], **_kwargs: Any) -> str:
        self.events.append((stream, fields))
        return "1-0"

    def exists(self, _key: str) -> int:
        return 0

    def close(self) -> None:
        pass


def _broker_config(core_root: Path) -> RedisBrokerConfig:
    return RedisBrokerConfig.model_validate(
        {
            "transport": {
                "url": "redis://127.0.0.1:6379/0",
                "driver_url": "redis://redis:6379/0",
                "block_ms": 1,
            },
            "channels": {
                "training": {
                    "task_stream_key": "knova:training:tasks",
                    "event_stream_prefix": "knova:training:events",
                    "cancel_key_prefix": "knova:training:cancel",
                    "consumer_group": "knova-training",
                    "outer_identity_field": "job_id",
                },
                "batch_inference": {
                    "task_stream_key": "knova:inference:tasks",
                    "event_stream_prefix": "knova:inference:events",
                    "cancel_key_prefix": "knova:inference:cancel",
                    "consumer_group": "knova-inference",
                    "outer_identity_field": "execution_id",
                },
            },
            "execution": {
                "project_root": str(core_root),
                "runtime_pip_packages": ["tributo-knova==0.1.0"],
            },
        }
    )


def test_training_request_adapts_to_existing_broker_contract() -> None:
    payload = {
        "protocol_version": "2.0",
        "job_id": "job-1",
        "algorithm": {"algorithm_key": "xgboost"},
        "datasource": {"type": "CLICKHOUSE", "password": "runtime-secret"},
    }

    request = parse_broker_request(
        json.dumps(payload),
        outer_operation_id="job-1",
        expected_operation_type="training",
    )
    prepared = prepare_broker_operation(request)

    assert request.operation_id == "job-1"
    assert request.operation_type == "training"
    assert request.execution_profile == "distributed"
    assert request.run_id == "job-1"
    assert request.request_digest is not None
    assert len(request.request_digest) == 64
    assert prepared.operation_payload["knova_request"]["job_id"] == "job-1"
    assert prepared.operation_payload["knova_request"]["datasource"] == payload[
        "datasource"
    ]


def test_request_digest_ignores_json_formatting() -> None:
    payload = {
        "protocol_version": "2.0",
        "execution_id": "inference-1",
    }

    compact = parse_broker_request(
        json.dumps(payload, separators=(",", ":")),
        outer_operation_id="inference-1",
        expected_operation_type="batch_inference",
    )
    pretty = parse_broker_request(
        json.dumps(payload, indent=2),
        outer_operation_id="inference-1",
        expected_operation_type="batch_inference",
    )

    assert compact.request_digest == pretty.request_digest


def test_prepare_rejects_non_knova_generic_request() -> None:
    request = GenericRequest(
        protocol_profile="tributo-generic-v1",
        protocol_version="1.0",
        operation_id="job-1",
        operation_type="training",
        execution_profile="distributed",
        spec={},
    )

    with pytest.raises(MappingFailure) as captured:
        prepare_broker_operation(request)

    assert captured.value.code == "INVALID_REQUEST"


def test_existing_broker_runtime_submits_a_knova_driver(tmp_path: Path) -> None:
    core_root = tmp_path / "core"
    (core_root / "src" / "tributo").mkdir(parents=True)
    payload = json.dumps(
        {
            "protocol_version": "2.0",
            "job_id": "job-1",
            "algorithm": {"algorithm_key": "xgboost"},
        }
    )
    redis = _FakeRedis(payload)
    submissions: list[tuple[str, dict[str, Any]]] = []

    def submit(entrypoint: str, **kwargs: Any) -> RayJobSubmission:
        submissions.append((entrypoint, kwargs))
        return RayJobSubmission(
            run_id=kwargs["run_id"],
            attempt_id=kwargs["attempt_id"],
            submission_id="submission-1",
            ray_job_id="ray-job-1",
            request_digest=kwargs["request_digest"],
        )

    runtime = RedisBrokerRuntime(
        _broker_config(core_root),
        redis_client=redis,
        submitter=submit,
        request_parser=parse_broker_request,
        operation_preparer=prepare_broker_operation,
        driver_entrypoint=DRIVER_ENTRYPOINT,
        reporter_factory=KnovaRedisEventReporter,
    )

    assert runtime.run_once(timeout_ms=0) is True
    assert submissions[0][0] == DRIVER_ENTRYPOINT
    encoded = submissions[0][1]["env_vars"]["TRIBUTO_REDIS_DRIVER_INPUT_B64"]
    driver_input = DriverInput.model_validate_json(
        base64.urlsafe_b64decode(encoded).decode("utf-8")
    )
    assert driver_input.operation_payload["knova_request"]["job_id"] == "job-1"
    assert redis.acked == [("knova:training:tasks", "knova-training", "1-0")]
    admitted = json.loads(redis.events[-1][1]["payload"])
    assert admitted == {
        "event_type": "PHASE",
        "job_id": "job-1",
        "message": "Ray job admitted",
        "phase": "QUEUED",
        "protocol_version": "2.0",
        "timestamp": admitted["timestamp"],
    }
