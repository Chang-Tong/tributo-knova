import base64
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tributo.ray_jobs import RayJobSubmission
from tributo_broker_redis.cancellation import (
    ActiveSubmission,
    ActiveSubmissionMap,
    CancelWatcher,
)
from tributo_broker_redis.config import RedisBrokerConfig
from tributo_broker_redis.operations import MappingFailure
from tributo_broker_redis.protocol import DriverInput, GenericRequest
from tributo_broker_redis.runtime import RedisBrokerRuntime

from tributo_knova.broker import (
    DRIVER_ENTRYPOINT,
    _matches_knova_terminal_event,
    _redis_broker_config,
    parse_broker_request,
    prepare_broker_operation,
)
from tributo_knova.reporter import KnovaRedisEventReporter


def _protocol_payload(kind: str = "training") -> dict[str, Any]:
    make = runpy.run_path(Path(__file__).with_name(f"test_{kind}.py"))["_request"]
    return make().model_dump(mode="json")


class _FakeRedis:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.delivered = False
        self.acked: list[tuple[str, str, str]] = []
        self.events: list[tuple[str, dict[str, str]]] = []
        self.cancelled: set[str] = set()
        self.claims = 0

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

    def xrevrange(self, stream: str, *, count: int) -> list[Any]:
        entries = [fields for name, fields in self.events if name == stream]
        return [("1-0", fields) for fields in reversed(entries[-count:])]

    def exists(self, _key: str) -> int:
        return int(_key in self.cancelled)

    def xautoclaim(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.claims += 1
        return ["0-0", []]

    def close(self) -> None:
        pass


@pytest.mark.parametrize("event_type", ["COMPLETED", "FAILED", "CANCELLED"])
def test_watcher_recognizes_knova_v2_terminal_without_submission_id(
    tmp_path: Path,
    event_type: str,
) -> None:
    config = _broker_config(tmp_path)
    redis = _FakeRedis("")
    item = ActiveSubmission(
        operation_id="job-terminal",
        operation_type="training",
        execution_profile="distributed",
        run_id="job-terminal",
        channel=config.channels.training,
        submission=RayJobSubmission(
            run_id="job-terminal",
            attempt_id="attempt-1",
            submission_id="submission-terminal",
        ),
    )
    active = ActiveSubmissionMap()
    active.put(item)
    redis.xadd(
        item.channel.event_stream_key(item.operation_id),
        {
            "job_id": item.operation_id,
            "payload": json.dumps(
                {
                    "protocol_version": "2.0",
                    "job_id": item.operation_id,
                    "event_type": event_type,
                }
            ),
        },
    )

    def unexpected_status(*args: Any, **kwargs: Any) -> str:
        pytest.fail("A terminal KnoVa operation must not query Ray or emit FAILED")

    watcher = CancelWatcher(
        redis,
        active,
        dashboard_url="http://ray:8265",
        interval_seconds=1,
        max_event_bytes=4096,
        max_stream_length=100,
        status_getter=unexpected_status,
        reporter_factory=KnovaRedisEventReporter,
        terminal_event_matcher=_matches_knova_terminal_event,
    )
    watcher.check_once()
    assert active.snapshot() == ()
    assert len(redis.events) == 1


def test_knova_terminal_matcher_rejects_other_protocols_and_identities(
    tmp_path: Path,
) -> None:
    config = _broker_config(tmp_path)
    item = ActiveSubmission(
        operation_id="job-terminal",
        operation_type="training",
        execution_profile="distributed",
        run_id="job-terminal",
        channel=config.channels.training,
        submission=RayJobSubmission(
            run_id="job-terminal",
            attempt_id="attempt-1",
            submission_id="submission-terminal",
        ),
    )
    event = {"protocol_version": "2.0", "job_id": "job-terminal"}
    assert _matches_knova_terminal_event(event, item)
    assert not _matches_knova_terminal_event({**event, "job_id": "other"}, item)
    assert not _matches_knova_terminal_event({**event, "protocol_version": "1.0"}, item)
    assert not _matches_knova_terminal_event(
        {**event, "submission_id": "another-submission"}, item
    )


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
    payload = _protocol_payload()

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
    assert (
        prepared.operation_payload["knova_request"]["datasource"]
        == payload["datasource"]
    )


def test_request_digest_ignores_json_formatting() -> None:
    payload = _protocol_payload("inference")
    payload["execution_id"] = "inference-1"

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
    request_payload = _protocol_payload()
    request_payload["resource_limits"] = {"max_training_time_seconds": 60}
    payload = json.dumps(request_payload)
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
    assert driver_input.received_at_ms is not None
    assert driver_input.deadline_at_ms == driver_input.received_at_ms + 60000
    assert driver_input.cancel_key == "knova:training:cancel:job-1"
    assert submissions[0][1]["metadata"]["tributo.received_at_ms"] == str(
        driver_input.received_at_ms
    )
    assert submissions[0][1]["metadata"]["tributo.deadline_at_ms"] == str(
        driver_input.deadline_at_ms
    )
    assert submissions[0][1]["metadata"]["tributo.run_id"] == "job-1"
    assert submissions[0][1]["metadata"]["tributo.attempt_id"] == "attempt-1"
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


def test_existing_broker_runtime_honors_queued_knova_cancellation(
    tmp_path: Path,
) -> None:
    core_root = tmp_path / "core"
    (core_root / "src" / "tributo").mkdir(parents=True)
    payload = json.dumps(
        {
            "protocol_version": "2.0",
            "job_id": "job-1",
            "model_id": "model-1",
            "version_id": "version-1",
            "tenant_id": "tenant-1",
            "algorithm": {"algorithm_key": "xgboost"},
        }
    )
    redis = _FakeRedis(payload)
    redis.cancelled.add("knova:training:cancel:job-1")
    submissions: list[str] = []
    runtime = RedisBrokerRuntime(
        _broker_config(core_root),
        redis_client=redis,
        submitter=lambda *_args, **_kwargs: submissions.append("submitted"),
        request_parser=parse_broker_request,
        operation_preparer=prepare_broker_operation,
        driver_entrypoint=DRIVER_ENTRYPOINT,
        reporter_factory=KnovaRedisEventReporter,
    )

    assert runtime.run_once(timeout_ms=0) is True

    assert submissions == []
    assert redis.acked == [("knova:training:tasks", "knova-training", "1-0")]
    event = json.loads(redis.events[-1][1]["payload"])
    assert event["event_type"] == "CANCELLED"
    assert event["job_id"] == "job-1"


def test_knova_config_identity_is_adapted_without_mutating_input() -> None:
    value = {"broker_id": "tributo-knova", "api_version": 1}

    adapted = _redis_broker_config(value)

    assert adapted == {"broker_id": "tributo-redis", "api_version": 1}
    assert value["broker_id"] == "tributo-knova"


def test_foreign_broker_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="broker_id must be tributo-knova"):
        _redis_broker_config({"broker_id": "other"})


def test_ray_failure_before_driver_emits_knova_terminal_event(tmp_path: Path) -> None:
    core_root = tmp_path / "core"
    (core_root / "src" / "tributo").mkdir(parents=True)
    config = _broker_config(core_root)
    redis = _FakeRedis("{}")
    submission = RayJobSubmission(
        run_id="job-before-driver",
        attempt_id="attempt-1",
        submission_id="submission-before-driver",
    )
    item = ActiveSubmission(
        operation_id="job-before-driver",
        operation_type="training",
        execution_profile="distributed",
        run_id=submission.run_id,
        channel=config.channels.training,
        submission=submission,
    )
    active = ActiveSubmissionMap()
    active.put(item)
    watcher = CancelWatcher(
        redis,
        active,
        dashboard_url="http://ray:8265",
        interval_seconds=1,
        max_event_bytes=1024 * 1024,
        max_stream_length=100,
        status_getter=lambda *_args, **_kwargs: "FAILED",
        reporter_factory=KnovaRedisEventReporter,
    )

    watcher.check_once()

    event = json.loads(redis.events[-1][1]["payload"])
    assert event["event_type"] == "FAILED"
    assert event["error_code"] == "TRAINING_FAILED"
    assert event["job_id"] == item.operation_id
    assert active.get(item.operation_id) is None


def test_broker_start_recovers_pending_and_running_knova_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_root = tmp_path / "core"
    (core_root / "src" / "tributo").mkdir(parents=True)
    redis = _FakeRedis("{}")
    running = SimpleNamespace(
        status="RUNNING",
        submission_id="submission-running",
        job_id="ray-job-running",
        metadata={
            "tributo.operation_id": "job-running",
            "tributo.operation_type": "training",
            "tributo.execution_profile": "distributed",
            "tributo.protocol_profile": "knova-v2",
            "tributo.run_id": "job-running",
            "tributo.attempt_id": "attempt-1",
            "tributo.driver_entrypoint": DRIVER_ENTRYPOINT,
            "tributo.task_stream": "knova:training:tasks",
            "tributo.request_digest": "a" * 64,
        },
    )
    runtime = RedisBrokerRuntime(
        _broker_config(core_root),
        redis_client=redis,
        request_parser=parse_broker_request,
        operation_preparer=prepare_broker_operation,
        driver_entrypoint=DRIVER_ENTRYPOINT,
        reporter_factory=KnovaRedisEventReporter,
        job_lister=lambda _url: [running],
    )
    monkeypatch.setattr(runtime._cancel_watcher, "start", lambda: None)

    runtime.start()
    runtime.start()

    assert redis.claims == 2
    recovered = runtime.active_submissions.get("job-running")
    assert recovered is not None
    assert recovered.submission.submission_id == "submission-running"
