import json
from typing import Any

import pytest

from tributo_knova.reporter import KnovaRedisEventReporter


class _Redis:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, str]]] = []

    def xadd(self, stream: str, fields: dict[str, str], **_kwargs: Any) -> str:
        self.entries.append((stream, fields))
        return "1-0"


def _event(redis: _Redis) -> dict[str, Any]:
    return json.loads(redis.entries[-1][1]["payload"])


def test_training_failure_uses_current_v2_envelope() -> None:
    redis = _Redis()
    reporter = KnovaRedisEventReporter(
        redis,
        event_stream_prefix="events:training",
        operation_id="job-1",
        operation_type="training",
        execution_profile="distributed",
        run_id="job-1",
        attempt_id="attempt-1",
        outer_identity_field="job_id",
    )

    reporter.publish(
        "FAILED",
        {"error_code": "INVALID_PAYLOAD", "sanitized_message": "invalid request"},
        phase="ADMISSION",
    )

    assert redis.entries[-1][0] == "events:training:job-1"
    assert redis.entries[-1][1]["job_id"] == "job-1"
    assert _event(redis) == {
        "duration_seconds": _event(redis)["duration_seconds"],
        "error_code": "INVALID_PAYLOAD",
        "error_message": "invalid request",
        "event_type": "FAILED",
        "job_id": "job-1",
        "phase": "QUEUED",
        "protocol_version": "2.0",
        "timestamp": _event(redis)["timestamp"],
    }


def test_training_metrics_callback_uses_current_v2_envelope() -> None:
    redis = _Redis()
    reporter = KnovaRedisEventReporter(
        redis,
        event_stream_prefix="events:training",
        operation_id="job-1",
        operation_type="training",
        execution_profile="distributed",
        run_id="job-1",
        attempt_id="attempt-1",
        outer_identity_field="job_id",
    )

    reporter.metrics(
        {
            "current_round": 2,
            "total_rounds": 10,
            "progress_percent": 20.0,
            "metrics": [{"metric_name": "logloss", "train": 0.4}],
        }
    )

    event = _event(redis)
    assert event["event_type"] == "METRICS"
    assert event["phase"] == "TRAINING"
    assert event["current_round"] == 2
    assert event["total_rounds"] == 10
    assert event["metrics"] == [{"metric_name": "logloss", "train": 0.4}]


def test_inference_admission_uses_current_v2_log_event() -> None:
    redis = _Redis()
    reporter = KnovaRedisEventReporter(
        redis,
        event_stream_prefix="events:inference",
        operation_id="execution-1",
        operation_type="batch_inference",
        execution_profile="distributed",
        run_id="execution-1",
        attempt_id="attempt-1",
        outer_identity_field="execution_id",
    )

    reporter.publish("ACCEPTED", phase="ADMITTED")

    assert redis.entries[-1][1]["execution_id"] == "execution-1"
    assert _event(redis)["event_type"] == "LOG"
    assert _event(redis)["execution_id"] == "execution-1"
    assert "protocol_profile" not in _event(redis)


@pytest.mark.parametrize("operation_type", ["training", "batch_inference"])
def test_failed_callback_preserves_approved_summary(operation_type: str) -> None:
    redis = _Redis()
    reporter = KnovaRedisEventReporter(
        redis,
        event_stream_prefix="events:test",
        operation_id="operation-1",
        operation_type=operation_type,
        execution_profile="distributed",
        run_id="operation-1",
        attempt_id="attempt-1",
    )

    reporter.failed(
        "EXECUTION_FAILED",
        "ConfigurationError",
        "PREPARING",
        sanitized_message="features must be a non-empty array",
    )

    assert _event(redis)["error_message"] == "features must be a non-empty array"


class _LifecycleRedis(_Redis):
    cancelled = False

    def xrevrange(self, stream: str, *, count: int) -> list[Any]:
        return [
            (str(i), fields)
            for i, (key, fields) in enumerate(reversed(self.entries))
            if key == stream
        ][:count]

    def exists(self, key: str) -> int:
        return int(self.cancelled)


def _reporter(redis: _Redis, **kwargs: Any) -> KnovaRedisEventReporter:
    return KnovaRedisEventReporter(
        redis,
        event_stream_prefix="events",
        operation_id="task",
        operation_type="batch_inference",
        execution_profile="distributed",
        run_id="task",
        attempt_id="attempt-1",
        **kwargs,
    )


def test_started_preserves_measured_total_rows() -> None:
    redis = _Redis()
    _reporter(redis).publish("STARTED", {"total_rows": 23}, phase="INIT")
    assert _event(redis)["event_type"] == "STARTED"
    assert _event(redis)["total_rows"] == 23


def test_late_admission_does_not_regress_driver_phase() -> None:
    redis = _LifecycleRedis()
    _reporter(redis).publish("PROGRESS", {"processed_rows": 8}, phase="WRITE_RESULT")
    _reporter(redis).publish("ACCEPTED", phase="ADMITTED")
    assert len(redis.entries) == 1
    assert _event(redis)["phase"] == "WRITE_RESULT"


def test_watcher_terminal_restores_progress_phase_and_receipt_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _LifecycleRedis()
    initial = _reporter(redis, received_at_ms=1000)
    initial.publish(
        "PROGRESS",
        {"processed_rows": 8, "result_rows": 4, "total_rows": 10, "percent": 80},
        phase="WRITE_RESULT",
    )
    initial.log("writing")
    monkeypatch.setattr("tributo_knova.reporter.time.time", lambda: 11.0)
    replacement = _reporter(redis, received_at_ms=1000)
    replacement.publish("CANCELLED", phase="CANCELLED")
    event = _event(redis)
    assert {
        key: event[key]
        for key in (
            "processed_rows",
            "result_rows",
            "total_rows",
            "percent",
            "phase",
            "duration_seconds",
        )
    } == {
        "processed_rows": 8,
        "result_rows": 4,
        "total_rows": 10,
        "percent": 80,
        "phase": "WRITE_RESULT",
        "duration_seconds": 10.0,
    }


def test_expired_execution_cannot_publish_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo_knova.reporter import ExecutionControlError

    redis = _LifecycleRedis()
    reporter = _reporter(redis, received_at_ms=1000, deadline_at_ms=2000)
    monkeypatch.setattr("tributo_knova.reporter.time.time", lambda: 3.0)
    with pytest.raises(ExecutionControlError) as caught:
        reporter.publish("COMPLETED", {})
    assert caught.value.code == "EXECUTION_TIMEOUT"
    assert redis.entries == []


def test_control_cancel_and_queued_phase_use_protocol_values() -> None:
    from tributo_knova.reporter import ExecutionControlError

    redis = _LifecycleRedis()
    redis.cancelled = True
    reporter = _reporter(redis, cancel_key="cancel:task")
    with pytest.raises(ExecutionControlError) as caught:
        reporter.check_control("QUEUED")
    assert caught.value.code == "CANCELLED"
    assert caught.value.phase == "INIT"
    reporter.publish("CANCELLED", phase="QUEUED")
    assert _event(redis)["phase"] == "INIT"
