import json
from typing import Any

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
