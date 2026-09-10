from __future__ import annotations

import base64
import json
import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from tributo_broker_redis.protocol import DriverInput

from tributo_knova import execution_driver
from tributo_knova.protocol import (
    InferenceExecutionRequest,
    TrainingExecutionRequest,
)
from tributo_knova.reporter import KnovaRedisEventReporter

_REQUEST_SECRET = "request-secret-must-not-leak"


class _Redis:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, str]]] = []
        self.closed = False

    def xadd(
        self,
        stream: str,
        fields: dict[str, str],
        **_kwargs: Any,
    ) -> str:
        self.entries.append((stream, fields))
        return "1-0"

    def close(self) -> None:
        self.closed = True


def _driver_input(operation_type: str = "training") -> DriverInput:
    if operation_type == "training":
        identity = "job-1"
        request: dict[str, object] = {
            "protocol_version": "2.0",
            "job_id": identity,
            "model_id": "model-1",
            "version_id": "version-1",
            "tenant_id": "tenant-1",
            "algorithm": {
                "algorithm_key": "xgboost",
                "api_token": _REQUEST_SECRET,
            },
        }
        outer_identity_field = "job_id"
        event_stream_prefix = "events:training"
    else:
        identity = "inference-1"
        request = {
            "protocol_version": "2.0",
            "execution_id": identity,
            "task_id": "task-1",
            "tenant_id": "tenant-1",
            "input": {"password": _REQUEST_SECRET},
        }
        outer_identity_field = "execution_id"
        event_stream_prefix = "events:inference"
    return DriverInput(
        operation_id=identity,
        operation_type=operation_type,
        execution_profile="distributed",
        run_id=identity,
        attempt_id="attempt-1",
        operation_payload={"knova_request": request},
        redis_url="redis://redis:6379/0",
        event_stream_prefix=event_stream_prefix,
        outer_identity_field=outer_identity_field,
        max_event_bytes=4096,
        max_stream_length=100,
    )


def _set_driver_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: DriverInput,
) -> None:
    encoded = base64.urlsafe_b64encode(
        value.model_dump_json().encode("utf-8")
    ).decode("ascii")
    monkeypatch.setenv("TRIBUTO_REDIS_DRIVER_INPUT_B64", encoded)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", "submission-1")
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", value.attempt_id)
    monkeypatch.setenv("TRIBUTO_RUN_ID", value.run_id)


def _event(redis: _Redis) -> dict[str, Any]:
    return json.loads(redis.entries[-1][1]["payload"])


def test_load_driver_input_validates_broker_runtime_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _driver_input()
    _set_driver_environment(monkeypatch, value)

    loaded, submission_id = execution_driver._load_driver_input()

    assert loaded == value
    assert submission_id == "submission-1"


@pytest.mark.parametrize(
    ("name", "env_value", "message"),
    [
        (
            "TRIBUTO_SUBMISSION_ID",
            None,
            "driver submission identity is required",
        ),
        ("TRIBUTO_ATTEMPT_ID", "attempt-other", "driver attempt identity mismatch"),
        ("TRIBUTO_RUN_ID", "run-other", "driver run identity mismatch"),
    ],
)
def test_load_driver_input_rejects_identity_mismatch_without_payload_leak(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    env_value: str | None,
    message: str,
) -> None:
    value = _driver_input()
    _set_driver_environment(monkeypatch, value)
    if env_value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, env_value)

    with pytest.raises(ValueError, match=message) as captured:
        execution_driver._load_driver_input()

    assert _REQUEST_SECRET not in str(captured.value)


def test_initialize_ray_uses_existing_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=lambda **kwargs: calls.append(kwargs)),
    )

    execution_driver._initialize_ray()

    assert calls == [{"address": "auto", "ignore_reinit_error": True}]


@pytest.mark.parametrize("operation_type", ["training", "batch_inference"])
def test_main_validates_and_dispatches_the_matching_request_type(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: str,
) -> None:
    value = _driver_input(operation_type)
    redis = _Redis()
    ray_calls: list[None] = []
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        execution_driver,
        "_load_driver_input",
        lambda: (value, "submission-1"),
    )
    monkeypatch.setattr(
        execution_driver,
        "create_redis_client",
        lambda _url: redis,
    )
    monkeypatch.setattr(
        execution_driver,
        "_initialize_ray",
        lambda: ray_calls.append(None),
    )
    monkeypatch.setattr(
        execution_driver,
        "_execute_training",
        lambda request, reporter: calls.append((request, reporter)),
    )
    monkeypatch.setattr(
        execution_driver,
        "_execute_inference",
        lambda request, reporter: calls.append((request, reporter)),
    )

    assert execution_driver.main() == 0
    assert ray_calls == [None]
    assert len(calls) == 1
    expected_type = (
        TrainingExecutionRequest
        if operation_type == "training"
        else InferenceExecutionRequest
    )
    assert isinstance(calls[0][0], expected_type)
    assert isinstance(calls[0][1], KnovaRedisEventReporter)
    assert redis.closed is True


def test_driver_rejects_payload_identity_mismatch_safely() -> None:
    value = _driver_input()
    payload = dict(value.operation_payload["knova_request"])
    payload["job_id"] = "job-other"
    mismatched = value.model_copy(
        update={"operation_payload": {"knova_request": payload}}
    )

    with pytest.raises(ValueError, match="request identity mismatch") as captured:
        execution_driver._validate_request(mismatched)

    assert _REQUEST_SECRET not in str(captured.value)


def test_unavailable_execution_publishes_sanitized_v2_failure_and_closes_redis(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    value = _driver_input()
    redis = _Redis()
    _set_driver_environment(monkeypatch, value)
    monkeypatch.setattr(
        execution_driver,
        "create_redis_client",
        lambda _url: redis,
    )
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(
        execution_driver,
        "_execute_training",
        lambda *_args: (_ for _ in ()).throw(
            execution_driver.ExecutionNotImplemented()
        ),
    )

    with caplog.at_level(logging.DEBUG):
        assert execution_driver.main() == 1

    event = _event(redis)
    assert event["protocol_version"] == "2.0"
    assert event["event_type"] == "FAILED"
    assert event["error_code"] == "EXECUTION_NOT_IMPLEMENTED"
    assert event["error_message"] == "operation failed"
    assert event["job_id"] == "job-1"
    assert _REQUEST_SECRET not in json.dumps(event)
    assert _REQUEST_SECRET not in caplog.text
    assert redis.closed is True


def test_import_error_from_dispatch_is_an_unimplemented_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _driver_input("batch_inference")
    redis = _Redis()
    _set_driver_environment(monkeypatch, value)
    monkeypatch.setattr(
        execution_driver,
        "create_redis_client",
        lambda _url: redis,
    )
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(
        execution_driver,
        "_execute_inference",
        lambda *_args: (_ for _ in ()).throw(ImportError(_REQUEST_SECRET)),
    )

    assert execution_driver.main() == 1

    event = _event(redis)
    assert event["error_code"] == "EXECUTION_NOT_IMPLEMENTED"
    assert _REQUEST_SECRET not in json.dumps(event)
    assert redis.closed is True


def test_terminal_publication_failure_preserves_broker_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _driver_input()
    redis = _Redis()
    _set_driver_environment(monkeypatch, value)
    monkeypatch.setattr(
        execution_driver,
        "create_redis_client",
        lambda _url: redis,
    )
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(
        execution_driver,
        "_execute_training",
        lambda *_args: (_ for _ in ()).throw(
            execution_driver._TerminalEventPublicationError()
        ),
    )

    assert execution_driver.main() == 1

    event = _event(redis)
    assert event["error_code"] == "TERMINAL_EVENT_PUBLICATION_FAILED"
    assert event["error_message"] == "operation failed"
    assert redis.closed is True
