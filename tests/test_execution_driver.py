from __future__ import annotations

import base64
import importlib
import json
import logging
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tributo_broker_redis.config import RedisConnectionConfig
from tributo_broker_redis.protocol import DriverInput

from tributo_knova import execution_driver
from tributo_knova.protocol import (
    InferenceExecutionRequest,
    TrainingExecutionRequest,
)
from tributo_knova.reporter import KnovaRedisEventReporter

_REQUEST_SECRET = "request-secret-must-not-leak"


def test_module_entrypoint_logs_missing_redis_secret_to_stderr() -> None:
    value = _driver_input().model_copy(
        update={
            "redis_url": None,
            "redis_transport": RedisConnectionConfig(
                mode="standalone",
                url="redis://127.0.0.1:6379/0",
                password_ref="env:KNOVA_TEST_MISSING_REDIS_PASSWORD",
            ),
        }
    )
    env = os.environ.copy()
    env.pop("KNOVA_TEST_MISSING_REDIS_PASSWORD", None)
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    env.update(
        {
            "KNOVA_DRIVER_LOG_LEVEL": "INFO",
            "TRIBUTO_REDIS_DRIVER_INPUT_B64": base64.urlsafe_b64encode(
                value.model_dump_json().encode()
            ).decode(),
            "TRIBUTO_RUN_ID": value.run_id,
            "TRIBUTO_ATTEMPT_ID": value.attempt_id,
            "TRIBUTO_SUBMISSION_ID": "module-entrypoint-test",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "tributo_knova.execution_driver"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "INFO tributo_knova.execution_driver" in result.stderr
    assert "Driver input loaded:" in result.stderr
    assert "Could not initialize Redis reporter connection" in result.stderr
    assert "RedisCredentialUnavailable" in result.stderr
    assert result.stderr.count("Traceback (most recent call last)") == 1
    assert _REQUEST_SECRET not in result.stderr


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
    encoded = base64.urlsafe_b64encode(value.model_dump_json().encode("utf-8")).decode(
        "ascii"
    )
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
        SimpleNamespace(
            init=lambda **kwargs: calls.append(kwargs),
            get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: "02000000"),
        ),
    )

    ray_job_id = execution_driver._initialize_ray()

    assert calls == [{"address": "auto", "ignore_reinit_error": True}]
    assert ray_job_id == "02000000"


def test_inference_finished_log_reads_top_level_result_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = SimpleNamespace(status="succeeded")
    monkeypatch.setitem(
        sys.modules,
        "tributo_knova.inference",
        SimpleNamespace(execute_inference=lambda *_args: result),
    )
    monkeypatch.setattr(execution_driver.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(execution_driver.logger, "propagate", False)
    request = InferenceExecutionRequest.model_validate(
        _driver_input("batch_inference").operation_payload["knova_request"]
    )

    with caplog.at_level(logging.INFO, logger=execution_driver.logger.name):
        actual = execution_driver._execute_inference(request, SimpleNamespace())

    assert actual is result
    assert "operation_id=inference-1 status=succeeded" in caplog.text
    assert "status=unknown" not in caplog.text
    assert _REQUEST_SECRET not in caplog.text


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


def test_main_uses_sentinel_driver_connection_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = RedisConnectionConfig(
        mode="sentinel",
        sentinel_nodes=("redis-sentinel:26379",),
        sentinel_master="mymaster",
        database=3,
        password_ref="env:KNOVA_REDIS_PASSWORD",
    )
    legacy = _driver_input()
    value = DriverInput.model_validate(
        {
            **legacy.model_dump(exclude={"redis_url", "redis_transport"}),
            "redis_transport": sentinel.model_dump(),
        }
    )
    redis = _Redis()
    connections: list[object] = []

    monkeypatch.setattr(
        execution_driver,
        "_load_driver_input",
        lambda: (value, "submission-1"),
    )
    monkeypatch.setattr(
        execution_driver,
        "create_redis_client",
        lambda connection: (connections.append(connection), redis)[1],
    )
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(execution_driver, "_execute_training", lambda *_args: None)

    assert execution_driver.main() == 0
    assert connections == [sentinel]
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
    assert event["error_code"] == "UNKNOWN"
    assert event["error_message"] == (
        "ExecutionNotImplemented: execution is unavailable; inspect Ray Job logs."
    )
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
    assert event["error_code"] == "UNKNOWN"
    assert _REQUEST_SECRET not in json.dumps(event)
    assert redis.closed is True


def test_terminal_publication_failure_maps_to_protocol_unknown(
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
    assert event["error_code"] == "UNKNOWN"
    assert event["error_message"] == (
        "TerminalEventPublicationError: completion notification failed; "
        "inspect Ray Job logs."
    )
    assert redis.closed is True


@pytest.mark.parametrize(
    ("operation_type", "module_name", "error_name", "message"),
    [
        (
            "training",
            "training",
            "_TrainingConfigurationError",
            "data_query.query.params is required for parameterized filters",
        ),
        (
            "training",
            "training",
            "_TrainingExecutionError",
            "training split row counts are inconsistent",
        ),
        (
            "batch_inference",
            "inference",
            "_InferenceConfigurationError",
            "input.tables must select exactly one native table",
        ),
        (
            "batch_inference",
            "inference",
            "_InferenceExecutionError",
            "input row count failed (ConnectionError)",
        ),
        (
            "batch_inference",
            "inference",
            "_EmptyInferenceInputError",
            "Inference input query returned zero rows",
        ),
    ],
)
def test_known_safe_failure_reaches_v2_event_and_log_without_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation_type: str,
    module_name: str,
    error_name: str,
    message: str,
) -> None:
    error_type = getattr(
        importlib.import_module(f"tributo_knova.{module_name}"), error_name
    )
    redis = _Redis()
    _set_driver_environment(monkeypatch, _driver_input(operation_type))
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(execution_driver.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(execution_driver.logger, "propagate", False)

    def fail(*_args: Any) -> None:
        try:
            raise RuntimeError(_REQUEST_SECRET)
        except RuntimeError as cause:
            error = error_type(message)
            error.add_note(_REQUEST_SECRET)
            raise error from cause

    monkeypatch.setattr(execution_driver, "_dispatch", fail)

    assert execution_driver.main() == 1
    event = _event(redis)
    assert event["event_type"] == "FAILED"
    assert (
        event["error_code"]
        == {
            "_EmptyInferenceInputError": "EMPTY_INPUT",
            "_InferenceConfigurationError": "INVALID_PAYLOAD",
            "_TrainingConfigurationError": "INVALID_PAYLOAD",
            "_TrainingExecutionError": "TRAINING_FAILED",
            "_InferenceExecutionError": "PREDICTION_FAILED",
        }[error_name]
    )
    assert event["error_message"] == message
    assert message in caplog.text
    assert "in fail" in caplog.text
    assert _REQUEST_SECRET not in json.dumps(event)
    assert _REQUEST_SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert redis.closed


@pytest.mark.parametrize("operation_type", ["training", "batch_inference"])
def test_unknown_failure_omits_message_notes_chain_and_source_from_event_and_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    operation_type: str,
) -> None:
    source = tmp_path / "credential_failure.py"
    source.write_text(
        "def fail(*args):\n"
        "    try:\n"
        f"        raise ValueError({_REQUEST_SECRET!r})\n"
        "    except ValueError as cause:\n"
        f"        error = RuntimeError({_REQUEST_SECRET!r})\n"
        f"        error.add_note({_REQUEST_SECRET!r})\n"
        "        raise error from cause\n",
        encoding="utf-8",
    )
    redis = _Redis()
    _set_driver_environment(monkeypatch, _driver_input(operation_type))
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(execution_driver, "_dispatch", runpy.run_path(source)["fail"])
    monkeypatch.setattr(execution_driver.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(execution_driver.logger, "propagate", False)

    assert execution_driver.main() == 1
    event = _event(redis)
    assert event["error_message"] == (
        "RuntimeError: operation failed; inspect Ray Job logs."
    )
    assert "credential_failure.py" in caplog.text
    assert "line 7, in fail" in caplog.text
    assert _REQUEST_SECRET not in json.dumps(event)
    assert _REQUEST_SECRET not in caplog.text
    assert "ValueError" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_failure_summary_rejects_spoofed_safe_class_and_never_formats_unknown() -> None:
    spoof = type(
        "_TrainingConfigurationError",
        (ValueError,),
        {"__module__": "tributo_knova.training"},
    )
    assert _REQUEST_SECRET not in execution_driver._failure_summary(
        spoof(_REQUEST_SECRET)
    )

    class UnformattableError(RuntimeError):
        def __str__(self) -> str:
            pytest.fail("unknown exceptions must not be converted to strings")

    assert execution_driver._failure_summary(UnformattableError()) == (
        "UnformattableError: operation failed; inspect Ray Job logs."
    )


@pytest.mark.parametrize(
    ("operation_type", "expected_message"),
    [
        ("training", "algorithm.category has an unsupported value"),
        ("batch_inference", "model must be an object"),
    ],
)
def test_real_invalid_request_reports_actionable_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: str,
    expected_message: str,
) -> None:
    redis = _Redis()
    _set_driver_environment(monkeypatch, _driver_input(operation_type))
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)

    assert execution_driver.main() == 1
    event = _event(redis)
    assert event["error_message"] == expected_message
    assert _REQUEST_SECRET not in json.dumps(event)


@pytest.mark.parametrize(
    ("balance", "expected_message"),
    [
        (
            {"strategy": _REQUEST_SECRET},
            "data_sampling.class_balance.strategy has an unsupported value",
        ),
        (
            {"strategy": "CUSTOM_AMOUNT", "class_amounts": {_REQUEST_SECRET: 0}},
            "data_sampling.class_balance.class_amounts value must be an integer >= 1",
        ),
    ],
)
def test_real_sampling_failure_does_not_reflect_values_into_v2_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    balance: dict[str, Any],
    expected_message: str,
) -> None:
    from tributo_knova import training

    # Reuse the full request fixture so the real mapping/driver/reporting path
    # is exercised without starting Ray, ClickHouse or a training workload.
    make_request = runpy.run_path(Path(__file__).with_name("test_training.py"))[
        "_request"
    ]
    request = make_request(data_sampling={"class_balance": balance})
    value = _driver_input().model_copy(
        update={"operation_payload": {"knova_request": request.model_dump(mode="json")}}
    )
    redis = _Redis()
    _set_driver_environment(monkeypatch, value)
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    monkeypatch.setattr(execution_driver.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(execution_driver.logger, "propagate", False)
    monkeypatch.setattr(
        training,
        "run_xgboost_training",
        lambda **_kwargs: pytest.fail(
            "invalid sampling must be rejected before training"
        ),
    )

    assert execution_driver.main() == 1
    event = _event(redis)
    assert event["error_message"] == expected_message
    assert _REQUEST_SECRET not in json.dumps(event)
    assert _REQUEST_SECRET not in caplog.text


@pytest.mark.parametrize(
    "operation_type,code",
    [("training", "TRAINING_TIMEOUT"), ("batch_inference", "EXECUTION_TIMEOUT")],
)
def test_expired_deadline_fails_before_ray_initialization(
    monkeypatch: pytest.MonkeyPatch, operation_type: str, code: str
) -> None:
    value = _driver_input(operation_type).model_copy(
        update={"received_at_ms": 1000, "deadline_at_ms": 2000}
    )
    redis = _Redis()
    _set_driver_environment(monkeypatch, value)
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(
        execution_driver,
        "_initialize_ray",
        lambda: pytest.fail("expired job must not initialize Ray"),
    )
    assert execution_driver.main() == 1
    assert _event(redis)["error_code"] == code
    assert redis.closed


def test_structured_inference_failure_keeps_code_phase_and_safe_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo_knova.inference import _InferenceExecutionError

    redis = _Redis()
    _set_driver_environment(monkeypatch, _driver_input("batch_inference"))
    monkeypatch.setattr(execution_driver, "create_redis_client", lambda _value: redis)
    monkeypatch.setattr(execution_driver, "_initialize_ray", lambda: None)
    error = _InferenceExecutionError(
        "result write failed", code="RESULT_WRITE_FAILED", phase="WRITE_RESULT"
    )
    monkeypatch.setattr(
        execution_driver, "_dispatch", lambda *a: (_ for _ in ()).throw(error)
    )
    assert execution_driver.main() == 1
    event = _event(redis)
    assert event["error_code"] == "RESULT_WRITE_FAILED"
    assert event["phase"] == "WRITE_RESULT"
    assert event["error_message"] == "result write failed"
