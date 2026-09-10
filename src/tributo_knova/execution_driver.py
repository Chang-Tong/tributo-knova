"""Single Ray Job driver for KnoVa training and batch inference."""

from __future__ import annotations

import base64
import os
import sys
from typing import Any

from pydantic import ValidationError
from tributo_broker_redis.protocol import DriverInput
from tributo_broker_redis.redis_client import create_redis_client

from tributo_knova.protocol import (
    InferenceExecutionRequest,
    KnovaExecutionRequest,
    TrainingExecutionRequest,
)
from tributo_knova.reporter import KnovaRedisEventReporter

_DRIVER_ENV = "TRIBUTO_REDIS_DRIVER_INPUT_B64"


class ExecutionNotImplemented(Exception):
    """The operation-specific KnoVa execution module is not available."""


class _TerminalEventPublicationError(Exception):
    """A completed execution whose terminal notification could not be emitted."""


def _load_driver_input() -> tuple[DriverInput, str]:
    raw = os.environ.get(_DRIVER_ENV)
    if raw is None:
        raise ValueError(f"{_DRIVER_ENV} is required")
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")
    except Exception:
        raise ValueError("driver input is not valid base64 UTF-8") from None
    value = DriverInput.model_validate_json(decoded)
    submission_id = os.environ.get("TRIBUTO_SUBMISSION_ID")
    if not submission_id:
        raise ValueError("driver submission identity is required")
    if os.environ.get("TRIBUTO_ATTEMPT_ID") != value.attempt_id:
        raise ValueError("driver attempt identity mismatch")
    if os.environ.get("TRIBUTO_RUN_ID") != value.run_id:
        raise ValueError("driver run identity mismatch")
    return value, submission_id


def _reporter(
    value: DriverInput,
    redis_client: object,
    submission_id: str,
) -> KnovaRedisEventReporter:
    return KnovaRedisEventReporter(
        redis_client,
        event_stream_prefix=value.event_stream_prefix,
        operation_id=value.operation_id,
        operation_type=value.operation_type,
        execution_profile=value.execution_profile,
        run_id=value.run_id,
        attempt_id=value.attempt_id,
        submission_id=submission_id,
        ray_job_id=os.environ.get("RAY_JOB_ID") or value.ray_job_id,
        outer_identity_field=value.outer_identity_field,
        max_event_bytes=value.max_event_bytes,
        max_stream_length=value.max_stream_length,
    )


def _initialize_ray() -> None:
    try:
        import ray
    except ImportError:
        raise RuntimeError("Ray runtime is unavailable") from None

    ray.init(address="auto", ignore_reinit_error=True)


def _validate_request(value: DriverInput) -> KnovaExecutionRequest:
    payload = value.operation_payload.get("knova_request")
    request_type: type[KnovaExecutionRequest]
    if value.operation_type == "training":
        request_type = TrainingExecutionRequest
    else:
        request_type = InferenceExecutionRequest
    try:
        request = request_type.model_validate(payload)
    except ValidationError:
        raise ValueError("KnoVa request schema validation failed") from None
    if (
        request.execution_id != value.operation_id
        or request.execution_id != value.run_id
    ):
        raise ValueError("KnoVa driver request identity mismatch")
    return request


def _execute_training(
    request: TrainingExecutionRequest,
    reporter: KnovaRedisEventReporter,
) -> Any:
    try:
        from tributo_knova.training import execute_training
    except ImportError:
        raise ExecutionNotImplemented(
            "KnoVa training execution is unavailable"
        ) from None
    if not callable(execute_training):
        raise ExecutionNotImplemented("KnoVa training execution is unavailable")
    try:
        return execute_training(request, reporter)
    except (ImportError, NotImplementedError):
        raise ExecutionNotImplemented(
            "KnoVa training execution is unavailable"
        ) from None


def _execute_inference(
    request: InferenceExecutionRequest,
    reporter: KnovaRedisEventReporter,
) -> Any:
    try:
        from tributo_knova.inference import execute_inference
    except ImportError:
        raise ExecutionNotImplemented(
            "KnoVa inference execution is unavailable"
        ) from None
    if not callable(execute_inference):
        raise ExecutionNotImplemented("KnoVa inference execution is unavailable")
    try:
        return execute_inference(request, reporter)
    except (ImportError, NotImplementedError):
        raise ExecutionNotImplemented(
            "KnoVa inference execution is unavailable"
        ) from None


def _dispatch(
    value: DriverInput,
    request: KnovaExecutionRequest,
    reporter: KnovaRedisEventReporter,
) -> None:
    if value.operation_type == "training":
        if not isinstance(request, TrainingExecutionRequest):
            raise TypeError("training operation requires a training request")
        _execute_training(request, reporter)
        return
    if not isinstance(request, InferenceExecutionRequest):
        raise TypeError("batch inference operation requires an inference request")
    _execute_inference(request, reporter)


def main() -> int:
    value, submission_id = _load_driver_input()
    redis_client = create_redis_client(value.redis_url)
    try:
        reporter = _reporter(value, redis_client, submission_id)
        try:
            _initialize_ray()
            request = _validate_request(value)
            _dispatch(value, request, reporter)
            return 0
        except (ExecutionNotImplemented, ImportError, NotImplementedError):
            reporter.failed(
                "EXECUTION_NOT_IMPLEMENTED",
                "ExecutionNotImplemented",
                "PREPARING",
            )
            return 1
        except _TerminalEventPublicationError:
            reporter.failed(
                "TERMINAL_EVENT_PUBLICATION_FAILED",
                "TerminalEventPublicationError",
                "PUBLISHING",
            )
            return 1
        except Exception as exc:
            reporter.failed("EXECUTION_FAILED", type(exc).__name__, "EXECUTING")
            return 1
    finally:
        close = getattr(redis_client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ExecutionNotImplemented", "main"]
