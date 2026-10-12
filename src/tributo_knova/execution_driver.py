"""Single Ray Job driver for KnoVa training and batch inference."""

from __future__ import annotations

import base64
import logging
import os
import sys
from traceback import format_list, walk_tb
from typing import Any

from pydantic import ValidationError
from tributo_broker_redis.driver_transport import read_driver_input
from tributo_broker_redis.protocol import DriverInput
from tributo_broker_redis.redis_client import create_redis_client

from tributo_knova.protocol import (
    InferenceExecutionRequest,
    KnovaExecutionRequest,
    TrainingExecutionRequest,
)
from tributo_knova.reporter import ExecutionControlError, KnovaRedisEventReporter

_DRIVER_ENV = "TRIBUTO_REDIS_DRIVER_INPUT_B64"
# The entrypoint is invoked with ``python -m tributo_knova.execution_driver``.
# In that mode ``__name__`` is ``__main__``; using it here bypasses the
# ``tributo_knova`` handler and makes all driver diagnostics disappear from
# ``ray job logs``.  Keep a stable logger name for both import and -m modes.
logger = logging.getLogger("tributo_knova.execution_driver")


def _configure_driver_logging() -> None:
    """Ensure provider logs reach Ray Job stderr even when Ray owns logging."""
    selected = (
        os.environ.get("KNOVA_DRIVER_LOG_LEVEL")
        or os.environ.get("KNOVA_LOG_LEVEL")
        or "INFO"
    ).upper()
    level = getattr(logging, selected, logging.INFO)
    provider_logger = logging.getLogger("tributo_knova")
    provider_logger.setLevel(level)
    if not provider_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        provider_logger.addHandler(handler)
        provider_logger.propagate = False


_configure_driver_logging()
logger.info("KnoVa execution driver module loaded: path=%s", __file__)


class ExecutionNotImplemented(Exception):
    """The operation-specific KnoVa execution module is not available."""


class _TerminalEventPublicationError(Exception):
    """A completed execution whose terminal notification could not be emitted."""


def _failure_summary(error: Exception) -> str:
    """Expose only the exact provider-owned credential-safe exception types."""
    if isinstance(error, ExecutionControlError):
        return str(error)
    safe_types = {
        "tributo_knova.training": (
            "_TrainingConfigurationError",
            "_TrainingExecutionError",
        ),
        "tributo_knova.inference": (
            "_InferenceConfigurationError",
            "_InferenceExecutionError",
            "_EmptyInferenceInputError",
        ),
    }
    # The failing execution module is already loaded. Do not import another
    # execution stack while handling a dependency or runtime startup failure.
    for module_name, names in safe_types.items():
        module = sys.modules.get(module_name)
        if module is not None and any(
            type(error) is getattr(module, name, None) for name in names
        ):
            return str(error)[:1024]
    return f"{type(error).__name__}: operation failed; inspect Ray Job logs."


def _log_failure(message: str, error: Exception, *args: object) -> None:
    """Retain failure locations without rendering raw messages, chains or locals."""
    frames = [
        (frame.f_code.co_filename, lineno, frame.f_code.co_name, "")
        for frame, lineno in walk_tb(error.__traceback__)
    ]
    logger.error(
        message + " error_type=%s summary=%s\nTraceback (most recent call last):\n%s",
        *args,
        type(error).__name__,
        _failure_summary(error),
        "".join(format_list(frames)),
    )


def _failure_code(error: Exception) -> str:
    for module_name, names in (
        (
            "tributo_knova.inference",
            (
                "_EmptyInferenceInputError",
                "_InferenceConfigurationError",
                "_InferenceExecutionError",
            ),
        ),
        (
            "tributo_knova.training",
            ("_TrainingConfigurationError", "_TrainingExecutionError"),
        ),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in names:
            if type(error) is getattr(module, name, None):
                if name == "_EmptyInferenceInputError":
                    return "EMPTY_INPUT"
                return getattr(
                    error,
                    "code",
                    "EMPTY_INPUT"
                    if name == "_EmptyInferenceInputError"
                    else "INVALID_PAYLOAD"
                    if "Configuration" in name
                    else "EXECUTION_FAILED",
                )
    if isinstance(error, (ValueError, ValidationError)):
        return "INVALID_PAYLOAD"
    return "EXECUTION_FAILED"


def _failure_phase(error: Exception, reporter: KnovaRedisEventReporter) -> str:
    if type(error).__module__ in {"tributo_knova.inference", "tributo_knova.training"}:
        phase = getattr(error, "phase", None)
        if isinstance(phase, str):
            return phase
    return reporter._current_phase


def _load_driver_input() -> tuple[DriverInput, str]:
    raw = read_driver_input(os.environ)
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
    logger.info(
        "Driver input loaded: operation_id=%s operation_type=%s run_id=%s "
        "attempt_id=%s submission_id=%s encoded_input_bytes=%d",
        value.operation_id,
        value.operation_type,
        value.run_id,
        value.attempt_id,
        submission_id,
        len(raw.encode("ascii")),
    )
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
        received_at_ms=value.received_at_ms,
        deadline_at_ms=value.deadline_at_ms,
        cancel_key=value.cancel_key,
    )


def _initialize_ray() -> str:
    logger.info("Connecting to Ray cluster: address=auto")
    try:
        import ray
    except ImportError:
        raise RuntimeError("Ray runtime is unavailable") from None

    ray.init(address="auto", ignore_reinit_error=True)
    ray_job_id = ray.get_runtime_context().get_job_id()
    logger.info("Ray cluster connection established: ray_job_id=%s", ray_job_id)
    return ray_job_id


def _validate_request(value: DriverInput) -> KnovaExecutionRequest:
    logger.info(
        "Validating KnoVa request: operation_id=%s operation_type=%s",
        value.operation_id,
        value.operation_type,
    )
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
    if isinstance(request, TrainingExecutionRequest):
        features = request.model_dump(mode="python").get("features", [])
        logger.info(
            "KnoVa training request validated: operation_id=%s model_id=%s "
            "version_id=%s feature_count=%d",
            request.job_id,
            request.model_id,
            request.version_id,
            len(features) if isinstance(features, list) else 0,
        )
    else:
        logger.info(
            "KnoVa inference request validated: operation_id=%s task_id=%s",
            request.execution_id,
            request.task_id,
        )
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
    logger.info(
        "Starting KnoVa training execution: operation_id=%s model_id=%s version_id=%s",
        request.job_id,
        request.model_id,
        request.version_id,
    )
    try:
        result = execute_training(request, reporter)
        logger.info(
            "KnoVa training execution finished: operation_id=%s status=%s",
            request.job_id,
            getattr(getattr(result, "execution", None), "status", "unknown"),
        )
        return result
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
    logger.info(
        "Starting KnoVa inference execution: operation_id=%s task_id=%s",
        request.execution_id,
        request.task_id,
    )
    try:
        result = execute_inference(request, reporter)
        logger.info(
            "KnoVa inference execution finished: operation_id=%s status=%s",
            request.execution_id,
            getattr(result, "status", "unknown"),
        )
        return result
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
    try:
        value, submission_id = _load_driver_input()
    except Exception as exc:
        _log_failure("Could not load Ray driver input", exc)
        return 1
    try:
        redis_client = create_redis_client(value.redis_connection)
    except Exception as exc:
        _log_failure(
            "Could not initialize Redis reporter connection: operation_id=%s "
            "operation_type=%s submission_id=%s",
            exc,
            value.operation_id,
            value.operation_type,
            submission_id,
        )
        return 1
    try:
        reporter = _reporter(value, redis_client, submission_id)
        try:
            reporter.check_control()
            ray_job_id = _initialize_ray()
            logger.info(
                "Ray driver runtime initialized: operation_id=%s operation_type=%s "
                "submission_id=%s ray_job_id=%s",
                value.operation_id,
                value.operation_type,
                submission_id,
                (
                    ray_job_id
                    or os.environ.get("RAY_JOB_ID")
                    or value.ray_job_id
                    or "<unknown>"
                ),
            )
            request = _validate_request(value)
            reporter.check_control()
            _dispatch(value, request, reporter)
            logger.info(
                "Ray driver completed successfully: operation_id=%s "
                "operation_type=%s submission_id=%s",
                value.operation_id,
                value.operation_type,
                submission_id,
            )
            return 0
        except ExecutionControlError as exc:
            if exc.code == "CANCELLED":
                reporter.publish("CANCELLED", {"message": str(exc)}, phase=exc.phase)
            else:
                reporter.failed(
                    exc.code, type(exc).__name__, exc.phase, sanitized_message=str(exc)
                )
            return 1
        except (ExecutionNotImplemented, ImportError, NotImplementedError) as exc:
            _log_failure(
                "KnoVa execution is not implemented: operation_id=%s operation_type=%s",
                exc,
                value.operation_id,
                value.operation_type,
            )
            reporter.failed(
                "EXECUTION_NOT_IMPLEMENTED",
                "ExecutionNotImplemented",
                "PREPARING",
                sanitized_message=(
                    "ExecutionNotImplemented: execution is unavailable; "
                    "inspect Ray Job logs."
                ),
            )
            logger.info("FAILED event published: operation_id=%s", value.operation_id)
            return 1
        except _TerminalEventPublicationError as exc:
            _log_failure(
                "Could not publish terminal event: operation_id=%s operation_type=%s",
                exc,
                value.operation_id,
                value.operation_type,
            )
            reporter.failed(
                "TERMINAL_EVENT_PUBLICATION_FAILED",
                "TerminalEventPublicationError",
                "PUBLISHING",
                sanitized_message=(
                    "TerminalEventPublicationError: completion notification failed; "
                    "inspect Ray Job logs."
                ),
            )
            logger.info("FAILED event published: operation_id=%s", value.operation_id)
            return 1
        except Exception as exc:
            _log_failure(
                "Ray driver failed: operation_id=%s operation_type=%s phase=EXECUTING",
                exc,
                value.operation_id,
                value.operation_type,
            )
            reporter.failed(
                _failure_code(exc),
                type(exc).__name__,
                _failure_phase(exc, reporter),
                sanitized_message=_failure_summary(exc),
            )
            logger.info("FAILED event published: operation_id=%s", value.operation_id)
            return 1
    finally:
        close = getattr(redis_client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ExecutionNotImplemented", "main"]
