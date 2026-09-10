"""Minimal admission boundary for the existing KnoVa protocol v2."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from tributo_broker_redis.config import OperationType

PROTOCOL_VERSION = "2.0"


class KnovaProtocolFailure(ValueError):
    """Credential-safe rejection raised before broker admission."""

    def __init__(self, code: str, sanitized_message: str) -> None:
        super().__init__(sanitized_message)
        self.code = code
        self.sanitized_message = sanitized_message


class _Request(BaseModel):
    """Small common envelope; nested business fields remain protocol-owned."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        hide_input_in_errors=True,
    )

    protocol_version: str
    tenant_id: str = ""
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol_version")
    @classmethod
    def _supported_major(cls, value: str) -> str:
        parts = value.split(".", maxsplit=1)
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("protocol_version must use major.minor format")
        if parts[0] != PROTOCOL_VERSION.split(".", maxsplit=1)[0]:
            raise ValueError("unsupported protocol major version")
        return value


class TrainingExecutionRequest(_Request):
    """KnoVa training request with a unified Python-side name."""

    job_id: str = Field(min_length=1, max_length=256)
    algorithm: dict[str, Any]

    @field_validator("job_id")
    @classmethod
    def _trimmed_job_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("job_id must not have surrounding whitespace")
        return value

    @property
    def execution_id(self) -> str:
        """Return the canonical internal identity without changing the wire key."""
        return self.job_id

    @property
    def operation_type(self) -> Literal["training"]:
        return "training"


class InferenceExecutionRequest(_Request):
    """KnoVa batch-inference request."""

    execution_id: str = Field(min_length=1, max_length=256)
    model: dict[str, Any] | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    execution: dict[str, Any] = Field(default_factory=dict)

    @field_validator("execution_id")
    @classmethod
    def _trimmed_execution_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("execution_id must not have surrounding whitespace")
        return value

    @property
    def operation_type(self) -> Literal["batch_inference"]:
        return "batch_inference"


KnovaExecutionRequest = TrainingExecutionRequest | InferenceExecutionRequest


def parse_request(
    raw_payload: str,
    *,
    outer_operation_id: str,
    expected_operation_type: OperationType,
) -> KnovaExecutionRequest:
    """Parse one current KnoVa request and align its Redis envelope identity."""
    try:
        value = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise KnovaProtocolFailure(
            "INVALID_JSON", "payload is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise KnovaProtocolFailure(
            "INVALID_REQUEST", "payload root must be an object"
        )
    if "protocol_version" not in value:
        raise KnovaProtocolFailure(
            "UNSUPPORTED_PROTOCOL_VERSION", "protocol_version is required"
        )

    request_type: type[KnovaExecutionRequest]
    if expected_operation_type == "training":
        request_type = TrainingExecutionRequest
    else:
        request_type = InferenceExecutionRequest
    try:
        request = request_type.model_validate(value)
    except ValidationError as exc:
        version_errors = {
            tuple(error["loc"])
            for error in exc.errors(include_input=False)
            if tuple(error["loc"]) == ("protocol_version",)
        }
        code = (
            "UNSUPPORTED_PROTOCOL_VERSION"
            if version_errors
            else "INVALID_REQUEST"
        )
        raise KnovaProtocolFailure(code, "request schema validation failed") from exc

    if request.execution_id != outer_operation_id:
        raise KnovaProtocolFailure(
            "IDENTITY_MISMATCH",
            "outer operation identity does not match payload identity",
        )
    return request


__all__ = [
    "InferenceExecutionRequest",
    "KnovaExecutionRequest",
    "KnovaProtocolFailure",
    "PROTOCOL_VERSION",
    "TrainingExecutionRequest",
    "parse_request",
]

