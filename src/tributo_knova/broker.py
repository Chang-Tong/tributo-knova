"""Thin KnoVa protocol adapter for the public Redis broker runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from tributo.integrations.broker import BROKER_API_VERSION, BrokerPlugin, BrokerRuntime
from tributo_broker_redis.config import OperationType, normalize_config
from tributo_broker_redis.operations import MappingFailure, PreparedOperation
from tributo_broker_redis.plugin import RedisBrokerPlugin
from tributo_broker_redis.protocol import GenericRequest
from tributo_broker_redis.runtime import RedisBrokerRuntime

from tributo_knova.protocol import parse_request
from tributo_knova.reporter import KnovaRedisEventReporter

DRIVER_ENTRYPOINT = "python -m tributo_knova.execution_driver"


def _redis_broker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate only the provider identity to the imported Redis contract."""
    value = dict(config)
    broker_id = value.get("broker_id", "tributo-knova")
    if broker_id != "tributo-knova":
        raise ValueError("broker_id must be tributo-knova")
    value["broker_id"] = "tributo-redis"
    return value


def parse_broker_request(
    raw_payload: str,
    *,
    outer_operation_id: str,
    expected_operation_type: OperationType,
) -> GenericRequest:
    """Adapt a KnoVa v2 request to the broker's existing admission contract."""
    request = parse_request(
        raw_payload,
        outer_operation_id=outer_operation_id,
        expected_operation_type=expected_operation_type,
    )
    payload = request.model_dump(mode="json", exclude_none=True)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return GenericRequest(
        protocol_profile="tributo-generic-v1",
        protocol_version="1.0",
        operation_id=request.execution_id,
        operation_type=expected_operation_type,
        execution_profile="distributed",
        run_id=request.execution_id,
        request_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        spec={"knova_request": payload},
    )


def prepare_broker_operation(request: GenericRequest) -> PreparedOperation:
    """Preserve the validated KnoVa request as the Ray driver payload."""
    payload = request.spec.get("knova_request")
    if not isinstance(payload, dict):
        raise MappingFailure(
            "INVALID_REQUEST",
            "validated KnoVa request is missing from broker input",
        )
    return PreparedOperation(
        operation_payload={"knova_request": dict(payload)},
        credential_ref=None,
    )


class KnovaBrokerPlugin(BrokerPlugin):
    """KnoVa v2 configuration of the existing Redis Streams provider."""

    api_version = BROKER_API_VERSION
    broker_id = "tributo-knova"
    stability = "alpha"
    capabilities = RedisBrokerPlugin.capabilities.difference(
        {
            "operation.training.single_worker",
            "operation.batch_inference.single_worker",
        }
    )

    def validate_config(
        self,
        config: Mapping[str, Any],
        *,
        check_connectivity: bool = False,
    ) -> None:
        RedisBrokerPlugin().validate_config(
            _redis_broker_config(config),
            check_connectivity=check_connectivity,
        )

    def create_runtime(self, config: Mapping[str, Any]) -> BrokerRuntime:
        return RedisBrokerRuntime(
            normalize_config(_redis_broker_config(config)),
            request_parser=parse_broker_request,
            operation_preparer=prepare_broker_operation,
            driver_entrypoint=DRIVER_ENTRYPOINT,
            reporter_factory=KnovaRedisEventReporter,
        )


__all__ = [
    "DRIVER_ENTRYPOINT",
    "KnovaBrokerPlugin",
    "parse_broker_request",
    "prepare_broker_operation",
]
