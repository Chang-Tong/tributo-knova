"""Minimal KnoVa-to-Tributo mapping for distributed XGBoost training."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from typing import Any, NoReturn

from tributo.algorithms import (
    AlgorithmOperation,
    AlgorithmRequest,
    ExecutionProfile,
    ExecutionRequest,
    InputBinding,
    WorkerResources,
    build_algorithm_dispatcher,
)
from tributo.algorithms.api import AlgorithmRunResult
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.data import IngestionRequest, ProviderSourceConfig
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputInvocation,
)

from tributo_knova.protocol import TrainingExecutionRequest

_CLICKHOUSE_BINDING_ID = "tributo.knova.ray.clickhouse"
_INPUT_REFERENCE = "knova.training.input"
_SHARED_STORAGE_TYPES = frozenset({"nfs", "nas", "shared_fs"})
_CONTROL_HYPER_PARAMETERS = frozenset(
    {"n_estimators", "num_rounds", "objective", "num_class"}
)


class _TrainingConfigurationError(ValueError):
    """Credential-safe request mapping failure."""


class _TrainingExecutionError(RuntimeError):
    """Credential-safe execution failure."""


def _invalid(message: str) -> NoReturn:
    raise _TrainingConfigurationError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(f"{field} must be an object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _invalid(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0
    ):
        _invalid(f"{field} must be a positive number")
    return float(value)


def _relative_prefix(value: object) -> str:
    prefix = _text(value, "storage_context.prefix").strip("/")
    if not prefix or ".." in prefix.split("/"):
        _invalid("storage_context.prefix must be a safe relative path")
    return prefix


def _absolute_shared_path(value: object, field: str) -> str:
    raw = _text(value, field)
    normalized = posixpath.normpath(raw)
    if not raw.startswith("/") or normalized == "/" or ".." in raw.split("/"):
        _invalid(f"{field} must be an absolute shared path below root")
    return normalized


def _storage_config(payload: Mapping[str, Any]) -> tuple[str, str]:
    storage = _mapping(payload.get("storage_context"), "storage_context")
    storage_type = _text(storage.get("type"), "storage_context.type").lower()
    bucket = _text(storage.get("bucket"), "storage_context.bucket")
    prefix = _relative_prefix(storage.get("prefix"))
    properties = _mapping(storage.get("properties", {}), "storage_context.properties")

    if storage_type in _SHARED_STORAGE_TYPES:
        root = _absolute_shared_path(bucket, "storage_context.bucket")
        configured_ray_path = properties.get("ray_storage_path")
        ray_storage_path = (
            _absolute_shared_path(
                configured_ray_path,
                "storage_context.properties.ray_storage_path",
            )
            if configured_ray_path
            else root
        )
        return ray_storage_path, posixpath.join(root, prefix)

    if storage_type == "s3":
        if "://" in bucket or "/" in bucket.strip("/"):
            _invalid("storage_context.bucket must be an S3 bucket name")
        ray_storage_path = _absolute_shared_path(
            properties.get("ray_storage_path"),
            "storage_context.properties.ray_storage_path",
        )
        return ray_storage_path, f"s3://{bucket.strip('/')}/{prefix}"

    _invalid("storage_context.type must be nfs, nas, shared_fs, or s3")


def _columns(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    raw_features = payload.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        _invalid("features must be a non-empty array")

    feature_names: list[str] = []
    for index, raw_feature in enumerate(raw_features):
        feature = _mapping(raw_feature, f"features[{index}]")
        origin = _mapping(feature.get("origin"), f"features[{index}].origin")
        feature_names.append(
            _text(
                origin.get("column_name"),
                f"features[{index}].origin.column_name",
            )
        )

    target = _mapping(payload.get("target"), "target")
    target_origin = _mapping(target.get("origin"), "target.origin")
    label_name = _text(target_origin.get("column_name"), "target.origin.column_name")
    if len(set(feature_names)) != len(feature_names):
        _invalid("feature origin column names must be unique")
    if label_name in feature_names:
        _invalid("target origin column must not also be a feature column")
    return tuple(feature_names), label_name


def _objective(
    target: Mapping[str, Any], hyper_parameters: Mapping[str, Any]
) -> tuple[str, int | None]:
    task_type = _text(target.get("task_type"), "target.task_type").upper()
    configured = hyper_parameters.get("objective")

    if task_type == "BINARY_CLASSIFICATION":
        objective = configured or "binary:logistic"
        if not isinstance(objective, str) or not objective.startswith("binary:"):
            _invalid("binary classification requires a binary XGBoost objective")
        return objective, None

    if task_type == "MULTICLASS_CLASSIFICATION":
        objective = configured or "multi:softprob"
        if not isinstance(objective, str) or not objective.startswith("multi:"):
            _invalid("multiclass classification requires a multi XGBoost objective")
        raw_num_class = hyper_parameters.get("num_class")
        if raw_num_class is None:
            label_mapping = _mapping(
                target.get("label_mapping"), "target.label_mapping"
            )
            raw_num_class = len(set(label_mapping.values()))
        num_class = _positive_int(raw_num_class, "algorithm.hyper_params.num_class")
        if num_class < 2:
            _invalid("multiclass classification requires at least two classes")
        return objective, num_class

    if task_type == "REGRESSION":
        objective = configured or "reg:squarederror"
        if not isinstance(objective, str) or not objective.startswith("reg:"):
            _invalid("regression requires a regression XGBoost objective")
        return objective, None

    _invalid("target.task_type is not supported by XGBoost training")


def _algorithm_config(
    payload: Mapping[str, Any], label_name: str, ray_storage_path: str, bundle_uri: str
) -> tuple[dict[str, Any], int]:
    algorithm = _mapping(payload.get("algorithm"), "algorithm")
    algorithm_key = _text(
        algorithm.get("algorithm_key"), "algorithm.algorithm_key"
    ).lower()
    if algorithm_key != "xgboost":
        _invalid("only algorithm_key=xgboost is supported")

    hyper_parameters = _mapping(
        algorithm.get("hyper_params", {}), "algorithm.hyper_params"
    )
    target = _mapping(payload.get("target"), "target")
    objective, num_class = _objective(target, hyper_parameters)
    rounds = _positive_int(
        hyper_parameters.get("n_estimators", hyper_parameters.get("num_rounds", 100)),
        "algorithm.hyper_params.n_estimators",
    )
    model = {
        key: value
        for key, value in hyper_parameters.items()
        if key not in _CONTROL_HYPER_PARAMETERS
    }
    model["objective"] = objective
    if num_class is not None:
        model["num_class"] = num_class

    return (
        {
            "data": {"label_col": label_name},
            "model": model,
            "training": {"num_rounds": rounds},
            "ray": {"storage_path": ray_storage_path},
            "output": {"bundle_uri": bundle_uri},
        },
        rounds,
    )


def _runtime_config(payload: Mapping[str, Any]) -> tuple[int, WorkerResources]:
    extensions = _mapping(payload.get("extensions", {}), "extensions")
    tributo = _mapping(extensions.get("tributo", {}), "extensions.tributo")
    training_runtime = _mapping(
        tributo.get("training_runtime", {}),
        "extensions.tributo.training_runtime",
    )
    ray = _mapping(
        training_runtime.get("ray", {}),
        "extensions.tributo.training_runtime.ray",
    )
    num_workers = _positive_int(
        ray.get("num_workers", 2),
        "extensions.tributo.training_runtime.ray.num_workers",
    )
    if num_workers < 2:
        _invalid("XGBoost distributed training requires at least two workers")
    cpus_per_worker = _positive_number(
        ray.get("cpus_per_worker", 1),
        "extensions.tributo.training_runtime.ray.cpus_per_worker",
    )
    use_gpu = ray.get("use_gpu", False)
    if not isinstance(use_gpu, bool):
        _invalid("extensions.tributo.training_runtime.ray.use_gpu must be a boolean")
    return num_workers, WorkerResources(
        num_cpus=cpus_per_worker,
        num_gpus=1.0 if use_gpu else 0.0,
    )


def _ingestion_invocation(
    payload: Mapping[str, Any], feature_names: tuple[str, ...], label_name: str
) -> IngestionInputInvocation:
    datasource = _mapping(payload.get("datasource"), "datasource")
    datasource_type = _text(datasource.get("type"), "datasource.type").upper()
    if datasource_type != "CLICKHOUSE":
        _invalid("only datasource.type=CLICKHOUSE is supported")
    properties = _mapping(datasource.get("properties", {}), "datasource.properties")
    native_table = _text(
        properties.get("native_table"), "datasource.properties.native_table"
    )
    host = _text(datasource.get("host"), "datasource.host")
    database = _text(datasource.get("database_name"), "datasource.database_name")
    port = _positive_int(datasource.get("port", 8123), "datasource.port")
    if port > 65535:
        _invalid("datasource.port must be no greater than 65535")
    if any(character in host for character in "/@?#"):
        _invalid("datasource.host is invalid")
    uri_host = host if ":" not in host or host.startswith("[") else f"[{host}]"

    options: dict[str, Any] = {
        "table": native_table,
        "host": host,
        "port": port,
        "database": database,
        "columns": [*feature_names, label_name],
    }
    username = datasource.get("username")
    password = datasource.get("password")
    if username is not None:
        options["user"] = _text(username, "datasource.username")
    if password is not None:
        if not isinstance(password, str):
            _invalid("datasource.password must be a string")
        options["password"] = password

    source = ProviderSourceConfig(
        provider="tributo.clickhouse",
        uri=f"clickhouse://{uri_host}:{port}/{database}",
        options=options,
    )
    return IngestionInputInvocation(
        request=IngestionRequest(
            source=source,
            engine="ray",
            binding_id=_CLICKHOUSE_BINDING_ID,
        )
    )


def _build_execution(
    request: TrainingExecutionRequest,
) -> tuple[
    ExecutionRequest,
    InputExecutionContext,
    InputResolutionContext,
    int,
]:
    payload = request.model_dump(mode="python")
    feature_names, label_name = _columns(payload)
    ray_storage_path, bundle_uri = _storage_config(payload)
    algorithm_config, rounds = _algorithm_config(
        payload, label_name, ray_storage_path, bundle_uri
    )
    worker_count, resources = _runtime_config(payload)
    invocation = _ingestion_invocation(payload, feature_names, label_name)
    values = {_INPUT_REFERENCE: invocation}
    execution = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="xgboost",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference=_INPUT_REFERENCE,
                feature_names=feature_names,
                label_name=label_name,
            ),
            algorithm_config=algorithm_config,
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=worker_count,
        resources_per_worker=resources,
    )
    return (
        execution,
        InputExecutionContext(values=values),
        InputResolutionContext(values=values),
        rounds,
    )


def execute_training(
    request: TrainingExecutionRequest,
    reporter: Any,
) -> AlgorithmRunResult:
    """Execute one KnoVa request through public Tributo algorithm contracts."""
    try:
        execution, input_context, resolution_context, rounds = _build_execution(request)
    except _TrainingConfigurationError:
        raise
    except Exception:
        raise _TrainingConfigurationError("training request mapping failed") from None

    try:
        reporter.phase("PREPARING")
        reporter.publish(
            "METRICS",
            {
                "current_round": 0,
                "total_rounds": rounds,
                "progress_percent": 0.0,
                "metrics": [],
            },
            phase="EXECUTING",
        )
        result = build_algorithm_dispatcher().execute(
            execution,
            input_context,
            resolution_context=resolution_context,
        )
        if result.execution.status != "succeeded":
            raise _TrainingExecutionError("XGBoost training did not succeed")
        outputs = dict(result.execution.outputs)
        reporter.publish(
            "COMPLETED",
            {
                "result_summary": {
                    "algorithm_key": "xgboost",
                    "status": result.execution.status,
                    "bundle_uri": outputs.get("bundle_uri"),
                },
                "training_result": outputs,
            },
            phase="COMPLETED",
        )
        return result
    except _TrainingExecutionError:
        raise
    except Exception as exc:
        raise _TrainingExecutionError(
            f"XGBoost training execution failed ({type(exc).__name__})"
        ) from None


__all__ = ["execute_training"]
