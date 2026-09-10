"""Minimal KnoVa-to-Tributo mapping for distributed XGBoost training."""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Mapping
from typing import Any, NoReturn
from urllib.parse import urlsplit

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
from tributo.exporting import BundleRef, load_bundle
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
_SENSITIVE_PROPERTY_TOKENS = frozenset(
    {"access", "credential", "key", "password", "secret", "token"}
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
    native_table = _native_clickhouse_table(
        payload, properties, (*feature_names, label_name)
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


def _native_clickhouse_table(
    payload: Mapping[str, Any],
    properties: Mapping[str, Any],
    required_columns: tuple[str, ...],
) -> str:
    """Resolve KnoVa's simple DIRECT_QUERY shape to a native Ray table read."""
    configured = properties.get("native_table")
    if configured is not None:
        return _text(configured, "datasource.properties.native_table")

    tables = payload.get("tables")
    data_query = payload.get("data_query")
    if not isinstance(tables, list) or len(tables) != 1 or not isinstance(
        data_query, Mapping
    ):
        _invalid(
            "datasource.properties.native_table is required unless data_query "
            "is a simple single-table projection"
        )

    table = _mapping(tables[0], "tables[0]")
    database = _text(table.get("database_name"), "tables[0].database_name")
    table_name = _text(table.get("table_name"), "tables[0].table_name")
    query = _mapping(data_query.get("query"), "data_query.query")
    sql = _text(query.get("sql"), "data_query.query.sql")
    params = _mapping(query.get("params", {}), "data_query.query.params")
    if params:
        _invalid(
            "datasource.properties.native_table is required for parameterized queries"
        )

    identifier = r"`?[A-Za-z_][A-Za-z0-9_]*`?"
    match = re.fullmatch(
        rf"(?is)\s*select\s+(.+?)\s+from\s+"
        rf"({identifier}(?:\.{identifier})?)"
        rf"(?:\s+(?:as\s+)?({identifier}))?\s*;?\s*",
        sql,
    )
    source_name = match.group(2).replace("`", "") if match else None
    qualified = f"{database}.{table_name}"
    if source_name not in {table_name, qualified}:
        _invalid(
            "datasource.properties.native_table is required unless data_query "
            "is a simple single-table projection"
        )

    assert match is not None
    table_alias = match.group(3)
    if table_alias is not None:
        table_alias = table_alias.replace("`", "")
    projected_columns: list[str] = []
    for item in match.group(1).split(","):
        projection = re.fullmatch(
            rf"(?is)\s*(?:({identifier})\.)?({identifier})\s+as\s+"
            rf"{identifier}\s*",
            item,
        )
        if projection is None:
            _invalid("data_query must contain plain column projections only")
        projection_alias = projection.group(1)
        if projection_alias is not None:
            projection_alias = projection_alias.replace("`", "")
        if table_alias is not None and projection_alias not in {None, table_alias}:
            _invalid("data_query projection alias does not match its source table")
        projected_columns.append(projection.group(2).replace("`", ""))
    if tuple(projected_columns) != required_columns:
        _invalid("data_query columns do not match the requested features and target")
    return qualified


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


def _safe_storage_properties(value: object) -> dict[str, str]:
    """Keep protocol storage hints while excluding credentials and Ray-only paths."""
    properties = _mapping(value, "storage_context.properties")
    safe: dict[str, str] = {}
    for raw_name, raw_value in properties.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            continue
        name = raw_name.lower()
        if name == "ray_storage_path" or any(
            token in name for token in _SENSITIVE_PROPERTY_TOKENS
        ):
            continue
        safe[raw_name] = raw_value
    return safe


def _artifact_storage(
    request: TrainingExecutionRequest,
    canonical_uri: str,
) -> dict[str, Any]:
    payload = request.model_dump(mode="python")
    requested = _mapping(payload.get("storage_context"), "storage_context")
    requested_type = _text(requested.get("type"), "storage_context.type").lower()
    requested_bucket = _text(requested.get("bucket"), "storage_context.bucket")
    requested_prefix = _relative_prefix(requested.get("prefix"))
    properties = _safe_storage_properties(requested.get("properties", {}))

    if requested_type == "s3":
        parsed = urlsplit(canonical_uri)
        if parsed.scheme != "s3" or parsed.netloc != requested_bucket:
            raise _TrainingExecutionError(
                "published Bundle is outside the requested S3 storage"
            )
        prefix = posixpath.normpath(parsed.path.strip("/"))
        try:
            contained = posixpath.commonpath([requested_prefix, prefix])
        except ValueError as exc:
            raise _TrainingExecutionError(
                "published Bundle is outside the requested S3 storage"
            ) from exc
        if contained != requested_prefix or prefix == requested_prefix:
            raise _TrainingExecutionError(
                "published Bundle is outside the requested S3 storage"
            )
        return {
            "type": "s3",
            "bucket": requested_bucket,
            "prefix": f"{prefix}/",
            "properties": properties,
        }

    if requested_type in _SHARED_STORAGE_TYPES:
        root = posixpath.normpath(requested_bucket)
        requested_root = posixpath.join(root, requested_prefix)
        bundle_path = posixpath.normpath(canonical_uri)
        try:
            contained = posixpath.commonpath([requested_root, bundle_path])
        except ValueError as exc:
            raise _TrainingExecutionError(
                "published Bundle is outside the requested shared storage"
            ) from exc
        if contained != requested_root or bundle_path == requested_root:
            raise _TrainingExecutionError(
                "published Bundle is outside the requested shared storage"
            )
        return {
            "type": "nas",
            "bucket": root,
            "prefix": f"{posixpath.relpath(bundle_path, root).strip('/')}/",
            "properties": properties,
        }

    raise _TrainingExecutionError("published Bundle storage type is unsupported")


def _signature_names(signature: object, field: str) -> list[str]:
    if not isinstance(signature, Mapping):
        return []
    fields = signature.get(field.replace("_names", "_fields"), [])
    if isinstance(fields, list):
        names = [item.get("name") for item in fields if isinstance(item, Mapping)]
        if names and all(isinstance(name, str) and name for name in names):
            return list(names)
    names = signature.get(field, [])
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    return []


def _artifact_manifest(
    request: TrainingExecutionRequest,
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_id = _text(outputs.get("bundle_id"), "training output bundle_id")
    bundle_uri = _text(outputs.get("bundle_uri"), "training output bundle_uri")
    manifest_sha256 = _text(
        outputs.get("manifest_sha256"), "training output manifest_sha256"
    )
    try:
        bundle = load_bundle(
            BundleRef(
                canonical_uri=bundle_uri,
                bundle_id=bundle_id,
                manifest_sha256=manifest_sha256,
            )
        )
    except Exception:
        raise _TrainingExecutionError(
            "published Bundle manifest validation failed"
        ) from None

    loaded_bundle_id = _text(bundle.get("bundle_id"), "Bundle bundle_id")
    if loaded_bundle_id != bundle_id:
        raise _TrainingExecutionError("published Bundle identity validation failed")
    canonical_uri = _text(bundle.get("canonical_uri"), "Bundle canonical_uri")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list):
        raise _TrainingExecutionError("published Bundle artifacts are invalid")

    input_names = _signature_names(bundle.get("input_signature"), "input_names")
    output_names = _signature_names(bundle.get("output_signature"), "output_names")
    alternatives: list[dict[str, Any]] = []
    total_size = 0
    seen_formats: set[str] = set()
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        bundle_format = raw_artifact.get("format")
        if bundle_format not in {"onnx", "ubj"}:
            continue
        protocol_format = "xgboost" if bundle_format == "ubj" else "onnx"
        if protocol_format in seen_formats:
            raise _TrainingExecutionError(
                "published Bundle has duplicate model artifact formats"
            )
        artifact_name = _text(raw_artifact.get("name"), "Bundle artifact name")
        raw_files = raw_artifact.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise _TrainingExecutionError("published Bundle artifact files are invalid")
        files: list[dict[str, Any]] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise _TrainingExecutionError(
                    "published Bundle artifact file is invalid"
                )
            relative_path = _text(
                raw_file.get("relative_path"), "Bundle artifact relative_path"
            )
            sha256 = _text(raw_file.get("sha256"), "Bundle artifact sha256")
            size = raw_file.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise _TrainingExecutionError(
                    "published Bundle artifact size is invalid"
                )
            metadata: dict[str, Any] = {}
            if protocol_format == "onnx":
                metadata = {
                    "input_names": input_names,
                    "output_names": output_names,
                }
            elif protocol_format == "xgboost":
                metadata = {"supports_tree_shap": True}
            files.append(
                {
                    "path": posixpath.join(
                        "artifacts", artifact_name, relative_path
                    ),
                    "size": size,
                    "hash": f"sha256:{sha256}",
                    "metadata": metadata,
                }
            )
            total_size += size
        alternatives.append({"format": protocol_format, "files": files})
        seen_formats.add(protocol_format)

    if "onnx" not in seen_formats or "xgboost" not in seen_formats:
        raise _TrainingExecutionError(
            "published Bundle must contain ONNX and XGBoost artifacts"
        )

    algorithm = _mapping(request.algorithm, "algorithm")
    return {
        "model_id": request.model_id,
        "version_id": request.version_id,
        "tenant_id": request.tenant_id,
        "created_at": bundle.get("created_at"),
        "algorithm_key": _text(
            algorithm.get("algorithm_key"), "algorithm.algorithm_key"
        ).lower(),
        "storage": _artifact_storage(request, canonical_uri),
        "model_artifacts": {
            "model_weights": {
                "comment": "Tributo XGBoost model Bundle",
                "required_for_inference": True,
                "alternatives": alternatives,
            }
        },
        "total_size_bytes": total_size,
    }


def _training_rows(result: AlgorithmRunResult) -> int:
    receipt = getattr(result, "execution_receipt", None)
    if receipt is None:
        return 0
    return sum(worker.rows_processed or 0 for worker in receipt.workers)


def _evaluation_metrics(result: AlgorithmRunResult) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    raw_metrics = getattr(result.execution, "metrics", {})
    if not isinstance(raw_metrics, Mapping):
        return metrics
    for name, value in raw_metrics.items():
        if (
            isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            metrics.append({"metric_name": name.lower(), "value": float(value)})
    return sorted(metrics, key=lambda item: item["metric_name"])


def _completed_payload(
    request: TrainingExecutionRequest,
    result: AlgorithmRunResult,
) -> dict[str, Any]:
    payload = request.model_dump(mode="python")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise _TrainingExecutionError("training features are required")
    target = _mapping(payload.get("target"), "target")
    task_type = _text(target.get("task_type"), "target.task_type").upper()
    rows = _training_rows(result)
    metrics = _evaluation_metrics(result)
    requested_evaluation = payload.get("evaluation")
    primary_name = None
    if isinstance(requested_evaluation, Mapping):
        raw_primary = requested_evaluation.get("primary_metric")
        if isinstance(raw_primary, str):
            primary_name = raw_primary.lower()
    primary_metric = next(
        (metric for metric in metrics if metric["metric_name"] == primary_name),
        None,
    )
    primary = (
        {
            "name": primary_metric["metric_name"],
            "value": primary_metric["value"],
        }
        if primary_metric is not None
        else None
    )

    model_features: list[dict[str, Any]] = []
    for index, raw_feature in enumerate(features):
        feature = _mapping(raw_feature, f"features[{index}]")
        origin = _mapping(feature.get("origin"), f"features[{index}].origin")
        model_features.append(
            {
                "model_feature_index": index,
                "model_feature_name": _text(
                    origin.get("column_name"),
                    f"features[{index}].origin.column_name",
                ),
                "feature_id": str(feature.get("feature_id") or f"f{index + 1:03d}"),
                "transformation": "PASSTHROUGH",
            }
        )

    algorithm = _mapping(request.algorithm, "algorithm")
    return {
        "result_summary": {
            "primary_metric": primary,
            "sample_rows": {
                "total": rows,
                "train": rows,
                "validation": 0,
                "test": 0,
            },
        },
        "training_result": {
            "algorithm_key": _text(
                algorithm.get("algorithm_key"), "algorithm.algorithm_key"
            ).lower(),
            "task_type": task_type,
            "model_features": model_features,
            "evaluation": {
                "eval_type": task_type,
                "sample_rows": 0,
                "metrics": metrics,
                "details": {},
            },
            "feature_analysis": None,
            "tuning_result": None,
        },
        "artifact_manifest": _artifact_manifest(
            request, dict(result.execution.outputs)
        ),
    }


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
        reporter.publish(
            "COMPLETED",
            _completed_payload(request, result),
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
