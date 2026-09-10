"""KnoVa protocol mapping for Tributo's public batch-inference runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from tributo.data import (
    DataWriteTargetRequest,
    FilterEq,
    IngestionRequest,
    ProviderSourceConfig,
    TransformPipeline,
)
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference import (
    ArtifactModelReference,
    BundleModelReference,
    InferenceRequest,
    InputBindingSpec,
    OutputBindingSpec,
    RayExecutionPolicy,
    TensorInputBinding,
    TensorOutputBinding,
    run_inference,
)
from tributo.inference.contracts import ResultSinkReceipt

from tributo_knova.clickhouse import _dsn, _qualified_table
from tributo_knova.protocol import InferenceExecutionRequest

_CLICKHOUSE_BINDING_ID = "tributo.knova.ray.clickhouse"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FROM = re.compile(
    r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_WHERE = re.compile(
    r"\bWHERE\s+(.+?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_EQUALITY = re.compile(
    r"(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*=\s*\{([A-Za-z_][A-Za-z0-9_]*):[A-Za-z0-9_(), ]+\}",
    re.IGNORECASE,
)
_UNSUPPORTED_SQL = re.compile(
    r";|--|/\*|\b(?:JOIN|UNION|GROUP\s+BY|HAVING|WITH)\b",
    re.IGNORECASE,
)
_SHARED_STORAGE_TYPES = frozenset({"nfs", "nas", "shared_fs"})
_MEMORY_PRESSURE_ERRORS = frozenset(
    {"MemoryError", "OutOfMemoryError", "RayActorError", "RayOutOfMemoryError"}
)
_MAX_ADAPTIVE_RETRIES = 3


class _InferenceConfigurationError(ValueError):
    """Credential-safe request mapping failure."""


class _InferenceExecutionError(RuntimeError):
    """Credential-safe runtime failure."""


def _invalid(message: str) -> NoReturn:
    raise _InferenceConfigurationError(message)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(f"{field} must be an object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{field} must be a non-empty string")
    return value.strip()


def _identifier(value: object, field: str) -> str:
    result = _text(value, field)
    if _IDENTIFIER.fullmatch(result) is None:
        _invalid(f"{field} must be a SQL identifier")
    return result


def _positive_int(value: object, field: str, default: int) -> int:
    raw = default if value is None else value
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        _invalid(f"{field} must be a positive integer")
    return raw


def _storage_root(storage: Mapping[str, Any]) -> str:
    storage_type = _text(storage.get("type"), "model.storage.type").lower()
    bucket = _text(storage.get("bucket"), "model.storage.bucket")
    prefix = str(storage.get("prefix") or "").strip("/")
    if ".." in prefix.split("/"):
        _invalid("model.storage.prefix must be a safe relative path")
    if storage_type in _SHARED_STORAGE_TYPES:
        if not bucket.startswith("/"):
            _invalid("shared model storage bucket must be an absolute path")
        return posixpath.join(posixpath.normpath(bucket), prefix)
    if storage_type == "s3":
        if "://" in bucket or "/" in bucket.strip("/"):
            _invalid("model.storage.bucket must be an S3 bucket name")
        return f"s3://{bucket.strip('/')}" + (f"/{prefix}" if prefix else "")
    _invalid("model.storage.type must be nfs, nas, shared_fs, or s3")


def _join_uri(root: str, path: str) -> str:
    relative = _text(path, "model artifact path").lstrip("/")
    if ".." in relative.split("/"):
        _invalid("model artifact path must stay below model storage")
    return root.rstrip("/") + "/" + relative


def _signature_fields(
    model: Mapping[str, Any],
    *,
    feature_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_type = str(model.get("task_type") or "").upper()
    class_count = max(
        2, len(_mapping(model.get("label_mapping", {}), "model.label_mapping"))
    )
    inputs = [
        {"name": "float_input", "dtype": "float32", "shape": ["batch", feature_count]}
    ]
    if task_type in {"BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"}:
        outputs = [
            {"name": "label", "dtype": "int64", "shape": ["batch"]},
            {
                "name": "probabilities",
                "dtype": "float32",
                "shape": ["batch", class_count],
            },
        ]
    elif task_type == "REGRESSION":
        outputs = [{"name": "prediction", "dtype": "float32", "shape": ["batch", 1]}]
    else:
        _invalid("model.task_type is not supported by XGBoost inference")
    return inputs, outputs


def _artifact_alternative(
    model: Mapping[str, Any],
    *,
    prefer_ubj: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    artifacts = _mapping(model.get("model_artifacts"), "model.model_artifacts")
    weights = _mapping(
        artifacts.get("model_weights"), "model.model_artifacts.model_weights"
    )
    alternatives = weights.get("alternatives")
    if not isinstance(alternatives, list):
        _invalid("model.model_artifacts.model_weights.alternatives must be an array")
    supported: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for raw in alternatives:
        if not isinstance(raw, Mapping):
            continue
        fmt = str(raw.get("format") or "").lower()
        files = raw.get("files")
        if fmt not in {"onnx", "xgboost", "ubj"} or not isinstance(files, list):
            continue
        expected_suffix = ".onnx" if fmt == "onnx" else ".ubj"
        weight_files = [
            file
            for file in files
            if isinstance(file, Mapping)
            and str(file.get("path") or "").lower().endswith(expected_suffix)
        ]
        if len(weight_files) == 1:
            supported.append((raw, weight_files[0]))
    if not supported:
        _invalid("model artifacts require one ONNX or UBJ weight file")
    preferred_formats = {"xgboost", "ubj"} if prefer_ubj else {"onnx"}
    return next(
        (
            item
            for item in supported
            if str(item[0].get("format") or "").lower() in preferred_formats
        ),
        supported[0],
    )


def _explanation_options(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    extensions = payload.get("extensions")
    raw = extensions.get("explanation") if isinstance(extensions, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _invalid("extensions.explanation must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        _invalid("extensions.explanation.enabled must be a boolean")
    if not enabled:
        return None
    if str(raw.get("method") or "TREE_SHAP").upper() != "TREE_SHAP":
        _invalid("extensions.explanation.method must be TREE_SHAP")
    approximate = raw.get("approximate", False)
    if not isinstance(approximate, bool):
        _invalid("extensions.explanation.approximate must be a boolean")
    model = _mapping(payload.get("model"), "model")
    if str(model.get("algorithm_key") or "").lower() != "xgboost":
        _invalid("TREE_SHAP requires model.algorithm_key=xgboost")
    return {"method": "TREE_SHAP", "approximate": approximate}


def _model_reference(
    model_value: Mapping[str, Any],
    *,
    execution_id: str,
    features: tuple[str, ...],
    prefer_native: bool = False,
) -> BundleModelReference | ArtifactModelReference:
    model = dict(model_value)
    explicit_bundle = model.get("bundle_uri")
    storage = _mapping(model.get("storage", {}), "model.storage")
    properties = _mapping(storage.get("properties", {}), "model.storage.properties")
    explicit_bundle = explicit_bundle or properties.get("bundle_uri")
    if explicit_bundle is not None:
        return BundleModelReference(
            uri=_text(explicit_bundle, "model.bundle_uri"),
            role="native" if prefer_native else "inference",
        )

    root = _storage_root(storage)
    if not root.startswith("s3://") and (Path(root) / "manifest.json").is_file():
        return BundleModelReference(
            uri=root,
            role="native" if prefer_native else "inference",
        )

    alternative, file = _artifact_alternative(model, prefer_ubj=prefer_native)
    protocol_format = _text(
        alternative.get("format"), "model artifact format"
    ).lower()
    format_id = "ubj" if protocol_format == "xgboost" else protocol_format
    metadata = _mapping(file.get("metadata", {}), "model artifact metadata")
    input_fields, output_fields = _signature_fields(
        model,
        feature_count=len(features),
    )
    input_names = metadata.get("input_names")
    output_names = metadata.get("output_names")
    if isinstance(input_names, list) and len(input_names) == 1:
        input_fields[0]["name"] = _text(input_names[0], "model input name")
    if isinstance(output_names, list) and len(output_names) == len(output_fields):
        for field, name in zip(output_fields, output_names, strict=True):
            field["name"] = _text(name, "model output name")
    raw_hash = file.get("hash")
    expected_hash = None
    if isinstance(raw_hash, str):
        expected_hash = raw_hash.removeprefix("sha256:")
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            _invalid("model artifact hash must be sha256 hex")
    import_root = properties.get("import_bundle_uri") or _join_uri(
        root, f".tributo-imports/{execution_id}"
    )
    return ArtifactModelReference(
        provider_id="tributo.artifact",
        uri=_join_uri(root, _text(file.get("path"), "model artifact path")),
        format_id=format_id,
        flavor_id="onnx-runtime-v1" if format_id == "onnx" else "xgboost-native-v1",
        import_bundle_uri=_text(import_root, "model import bundle URI"),
        expected_sha256=expected_hash,
        options={
            "variant": "onnx" if format_id == "onnx" else None,
            "input_fields": input_fields,
            "output_fields": output_fields,
        }
        if format_id == "onnx"
        else {"input_fields": input_fields, "output_fields": output_fields},
    )


def _input_layout(
    input_value: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], TransformPipeline]:
    tables = input_value.get("tables")
    if not isinstance(tables, list) or len(tables) != 1:
        _invalid("input.tables must contain exactly one table")
    table = _mapping(tables[0], "input.tables[0]")
    if str(table.get("role") or "PRIMARY").upper() != "PRIMARY":
        _invalid("input.tables[0] must be the PRIMARY table")
    alias = _identifier(table.get("table_alias"), "input.tables[0].table_alias")
    database = _identifier(table.get("database_name"), "input.tables[0].database_name")
    table_name = _identifier(table.get("table_name"), "input.tables[0].table_name")

    features_raw = input_value.get("features")
    if not isinstance(features_raw, list) or not features_raw:
        _invalid("input.features must be a non-empty array")
    features: list[str] = []
    result_names: list[str] = []
    for index, raw in enumerate(features_raw):
        feature = _mapping(raw, f"input.features[{index}]")
        origin = _mapping(feature.get("origin"), f"input.features[{index}].origin")
        origin_alias = origin.get("table_alias")
        if origin_alias not in (None, alias):
            _invalid("all inference features must belong to the PRIMARY table")
        features.append(
            _identifier(
                origin.get("column_name"), f"input.features[{index}].origin.column_name"
            )
        )
        result_names.append(
            _text(
                feature.get("result_column") or features[-1],
                f"input.features[{index}].result_column",
            )
        )
    if len(set(features)) != len(features):
        _invalid("input feature origin columns must be unique")

    entity = _mapping(input_value.get("entity_key"), "input.entity_key")
    entity_origin = _mapping(entity.get("origin"), "input.entity_key.origin")
    if entity_origin.get("table_alias") not in (None, alias):
        _invalid("input.entity_key must belong to the PRIMARY table")
    entity_column = _identifier(
        entity_origin.get("column_name"), "input.entity_key.origin.column_name"
    )
    filters = _query_filters(
        input_value.get("query"),
        table=f"{database}.{table_name}",
        alias=alias,
    )
    return (
        f"{database}.{table_name}",
        entity_column,
        tuple(features),
        tuple(result_names),
        TransformPipeline(steps=filters),
    )


def _query_filters(
    query_value: object,
    *,
    table: str,
    alias: str,
) -> tuple[FilterEq, ...]:
    if query_value is None:
        return ()
    query = _mapping(query_value, "input.query")
    sql = str(query.get("sql") or "").strip()
    if not sql:
        return ()
    if _UNSUPPORTED_SQL.search(sql):
        _invalid(
            "input.query supports only a single-table SELECT with equality filters"
        )
    source = _FROM.search(sql)
    if source is None or source.group(1).lower() != table.lower():
        _invalid("input.query FROM must match the PRIMARY table")
    sql_alias = source.group(2)
    if sql_alias is not None and sql_alias.lower() != alias.lower():
        _invalid("input.query alias must match the PRIMARY table alias")
    where = _WHERE.search(sql)
    if where is None:
        return ()
    params = _mapping(query.get("params", {}), "input.query.params")
    filters: list[FilterEq] = []
    for raw_condition in re.split(r"\s+AND\s+", where.group(1), flags=re.IGNORECASE):
        condition = raw_condition.strip().strip("()")
        match = _EQUALITY.fullmatch(condition)
        if match is None or match.group(1) not in (None, alias):
            _invalid("input.query WHERE supports only parameterized equality filters")
        parameter = match.group(3)
        if parameter not in params:
            _invalid("input.query WHERE references a missing parameter")
        filters.append(FilterEq(column=match.group(2), value=params[parameter]))
    return tuple(filters)


def _credentials(datasource: Mapping[str, Any]) -> tuple[str, str]:
    username = str(datasource.get("username") or "")
    password = datasource.get("password")
    if password is None:
        reference = datasource.get("credential_ref")
        if isinstance(reference, str) and reference.startswith("env://"):
            password = os.environ.get(reference.removeprefix("env://"), "")
        else:
            password = os.environ.get("TRIBUTO_CLICKHOUSE_PASSWORD", "")
    if not isinstance(password, str):
        _invalid("ClickHouse password must be a string or env credential reference")
    return username, password


@contextmanager
def _input_environment(username: str, password: str) -> Iterator[None]:
    names = {
        "TRIBUTO_CLICKHOUSE_USER": username,
        "TRIBUTO_CLICKHOUSE_PASSWORD": password,
    }
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, value in names.items():
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _positive_label_index(model: Mapping[str, Any]) -> int:
    mapping = _mapping(model.get("label_mapping", {}), "model.label_mapping")
    positive = model.get("positive_label_value")
    value = mapping.get(positive, 1) if positive is not None else 1
    return int(value)


def _output_bindings(model: Mapping[str, Any]) -> tuple[TensorOutputBinding, ...]:
    task_type = str(model.get("task_type") or "").upper()
    if task_type in {"BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"}:
        return (
            TensorOutputBinding(
                tensor_name="label",
                column="__knova_label",
                semantic="label",
                dtype="int64",
            ),
            TensorOutputBinding(
                tensor_name="probabilities",
                column="__knova_probabilities",
                semantic="probability",
                dtype="float32",
            ),
        )
    if task_type == "REGRESSION":
        return (
            TensorOutputBinding(
                tensor_name="prediction",
                column="__knova_prediction",
                semantic="score",
                dtype="float32",
                squeeze_singleton=True,
            ),
        )
    _invalid("model.task_type is not supported by XGBoost inference")


def _build_request(
    request: InferenceExecutionRequest,
    *,
    measured_rows: int | None = None,
    batch_size_override: int | None = None,
) -> tuple[InferenceRequest, _ClickHouseResultSink, tuple[str, str]]:
    payload = request.model_dump(mode="python")
    model = _mapping(payload.get("model"), "model")
    input_value = _mapping(payload.get("input"), "input")
    output = _mapping(payload.get("output"), "output")
    table, entity_column, features, result_names, transforms = _input_layout(
        input_value
    )
    datasource = _mapping(input_value.get("datasource"), "input.datasource")
    if _text(datasource.get("type"), "input.datasource.type").upper() != "CLICKHOUSE":
        _invalid("only input.datasource.type=CLICKHOUSE is supported")
    host = _text(datasource.get("host"), "input.datasource.host")
    port = _positive_int(datasource.get("port"), "input.datasource.port", 8123)
    properties = _mapping(
        datasource.get("properties", {}), "input.datasource.properties"
    )
    port = _positive_int(
        properties.get("http_port", port), "input ClickHouse HTTP port", 8123
    )
    database = table.split(".", 1)[0]
    username, password = _credentials(datasource)
    columns = list(
        dict.fromkeys(
            (entity_column, *features, *(step.column for step in transforms.steps))
        )
    )
    batch_size, sink_batch_size, concurrency = _execution_policy(
        payload,
        measured_rows=measured_rows,
        batch_size_override=batch_size_override,
    )
    source = ProviderSourceConfig(
        provider="tributo.clickhouse",
        uri=f"clickhouse://{host}:{port}/{database}",
        options={
            "table": table,
            "host": host,
            "port": port,
            "database": database,
            "columns": columns,
            "partitioning": {
                "mode": "parallel",
                "column": entity_column,
                "num_partitions": concurrency,
            },
        },
    )
    explanation = _explanation_options(payload)
    model_reference = _model_reference(
        model,
        execution_id=request.execution_id,
        features=features,
    )
    attribution_model_reference = model_reference
    if explanation is not None:
        attribution_model_reference = _model_reference(
            model,
            execution_id=request.execution_id,
            features=features,
            prefer_native=True,
        )
    if explanation is not None and (
        not isinstance(attribution_model_reference, BundleModelReference)
        and attribution_model_reference.format_id != "ubj"
    ):
        _invalid("TREE_SHAP requires a UBJ model artifact")
    input_name = (
        model_reference.options["input_fields"][0]["name"]
        if isinstance(model_reference, ArtifactModelReference)
        else "float_input"
    )
    output_bindings = list(_output_bindings(model))
    if isinstance(model_reference, ArtifactModelReference):
        names = [item["name"] for item in model_reference.options["output_fields"]]
        output_bindings = [
            binding.model_copy(update={"tensor_name": name})
            for binding, name in zip(output_bindings, names, strict=True)
        ]

    output_datasource = _mapping(output.get("datasource"), "output.datasource")
    if (
        _text(output_datasource.get("type"), "output.datasource.type").upper()
        != "CLICKHOUSE"
    ):
        _invalid("only output.datasource.type=CLICKHOUSE is supported")
    output_host = _text(output_datasource.get("host"), "output.datasource.host")
    output_port = _positive_int(
        output_datasource.get("port"), "output.datasource.port", 8123
    )
    output_properties = _mapping(
        output_datasource.get("properties", {}), "output.datasource.properties"
    )
    output_port = _positive_int(
        output_properties.get("http_port", output_port),
        "output ClickHouse HTTP port",
        8123,
    )
    output_database = _identifier(
        output_datasource.get("database_name"), "output.datasource.database_name"
    )
    output_table = _qualified_table(
        output_database,
        _identifier(output.get("table_name"), "output.table_name"),
    )
    output_username, output_password = _credentials(output_datasource)
    sink_request = DataWriteTargetRequest(
        target_kind="clickhouse",
        target=output_table,
        mode="append",
    )
    core_request = InferenceRequest(
        model=model_reference,
        input=IngestionRequest(
            source=source,
            engine="ray",
            binding_id=_CLICKHOUSE_BINDING_ID,
            transforms=transforms,
        ),
        input_binding=InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name=str(input_name),
                    columns=features,
                    dtype="float32",
                ),
            ),
            passthrough_columns=(entity_column,),
        ),
        output_binding=OutputBindingSpec(
            tensors=tuple(output_bindings),
            preserve_features=True,
        ),
        result_sink=sink_request,
        execution=RayExecutionPolicy(
            batch_size=batch_size,
            concurrency=concurrency,
        ),
        run_id=request.execution_id,
    )
    sink = _ClickHouseResultSink(
        host=output_host,
        port=output_port,
        database=output_database,
        table=output_table,
        username=output_username,
        password=output_password,
        request=request,
        entity_column=entity_column,
        feature_columns=features,
        feature_result_names=result_names,
        batch_size=sink_batch_size,
        concurrency=concurrency,
        explanation=explanation,
        model_reference=attribution_model_reference,
    )
    return core_request, sink, (username, password)


def _execution_policy(
    payload: Mapping[str, Any],
    *,
    measured_rows: int | None,
    batch_size_override: int | None,
) -> tuple[int, int, int]:
    execution = _mapping(payload.get("execution", {}), "execution")
    extensions = payload.get("extensions")
    tributo = extensions.get("tributo") if isinstance(extensions, Mapping) else None
    runtime = tributo.get("inference_runtime") if isinstance(tributo, Mapping) else None
    options = runtime if isinstance(runtime, Mapping) else {}
    target_batches = _positive_int(
        options.get("adaptive_target_batches"),
        "extensions.tributo.inference_runtime.adaptive_target_batches",
        20,
    )
    minimum = _positive_int(
        options.get("adaptive_min_batch_size"),
        "extensions.tributo.inference_runtime.adaptive_min_batch_size",
        50_000,
    )
    maximum = _positive_int(
        options.get("adaptive_max_batch_size"),
        "extensions.tributo.inference_runtime.adaptive_max_batch_size",
        1_000_000,
    )
    if minimum > maximum:
        _invalid("adaptive_min_batch_size must not exceed adaptive_max_batch_size")

    if batch_size_override is not None:
        base = _positive_int(batch_size_override, "adaptive batch override", 1)
    elif measured_rows is not None and measured_rows > 0:
        base = min(
            measured_rows,
            max(minimum, math.ceil(measured_rows / target_batches)),
            maximum,
        )
    else:
        base = _positive_int(execution.get("batch_size"), "execution.batch_size", 4096)
    predictor_batch = _positive_int(
        options.get("predictor_batch_size"),
        "extensions.tributo.inference_runtime.predictor_batch_size",
        base,
    )
    sink_batch = _positive_int(
        options.get("sink_batch_size"),
        "extensions.tributo.inference_runtime.sink_batch_size",
        min(base, minimum, 50_000),
    )
    concurrency = _positive_int(
        execution.get("concurrency", options.get("max_predictor_actors")),
        "execution.concurrency",
        4,
    )
    return predictor_batch, sink_batch, concurrency


def _parameter_type(value: object) -> str:
    if isinstance(value, bool):
        return "UInt8"
    if isinstance(value, int):
        return "Int64"
    if isinstance(value, float):
        return "Float64"
    if isinstance(value, str):
        return "String"
    _invalid("input.query equality parameters must be scalar JSON values")


def _measure_input_rows(
    request: InferenceRequest,
    credentials: tuple[str, str],
) -> int:
    import clickhouse_connect

    source = request.input.source
    options = source.options
    table = _text(options.get("table"), "input ClickHouse table")
    host = _text(options.get("host"), "input ClickHouse host")
    port = _positive_int(options.get("port"), "input ClickHouse port", 8123)
    database = _text(options.get("database"), "input ClickHouse database")
    clauses: list[str] = []
    parameters: dict[str, object] = {}
    for index, step in enumerate(request.input.transforms.steps):
        if not isinstance(step, FilterEq):
            _invalid("adaptive row counting supports equality filters only")
        name = f"filter_{index}"
        column = _identifier(step.column, "input query filter column")
        clauses.append(f"`{column}` = {{{name}:{_parameter_type(step.value)}}}")
        parameters[name] = step.value
    sql = f"SELECT count() FROM {table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        database=database,
        username=credentials[0],
        password=credentials[1],
    )
    try:
        return int(client.query(sql, parameters=parameters).first_row[0])
    except Exception as exc:
        raise _InferenceExecutionError(
            f"input row count failed ({type(exc).__name__})"
        ) from None
    finally:
        client.close()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class _TreeShapBatch:
    """Load the official UBJ runtime once per Ray actor and add row attributions."""

    def __init__(
        self,
        *,
        model_reference: Mapping[str, Any],
        feature_columns: tuple[str, ...],
        task_type: str,
        approximate: bool,
    ) -> None:
        import xgboost

        self._runtime: Any | None = None
        self._feature_columns = feature_columns
        self._task_type = task_type
        self._approximate = approximate
        if model_reference.get("kind") == "bundle":
            from tributo.exporting.runtime import BundleModelLoader

            self._runtime = BundleModelLoader().open(
                _text(model_reference.get("uri"), "model bundle URI"),
                role=_text(model_reference.get("role"), "model bundle role"),
                storage_profile=model_reference.get("storage_profile"),
                expected_manifest_sha256=model_reference.get(
                    "expected_manifest_sha256"
                ),
                use_case="batch",
            )
            model = self._runtime.model
            booster = getattr(model, "native_model_object", None)
            if not isinstance(booster, xgboost.Booster):
                raise TypeError("Bundle native role is not an XGBoost Booster")
            self._booster = booster
        else:
            uri = _text(model_reference.get("uri"), "UBJ artifact URI")
            payload = _read_artifact_bytes(uri)
            expected = model_reference.get("expected_sha256")
            if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("UBJ artifact digest does not match expected_sha256")
            self._booster = xgboost.Booster()
            self._booster.load_model(bytearray(payload))

        model_names = tuple(self._booster.feature_names or ())
        if model_names and model_names != feature_columns:
            raise ValueError("XGBoost feature names do not match inference columns")

    def __call__(self, frame: Any) -> Any:
        import numpy as np
        import xgboost

        values = frame.loc[:, list(self._feature_columns)].to_numpy(dtype=np.float32)
        matrix = xgboost.DMatrix(
            values,
            feature_names=list(self._feature_columns),
        )
        contributions = np.asarray(
            self._booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=self._approximate,
                strict_shape=True,
            ),
            dtype=np.float64,
        )
        expected = (len(frame), len(self._feature_columns) + 1)
        if contributions.ndim != 3 or contributions.shape[0] != len(frame):
            raise ValueError("XGBoost TreeSHAP output shape is invalid")
        if contributions.shape[2] != expected[1]:
            raise ValueError("XGBoost TreeSHAP feature width is invalid")
        if self._task_type == "MULTICLASS_CLASSIFICATION":
            groups = np.asarray(frame["__knova_label"], dtype=np.int64)
        else:
            groups = np.zeros(len(frame), dtype=np.int64)
        if np.any(groups < 0) or np.any(groups >= contributions.shape[1]):
            raise ValueError("XGBoost TreeSHAP output group is invalid")
        selected = contributions[np.arange(len(frame)), groups]
        result = frame.copy()
        result["__knova_shap_values"] = list(selected[:, :-1])
        result["__knova_shap_base"] = selected[:, -1]
        result["__knova_shap_group"] = groups
        return result

    def __del__(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            with suppress(Exception):
                runtime.close()


def _read_artifact_bytes(uri: str) -> bytes:
    if uri.startswith("s3://"):
        from urllib.parse import urlsplit

        import boto3

        parsed = urlsplit(uri)
        key = parsed.path.lstrip("/")
        if not parsed.netloc or not key:
            raise ValueError("S3 UBJ artifact URI is invalid")
        response = boto3.client(
            "s3",
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        ).get_object(Bucket=parsed.netloc, Key=key)
        body = response["Body"]
        try:
            return bytes(body.read())
        finally:
            body.close()
    path = Path(uri.removeprefix("file://"))
    return path.read_bytes()


def _protocol_batch(
    frame: Any,
    *,
    task: Mapping[str, Any],
    entity_column: str,
    feature_columns: tuple[str, ...],
    feature_result_names: tuple[str, ...],
) -> Any:
    import numpy as np
    import pandas as pd

    model = _mapping(task.get("model"), "model")
    task_type = str(model.get("task_type") or "").upper()
    rows = len(frame)
    labels = np.asarray(frame.get("__knova_label", np.full(rows, -1)))
    reverse = {
        int(value): str(key)
        for key, value in dict(model.get("label_mapping") or {}).items()
    }
    probabilities = frame.get("__knova_probabilities")
    probability_rows = (
        None if probabilities is None else np.asarray(list(probabilities))
    )
    if task_type in {"BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"}:
        label_values = [reverse.get(int(value), str(int(value))) for value in labels]
    else:
        label_values = [None] * rows
    if task_type == "BINARY_CLASSIFICATION" and probability_rows is not None:
        positive = _positive_label_index(model)
        probability_values = [float(row[positive]) for row in probability_rows]
    else:
        probability_values = [None] * rows
    if task_type == "MULTICLASS_CLASSIFICATION" and probability_rows is not None:
        probability_vectors = [
            json.dumps([float(value) for value in row], separators=(",", ":"))
            for row in probability_rows
        ]
    else:
        probability_vectors = [None] * rows
    prediction = frame.get("__knova_prediction")
    prediction_values = (
        [float(value) for value in prediction]
        if prediction is not None and task_type == "REGRESSION"
        else [None] * rows
    )
    explanation = _explanation_options(task)
    shap_values = frame.get("__knova_shap_values")
    shap_bases = frame.get("__knova_shap_base")
    shap_groups = frame.get("__knova_shap_group")
    positions = {name: frame.columns.get_loc(name) for name in feature_columns}
    extras: list[str] = []
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        extra: dict[str, Any] = {
            "inference_feature_values": [
                {
                    "feature_name": result_name,
                    "value": _json_value(row[positions[column]]),
                }
                for column, result_name in zip(
                    feature_columns, feature_result_names, strict=True
                )
            ]
        }
        if explanation is not None:
            if shap_values is None or shap_bases is None or shap_groups is None:
                raise ValueError("TREE_SHAP result columns are missing")
            group = int(shap_groups.iloc[row_index])
            explained_index = (
                _positive_label_index(model)
                if task_type == "BINARY_CLASSIFICATION"
                else group
            )
            explanation_payload: dict[str, Any] = {
                "method": "TREE_SHAP",
                "output_space": "RAW_MARGIN",
                "exactness": ("approximate" if explanation["approximate"] else "exact"),
                "approximate": explanation["approximate"],
                "base_value": float(shap_bases.iloc[row_index]),
                "feature_contributions": [
                    {
                        "feature_name": name,
                        "shap_value": float(value),
                    }
                    for name, value in zip(
                        feature_result_names,
                        shap_values.iloc[row_index],
                        strict=True,
                    )
                ],
            }
            if task_type in {
                "BINARY_CLASSIFICATION",
                "MULTICLASS_CLASSIFICATION",
            }:
                explanation_payload["explained_class_index"] = explained_index
                explanation_payload["explained_class_label"] = reverse.get(
                    explained_index, str(explained_index)
                )
            extra["explanation"] = explanation_payload
        extras.append(
            json.dumps(
                extra,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    result = pd.DataFrame(
        {
            "entity_id": frame[entity_column].map(
                lambda value: "" if pd.isna(value) else str(value)
            ),
            "pred_label": label_values,
            "pred_probability": probability_values,
            "pred_probabilities": probability_vectors,
            "pred_value": prediction_values,
            "pred_extra": extras,
            "execution_id": str(task.get("execution_id") or ""),
            "model_id": str(model.get("model_id") or ""),
            "version_id": str(model.get("version_id") or ""),
            "tenant_id": str(task.get("tenant_id") or ""),
            "inferred_at": datetime.now(UTC),
        }
    )
    output = _mapping(task.get("output"), "output")
    result_filter = output.get("result_filter")
    if isinstance(result_filter, Mapping) and result_filter.get("labels"):
        wanted = {str(value) for value in result_filter["labels"]}
        mode = str(result_filter.get("mode") or "").upper()
        if mode == "INCLUDE":
            result = result[result["pred_label"].isin(wanted)]
        elif mode == "EXCLUDE":
            result = result[~result["pred_label"].isin(wanted)]
        else:
            _invalid("output.result_filter.mode must be INCLUDE or EXCLUDE")
    return result


class _ClickHouseResultSink:
    api_version = 1

    def __init__(
        self,
        *,
        host: str,
        port: int,
        database: str,
        table: str,
        username: str,
        password: str,
        request: InferenceExecutionRequest,
        entity_column: str,
        feature_columns: tuple[str, ...],
        feature_result_names: tuple[str, ...],
        batch_size: int,
        concurrency: int,
        explanation: Mapping[str, Any] | None,
        model_reference: BundleModelReference | ArtifactModelReference,
    ) -> None:
        self._host = host
        self._port = port
        self._database = database
        self._table = table
        self._username = username
        self._password = password
        self._task = request.model_dump(mode="python")
        self._entity_column = entity_column
        self._feature_columns = feature_columns
        self._feature_result_names = feature_result_names
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._explanation = dict(explanation) if explanation is not None else None
        self._model_reference = model_reference.model_dump(mode="python")

    @property
    def sink_id(self) -> str:
        return "data-write-v1"

    def write(
        self,
        dataset: Any,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        import ray.data

        try:
            if self._explanation is not None:
                # The public inference runtime uses a predictor ActorPool. Finish
                # that stage before allocating attribution actors so both pools
                # do not reserve cluster CPUs at the same time.
                dataset = dataset.materialize()
                model = _mapping(self._task.get("model"), "model")
                dataset = dataset.map_batches(
                    _TreeShapBatch,
                    batch_format="pandas",
                    batch_size=self._batch_size,
                    compute=ray.data.ActorPoolStrategy(size=self._concurrency),
                    fn_constructor_kwargs={
                        "model_reference": self._model_reference,
                        "feature_columns": self._feature_columns,
                        "task_type": str(model.get("task_type") or "").upper(),
                        "approximate": self._explanation["approximate"],
                    },
                )
            transformed = dataset.map_batches(
                _protocol_batch,
                batch_format="pandas",
                fn_kwargs={
                    "task": self._task,
                    "entity_column": self._entity_column,
                    "feature_columns": self._feature_columns,
                    "feature_result_names": self._feature_result_names,
                },
            )
            transformed = transformed.materialize()
            rows_written = transformed.count()
        except Exception as exc:
            raise ResultMaterializationError(type(exc).__name__) from None
        try:
            transformed.write_clickhouse(
                table=self._table,
                dsn=_dsn(
                    host=self._host,
                    port=self._port,
                    database=self._database,
                    user=self._username,
                    password=self._password,
                ),
                mode=ray.data.SinkMode.APPEND,
                max_insert_block_rows=self._batch_size,
            )
        except Exception as exc:
            raise ResultWriteError(
                f"ClickHouse result write failed ({type(exc).__name__})"
            ) from None
        result_id = hashlib.sha256(
            f"{run_id}|{plan_digest}|{self._table}".encode()
        ).hexdigest()
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id=result_id,
            uri=f"clickhouse://{self._host}:{self._port}/{self._table}",
            rows_written=rows_written,
            metadata={"target_kind": "clickhouse", "committed": "true"},
        )


def execute_inference(
    request: InferenceExecutionRequest,
    reporter: Any,
) -> Any:
    """Execute one KnoVa batch request through Tributo's public inference API."""
    try:
        reporter.phase("PREPARING")
        initial_request, _initial_sink, input_credentials = _build_request(request)
        measured_rows = _measure_input_rows(initial_request, input_credentials)
        core_request, sink, input_credentials = _build_request(
            request,
            measured_rows=measured_rows,
        )
    except _InferenceConfigurationError:
        raise
    except Exception:
        raise _InferenceConfigurationError("inference request mapping failed") from None

    try:
        reporter.phase("EXECUTING")
        retries = 0
        while True:
            with _input_environment(*input_credentials):
                result = run_inference(core_request, bound_sink=sink)
            failure = result.failure
            if (
                result.status == "failed"
                and failure is not None
                and failure.phase == "materialization"
                and failure.error_type in _MEMORY_PRESSURE_ERRORS
                and core_request.execution.batch_size > 1
                and retries < _MAX_ADAPTIVE_RETRIES
            ):
                retries += 1
                next_batch = max(1, core_request.execution.batch_size // 2)
                reporter.publish(
                    "LOG",
                    {
                        "message": (
                            "Memory pressure detected; retrying inference with "
                            f"batch_size={next_batch}"
                        )
                    },
                    phase="EXECUTING",
                )
                core_request, sink, input_credentials = _build_request(
                    request,
                    measured_rows=measured_rows,
                    batch_size_override=next_batch,
                )
                continue
            break
        if result.status != "succeeded":
            failure_type = (
                result.failure.error_type if result.failure else "InferenceFailed"
            )
            raise _InferenceExecutionError(
                f"Tributo inference did not succeed ({failure_type})"
            )
        reporter.publish(
            "COMPLETED",
            {
                "processed_rows": measured_rows,
                "result_rows": result.output_rows or 0,
                "total_rows": measured_rows,
                "result_summary": {
                    "status": result.status,
                    "output_rows": result.output_rows,
                    "result_uri": result.sink_receipt.uri
                    if result.sink_receipt
                    else None,
                },
                "inference_result": result.model_dump(mode="json"),
            },
            phase="COMPLETED",
        )
        return result
    except _InferenceExecutionError:
        raise
    except Exception as exc:
        raise _InferenceExecutionError(
            f"inference execution failed ({type(exc).__name__})"
        ) from None


__all__ = ["execute_inference"]
