"""KnoVa protocol mapping for Tributo's public batch-inference runtime."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
        if fmt not in {"onnx", "ubj"} or not isinstance(files, list) or len(files) != 1:
            continue
        file = files[0]
        if isinstance(file, Mapping):
            supported.append((raw, file))
    if not supported:
        _invalid("model artifacts require one ONNX or UBJ weight file")
    return next(
        (item for item in supported if item[0].get("format") == "onnx"), supported[0]
    )


def _model_reference(
    model_value: Mapping[str, Any],
    *,
    execution_id: str,
    features: tuple[str, ...],
) -> BundleModelReference | ArtifactModelReference:
    model = dict(model_value)
    explicit_bundle = model.get("bundle_uri")
    storage = _mapping(model.get("storage", {}), "model.storage")
    properties = _mapping(storage.get("properties", {}), "model.storage.properties")
    explicit_bundle = explicit_bundle or properties.get("bundle_uri")
    if explicit_bundle is not None:
        return BundleModelReference(uri=_text(explicit_bundle, "model.bundle_uri"))

    root = _storage_root(storage)
    if not root.startswith("s3://") and (Path(root) / "manifest.json").is_file():
        return BundleModelReference(uri=root)

    alternative, file = _artifact_alternative(model)
    format_id = _text(alternative.get("format"), "model artifact format").lower()
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
    execution = _mapping(payload.get("execution", {}), "execution")
    batch_size = _positive_int(
        execution.get("batch_size"), "execution.batch_size", 4096
    )
    concurrency = _positive_int(
        execution.get("concurrency"), "execution.concurrency", 4
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
    model_reference = _model_reference(
        model,
        execution_id=request.execution_id,
        features=features,
    )
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
        batch_size=batch_size,
    )
    return core_request, sink, (username, password)


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
    positions = {name: frame.columns.get_loc(name) for name in feature_columns}
    extras: list[str] = []
    for row in frame.itertuples(index=False, name=None):
        extras.append(
            json.dumps(
                {
                    "inference_feature_values": [
                        {
                            "feature_name": result_name,
                            "value": _json_value(row[positions[column]]),
                        }
                        for column, result_name in zip(
                            feature_columns, feature_result_names, strict=True
                        )
                    ]
                },
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
        import ray.data

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
        core_request, sink, input_credentials = _build_request(request)
    except _InferenceConfigurationError:
        raise
    except Exception:
        raise _InferenceConfigurationError("inference request mapping failed") from None

    try:
        reporter.phase("PREPARING")
        reporter.phase("EXECUTING")
        with _input_environment(*input_credentials):
            result = run_inference(core_request, bound_sink=sink)
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
                "processed_rows": result.output_rows or 0,
                "result_rows": result.output_rows or 0,
                "total_rows": result.output_rows or 0,
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
