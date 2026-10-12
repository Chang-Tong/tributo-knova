"""KnoVa protocol mapping for Tributo's public batch-inference runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

from tributo.data import (
    DataWriteTargetRequest,
    FilterComparison,
    FilterEq,
    FilterNotEq,
    IngestionRequest,
    Limit,
    ProviderSourceConfig,
    RayDataHandle,
    ReadOptions,
    TransformPipeline,
    default_write_gateway,
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

from tributo_knova.protocol import InferenceExecutionRequest
from tributo_knova.reporter import ExecutionControlError
from tributo_knova.validation import ProtocolValidationError, validate_inference_payload

_CLICKHOUSE_BINDING_ID = "ray_clickhouse.ray.clickhouse"
_CLICKHOUSE_WRITE_BINDING_ID = "tributo.ray.clickhouse"
_OUTPUT_PASSWORD_ENV = "KNOVA_OUTPUT_CLICKHOUSE_PASSWORD"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FROM = re.compile(
    r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_WHERE = re.compile(
    r"\bWHERE\s+(.+?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_LIMIT_TOKEN = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_DIRECT_LIMIT = re.compile(r"\bLIMIT\s+\d+\s*$", re.IGNORECASE)
_LIMITED_SUBQUERY = re.compile(
    r"^\s*SELECT\s+\*\s+FROM\s*\(\s*SELECT\b.+"
    r"\bLIMIT\s+\d+\s*\)\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PREDICATE = re.compile(
    r"(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(<=|>=|!=|<>|=|<|>)\s*"
    r"\{([A-Za-z_][A-Za-z0-9_]*):([A-Za-z0-9_(), ]+)\}",
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
_CLICKHOUSE_READ_TASK_ROWS = 200_000
_CLICKHOUSE_READ_TASK_BYTES = 64 * 1024 * 1024
_RESULT_COLUMNS = (
    "entity_id",
    "pred_label",
    "pred_probability",
    "pred_probabilities",
    "pred_value",
    "pred_extra",
    "execution_id",
    "model_id",
    "version_id",
    "tenant_id",
    "inferred_at",
)


class _InferenceConfigurationError(ValueError):
    """Credential-safe request mapping failure."""

    def __init__(
        self, message: str, *, code: str = "INVALID_PAYLOAD", phase: str = "LOAD_MODEL"
    ) -> None:
        super().__init__(message)
        self.code, self.phase = code, phase


class _InferenceExecutionError(RuntimeError):
    """Credential-safe runtime failure."""

    def __init__(
        self, message: str, *, code: str = "PREDICTION_FAILED", phase: str = "PREDICT"
    ) -> None:
        super().__init__(message)
        self.code, self.phase = code, phase


class _EmptyInferenceInputError(_InferenceConfigurationError):
    """A successfully measured inference query returned zero rows."""


def _invalid(
    message: str, *, code: str = "INVALID_PAYLOAD", phase: str = "LOAD_MODEL"
) -> NoReturn:
    raise _InferenceConfigurationError(message, code=code, phase=phase)


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


def _qualified_table(database: str, table: str) -> str:
    database_name = _identifier(database, "ClickHouse database")
    table_name = _identifier(table, "ClickHouse table")
    return f"{database_name}.{table_name}"


def _native_table_name(value: str, database: str) -> str:
    parts = value.split(".")
    if len(parts) != 2 or parts[0] != database:
        _invalid("input table must use the same database as its datasource")
    return _identifier(parts[1], "input table name")


def _positive_int(value: object, field: str, default: int) -> int:
    raw = default if value is None else value
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        _invalid(f"{field} must be a positive integer")
    return raw


def _storage_root(storage: Mapping[str, Any]) -> str:
    storage_type = _text(storage.get("type"), "model.storage.type").lower()
    bucket = _text(storage.get("bucket"), "model.storage.bucket")
    raw_prefix = storage.get("prefix")
    if (
        not isinstance(raw_prefix, str)
        or not raw_prefix.endswith("/")
        or raw_prefix.startswith("/")
        or ":" in raw_prefix
        or "\\" in raw_prefix
    ):
        _invalid("model.storage.prefix must be relative and end with /")
    prefix = raw_prefix.rstrip("/")
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
    relative = _text(path, "model artifact path")
    if (
        relative.startswith("/")
        or ":" in relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        _invalid("model artifact path must stay below model storage")
    return root.rstrip("/") + "/" + relative


def _signature_fields(
    model: Mapping[str, Any],
    *,
    feature_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_type = str(model.get("task_type") or "").upper()
    class_count = max(
        2, len(_mapping(model.get("label_mapping") or {}, "model.label_mapping"))
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


def _validate_artifacts(model: Mapping[str, Any]) -> None:
    artifacts = _mapping(model.get("model_artifacts"), "model.model_artifacts")
    if not artifacts or set(artifacts) - {"pipeline", "model_weights", "preprocessor"}:
        _invalid("model_artifacts contains an unknown or missing artifact part")
    formats = {
        "onnx",
        "sklearn",
        "spark_pipeline_model",
        "pytorch",
        "tensorflow",
        "tensorrt",
        "xgboost",
        "lightgbm",
        "ray_checkpoint",
        "ray_preprocessor",
    }
    paths: set[str] = set()
    unsupported_required = False
    for name, raw in artifacts.items():
        entry = _mapping(raw, "artifact entry")
        if not isinstance(entry.get("required_for_inference"), bool):
            _invalid("artifact required_for_inference must be boolean")
        alternatives = entry.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            _invalid("artifact alternatives must be a non-empty array")
        seen_formats: set[str] = set()
        for raw_alternative in alternatives:
            alternative = _mapping(raw_alternative, "artifact alternative")
            fmt = alternative.get("format")
            if not isinstance(fmt, str) or fmt not in formats or fmt in seen_formats:
                _invalid(
                    "artifact formats must be supported protocol names and "
                    "unique per part"
                )
            seen_formats.add(fmt)
            files = alternative.get("files")
            if not isinstance(files, list) or not files:
                _invalid("artifact files must be a non-empty array")
            for raw_file in files:
                file = _mapping(raw_file, "artifact file")
                path = _text(file.get("path"), "artifact path")
                _join_uri("root", path)
                if path in paths:
                    _invalid("artifact paths must be globally unique")
                paths.add(path)
                size = file.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    _invalid("artifact size must be a non-negative integer")
                if (
                    not isinstance(file.get("hash"), str)
                    or re.fullmatch(r"sha256:[0-9a-fA-F]{64}", file["hash"]) is None
                ):
                    _invalid(
                        "artifact hash must use sha256 with 64 hexadecimal characters"
                    )
        unsupported_required |= (
            name != "model_weights" and entry["required_for_inference"]
        )
    if unsupported_required or "model_weights" not in artifacts:
        _invalid(
            "This runtime cannot apply the required pipeline or preprocessor",
            code="MODEL_FORMAT_UNSUPPORTED",
        )


def _verified_model_schema(
    model: Mapping[str, Any], reference: ArtifactModelReference
) -> tuple[tuple[str, ...], int | None]:
    """Inspect only verified bytes; table ordering comes from the model itself."""
    alternative, selected = _artifact_alternative(
        model, prefer_ubj=reference.format_id == "ubj"
    )
    root = _storage_root(_mapping(model.get("storage"), "model.storage"))
    selected_bytes = None
    sidecar_bytes = None
    for file in alternative["files"]:
        try:
            data = _read_artifact_bytes(_join_uri(root, file["path"]))
        except FileNotFoundError:
            _invalid(
                "Declared model artifact was not found", code="MODEL_ARTIFACT_NOT_FOUND"
            )
        except Exception:
            _invalid(
                "Declared model artifact could not be read", code="MODEL_LOAD_FAILED"
            )
        if (
            len(data) != file["size"]
            or hashlib.sha256(data).hexdigest() != file["hash"][7:].lower()
        ):
            _invalid(
                "Model artifact size or SHA-256 does not match",
                code="MODEL_ARTIFACT_CORRUPTED",
            )
        if file is selected:
            selected_bytes = data
        if file["path"].split("/")[-1] == "feature_names.json":
            if sidecar_bytes is not None:
                _invalid(
                    "Model feature schema sidecar is ambiguous",
                    code="MODEL_SCHEMA_MISMATCH",
                )
            sidecar_bytes = data
    class_count = None
    try:
        if reference.format_id == "onnx":
            import onnx

            proto = onnx.load_model_from_string(selected_bytes)
            if any(
                tensor.data_location == onnx.TensorProto.EXTERNAL
                for tensor in proto.graph.initializer
            ):
                _invalid(
                    "This runtime requires a self-contained ONNX artifact",
                    code="MODEL_FORMAT_UNSUPPORTED",
                )
            if [value.name for value in proto.graph.input] != [
                field["name"] for field in reference.options["input_fields"]
            ] or [value.name for value in proto.graph.output] != [
                field["name"] for field in reference.options["output_fields"]
            ]:
                _invalid(
                    "Model tensor names do not match declared task signature",
                    code="MODEL_SCHEMA_MISMATCH",
                )
            expected_outputs = reference.options["output_fields"]
            if model["task_type"] in {
                "BINARY_CLASSIFICATION",
                "MULTICLASS_CLASSIFICATION",
            }:
                dims = proto.graph.output[1].type.tensor_type.shape.dim
                class_count = dims[1].dim_value if len(dims) == 2 else None
                if not class_count:
                    _invalid(
                        "Classification artifact must declare probability width",
                        code="MODEL_SCHEMA_MISMATCH",
                    )
            for tensor, expected_field in zip(
                proto.graph.output, expected_outputs, strict=True
            ):
                shape = tensor.type.tensor_type.shape.dim
                expected_shape = list(expected_field["shape"])
                if (
                    expected_field["name"] == expected_outputs[-1]["name"]
                    and class_count is not None
                ):
                    expected_shape = ["batch", class_count]
                if len(shape) != len(expected_shape) or any(
                    isinstance(size, int) and dim.dim_value and dim.dim_value != size
                    for dim, size in zip(shape, expected_shape, strict=True)
                ):
                    _invalid(
                        "Model output shape does not match task or label mapping",
                        code="MODEL_SCHEMA_MISMATCH",
                    )
            metadata = {item.key: item.value for item in proto.metadata_props}
            names = json.loads(metadata.get("feature_names", "null"))
            import onnxruntime

            onnxruntime.InferenceSession(
                selected_bytes, providers=["CPUExecutionProvider"]
            )
            dimensions = proto.graph.input[0].type.tensor_type.shape.dim
            if len(dimensions) != 2 or (
                dimensions[1].dim_value and dimensions[1].dim_value != len(names or [])
            ):
                _invalid(
                    "Model input width does not match its feature metadata",
                    code="MODEL_SCHEMA_MISMATCH",
                )
        else:
            import xgboost

            booster = xgboost.Booster()
            booster.load_model(bytearray(selected_bytes))
            config = json.loads(booster.save_config())["learner"]
            objective = config["objective"]["name"]
            task = model["task_type"]
            if (
                (task == "REGRESSION" and not objective.startswith("reg:"))
                or (task == "BINARY_CLASSIFICATION" and objective != "binary:logistic")
                or (
                    task == "MULTICLASS_CLASSIFICATION"
                    and not objective.startswith("multi:")
                )
            ):
                _invalid(
                    "Model objective does not match task_type",
                    code="MODEL_SCHEMA_MISMATCH",
                )
            class_count = (
                int(config["learner_model_param"]["num_class"])
                if task == "MULTICLASS_CLASSIFICATION"
                else (2 if task == "BINARY_CLASSIFICATION" else None)
            )
            names = booster.feature_names
            if not names and sidecar_bytes is not None:
                names = json.loads(sidecar_bytes)
            if names and booster.num_features() != len(names):
                _invalid(
                    "Native model width does not match its feature schema",
                    code="MODEL_SCHEMA_MISMATCH",
                )
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) and n for n in names)
            or len(names) != len(set(names))
        ):
            _invalid(
                "Model must contain unique feature_names metadata",
                code="MODEL_SCHEMA_MISMATCH",
            )
        mapping = model.get("label_mapping") or {}
        if class_count is not None and (
            model["task_type"] == "BINARY_CLASSIFICATION"
            and class_count != 2
            or mapping
            and (
                len(mapping) != class_count
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in mapping.values()
                )
                or set(mapping.values()) != set(range(class_count))
            )
        ):
            _invalid(
                "Model class count does not match task or label mapping",
                code="MODEL_SCHEMA_MISMATCH",
            )
        return tuple(names), class_count
    except _InferenceConfigurationError:
        raise
    except Exception:
        _invalid(
            "Verified model artifact could not be loaded", code="MODEL_LOAD_FAILED"
        )


def _normalize_native_model(
    reference: ArtifactModelReference, names: tuple[str, ...], execution_id: str
) -> BundleModelReference:
    """Use the installed official exporter/flavor, not a retired Core flavor."""
    import xgboost
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    payload = _read_artifact_bytes(reference.uri)
    if hashlib.sha256(payload).hexdigest() != reference.expected_sha256:
        _invalid(
            "Native artifact changed during preparation",
            code="MODEL_ARTIFACT_CORRUPTED",
        )
    booster = xgboost.Booster()
    booster.load_model(bytearray(payload))
    # The official runtime restores opaque names from its verified sidecar;
    # its numerical export validator expects the Booster tensor without names.
    booster.feature_names = None
    objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
    contract = ExportCheckpointV1(
        trainer_type="xgboost",
        architecture_id="xgboost",
        input_schema=tuple(
            CheckpointField.model_validate(field)
            for field in reference.options["input_fields"]
        ),
        output_schema=tuple(
            CheckpointField.model_validate(field)
            for field in reference.options["output_fields"]
        ),
        preprocessing={"type": "none"},
        task_type="classification"
        if objective.startswith(("binary:", "multi:"))
        else "regression",
        framework="xgboost",
        framework_version=xgboost.__version__,
        checkpoint_format_version=1,
    )
    source = ExportSource(
        source_kind="xgboost_result",
        model_object=booster,
        architecture_id="xgboost",
        feature_schema={"feature_names": list(names)},
        metadata={
            "framework": "xgboost",
            "framework_versions": {"xgboost": xgboost.__version__},
            "objective": objective,
        },
        source_fingerprint=reference.expected_sha256,
        checkpoint_contract=contract,
    )
    try:
        bundle = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=reference.import_bundle_uri,
                request_id=execution_id,
                run_id=execution_id,
                targets=[
                    ExportTarget(
                        name="native-model",
                        format="ubj",
                        exporter_id="official-xgboost-ubj-v1",
                    )
                ],
                roles={"inference": "native-model"},
            ),
        )
    except Exception:
        _invalid(
            "Verified native model could not be normalized", code="MODEL_LOAD_FAILED"
        )
    return BundleModelReference.from_bundle_ref(bundle)


def _prepare_model(
    core: InferenceRequest, sink: Any, request: InferenceExecutionRequest
) -> InferenceRequest:
    names, class_count = _verified_model_schema(request.model, core.model)
    if class_count is not None:
        import copy

        options = copy.deepcopy(core.model.options)
        options["output_fields"][-1]["shape"] = ["batch", class_count]
        core = core.model_copy(
            update={"model": core.model.model_copy(update={"options": options})}
        )
    mapping = dict(zip(sink._feature_result_names, sink._feature_columns, strict=True))
    if set(names) != set(mapping):
        _invalid(
            "Model feature names do not match input result_column names",
            code="MODEL_SCHEMA_MISMATCH",
        )
    columns = tuple(mapping[name] for name in names)
    sink._feature_columns, sink._feature_result_names = columns, names
    sink._model_feature_names = names
    if sink._explanation is not None:
        native = ArtifactModelReference.model_validate(sink._model_reference)
        native_names, native_count = _verified_model_schema(request.model, native)
        if native_names != names or native_count != class_count:
            _invalid(
                "Prediction and explanation model schemas differ",
                code="MODEL_SCHEMA_MISMATCH",
            )
    model_reference = core.model
    if model_reference.format_id == "ubj":
        model_reference = _normalize_native_model(
            model_reference, names, request.execution_id
        )
    binding = core.input_binding.tensors[0].model_copy(update={"columns": columns})
    return core.model_copy(
        update={
            "model": model_reference,
            "input_binding": core.input_binding.model_copy(
                update={"tensors": (binding,)}
            ),
        }
    )


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
        if fmt not in {"onnx", "xgboost"} or not isinstance(files, list):
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
        _invalid(
            "model artifacts require one ONNX or XGBoost UBJ weight file",
            code="MODEL_FORMAT_UNSUPPORTED",
        )
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
        return None
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        return None
    if not enabled:
        return None
    if str(raw.get("method") or "TREE_SHAP").upper() != "TREE_SHAP":
        return None
    approximate = raw.get("approximate", False)
    if not isinstance(approximate, bool):
        return None
    model = _mapping(payload.get("model"), "model")
    if str(model.get("algorithm_key") or "").lower() != "xgboost":
        return None
    return {"method": "TREE_SHAP", "approximate": approximate}


def _model_reference(
    model_value: Mapping[str, Any],
    *,
    execution_id: str,
    features: tuple[str, ...],
    prefer_native: bool = False,
) -> BundleModelReference | ArtifactModelReference:
    model = dict(model_value)
    storage = _mapping(model.get("storage"), "model.storage")
    properties = _mapping(storage.get("properties", {}), "model.storage.properties")
    root = _storage_root(storage)
    _validate_artifacts(model)
    # Always resolve the declared, hashed wire artifact. A colocated internal
    # bundle or a bundle_uri extension must not bypass the protocol manifest.
    alternative, file = _artifact_alternative(model, prefer_ubj=prefer_native)
    protocol_format = _text(alternative.get("format"), "model artifact format").lower()
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
        expected_hash = raw_hash.removeprefix("sha256:").lower()
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
    transforms = _query_transforms(
        input_value.get("query"),
        table=f"{database}.{table_name}",
        alias=alias,
    )
    _validate_projection(
        _mapping(input_value.get("query"), "input.query"),
        table=f"{database}.{table_name}",
        alias=alias,
        expected={
            entity_column: _text(
                entity.get("result_column"), "input.entity_key.result_column"
            ),
            **dict(zip(features, result_names, strict=True)),
        },
    )
    return (
        f"{database}.{table_name}",
        entity_column,
        tuple(features),
        tuple(result_names),
        TransformPipeline(steps=transforms),
    )


def _query_transforms(
    query_value: object,
    *,
    table: str,
    alias: str,
) -> tuple[FilterEq | FilterNotEq | FilterComparison | Limit, ...]:
    if query_value is None:
        _invalid("input.query is required")
    query = _mapping(query_value, "input.query")
    sql = str(query.get("sql") or "").strip()
    if not sql:
        _invalid("input.query.sql is required")
    # Ingestion Transform IR currently has no sorting operation. Dropping this
    # clause would change which entities a subsequent LIMIT selects.
    if _ORDER_BY.search(sql):
        _invalid(
            "input.query ORDER BY is not supported by this runtime; "
            "disable deterministic ordering"
        )
    if _UNSUPPORTED_SQL.search(sql):
        _invalid("input.query supports only a single-table SELECT with scalar filters")
    select_count = len(re.findall(r"\bSELECT\b", sql, flags=re.IGNORECASE))
    if select_count not in {1, 2} or (
        select_count == 2 and _LIMITED_SUBQUERY.fullmatch(sql) is None
    ):
        _invalid("input.query supports only a single-table SELECT with scalar filters")
    source = _FROM.search(sql)
    if source is None or source.group(1) != table:
        _invalid("input.query FROM must match the PRIMARY table")
    sql_alias = source.group(2)
    if sql_alias is not None and sql_alias != alias:
        _invalid("input.query alias must match the PRIMARY table alias")
    where = _WHERE.search(sql)
    params = _mapping(query.get("params", {}), "input.query.params")
    transforms: list[FilterEq | FilterNotEq | FilterComparison | Limit] = []
    if where is not None:
        for raw_condition in re.split(
            r"\s+AND\s+", where.group(1), flags=re.IGNORECASE
        ):
            condition = raw_condition.strip().strip("()")
            match = _PREDICATE.fullmatch(condition)
            condition_alias = None if match is None else match.group(1)
            if match is None or (
                condition_alias is not None and condition_alias != alias
            ):
                _invalid("input.query WHERE supports only parameterized scalar filters")
            parameter = match.group(4)
            if parameter not in params:
                _invalid("input.query WHERE references a missing parameter")
            column = match.group(2)
            operator = match.group(3)
            value = _typed_query_parameter(params[parameter], match.group(5))
            if operator == "=":
                transforms.append(FilterEq(column=column, value=value))
            elif operator in {"!=", "<>"}:
                transforms.append(FilterNotEq(column=column, value=value))
            else:
                transforms.append(
                    FilterComparison(
                        column=column,
                        operator={
                            "<": "lt",
                            "<=": "lte",
                            ">": "gt",
                            ">=": "gte",
                        }[operator],
                        value=value,
                    )
                )
    limits = _LIMIT.findall(sql)
    if (
        len(limits) > 1
        or (_LIMIT_TOKEN.search(sql) is not None and not limits)
        or (select_count == 1 and limits and _DIRECT_LIMIT.search(sql) is None)
    ):
        _invalid("input.query LIMIT must be a single non-negative integer")
    if limits:
        transforms.append(Limit(count=int(limits[0])))
    return tuple(transforms)


def _typed_query_parameter(value: object, sql_type: str) -> object:
    sql_type = sql_type.strip()
    if sql_type == "String" and isinstance(value, str):
        return value
    integer = re.fullmatch(r"(U?)Int(8|16|32|64)", sql_type)
    if (
        integer
        and not isinstance(value, bool)
        and (
            isinstance(value, int)
            or isinstance(value, str)
            and re.fullmatch(r"[+-]?\d+", value)
        )
    ):
        number = int(value)
        bits = int(integer.group(2))
        unsigned = bool(integer.group(1))
        if (
            (0 if unsigned else -(2 ** (bits - 1)))
            <= number
            < 2 ** (bits if unsigned else bits - 1)
        ):
            return number
    if (
        sql_type == "Float64"
        and not isinstance(value, bool)
        and isinstance(value, (int, float, str))
    ):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
    _invalid(
        "input.query parameter type or value is unsupported; implicit coercion "
        "is disabled"
    )


def _validate_projection(
    query: Mapping[str, Any], *, table: str, alias: str, expected: Mapping[str, str]
) -> None:
    sql = query["sql"].strip()
    if _LIMITED_SUBQUERY.fullmatch(sql):
        sql = sql[sql.index("(") + 1 : sql.rfind(")")].strip()
    match = re.fullmatch(
        r"SELECT\s+(.+?)\s+FROM\s+"
        + re.escape(table)
        + r"\s+(?:AS\s+)?"
        + re.escape(alias)
        + r"(?:\s+WHERE\s+.+?)?(?:\s+LIMIT\s+\d+)?",
        sql,
        re.I | re.S,
    )
    if match is None:
        _invalid("input.query supports only a single-table SELECT with scalar filters")
    projections = {}
    for item in match.group(1).split(","):
        projection = re.fullmatch(
            r"\s*(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+(?:`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))\s*",
            item,
            re.I,
        )
        if projection is None:
            _invalid(
                "input.query SELECT must use direct columns with declared "
                "result_column aliases"
            )
        qualifier, column, quoted, plain = projection.groups()
        if qualifier is not None and qualifier != alias:
            _invalid("input.query SELECT qualifier must match the declared table alias")
        result = quoted or plain
        if column in projections or result in projections.values():
            _invalid("input.query SELECT aliases must be unique")
        projections[column] = result
    if projections != dict(expected):
        _invalid(
            "input.query SELECT must match declared origin and result_column mappings"
        )


def _credentials(
    datasource: Mapping[str, Any],
    *,
    inline_password_env: str,
) -> tuple[str, str, str]:
    username = str(datasource.get("username") or "")
    password = datasource.get("password")
    password_env = inline_password_env
    reference = datasource.get("credential_ref")
    if reference is not None or password is None:
        if isinstance(reference, str) and reference.startswith("env://"):
            password_env = reference.removeprefix("env://")
            if _ENV_NAME.fullmatch(password_env) is None:
                _invalid("ClickHouse credential_ref must name an environment variable")
            password = os.environ.get(password_env)
            if password is None:
                _invalid(
                    "ClickHouse credential_ref environment variable is not configured"
                )
        elif reference is not None:
            _invalid("ClickHouse credential_ref must use env://VARIABLE")
        else:
            password = os.environ.get("TRIBUTO_CLICKHOUSE_PASSWORD", "")
    if not isinstance(password, str):
        _invalid("ClickHouse password must be a string or env credential reference")
    return username, password, password_env


@contextmanager
def _runtime_environment(names: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, value in names.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _positive_label_index(model: Mapping[str, Any]) -> int:
    mapping = _mapping(model.get("label_mapping") or {}, "model.label_mapping")
    positive = model.get("positive_label_value")
    if mapping:
        if (
            len(mapping) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in mapping.values()
            )
            or set(mapping.values()) != {0, 1}
        ):
            _invalid("binary model.label_mapping must uniquely encode classes 0 and 1")
        if positive is not None and (
            not isinstance(positive, str) or positive not in mapping
        ):
            _invalid("model.positive_label_value must belong to model.label_mapping")
        return mapping[positive] if positive is not None else 1
    if positive is None:
        return 1
    if isinstance(positive, str) and positive in {"0", "1"}:
        return int(positive)
    if (
        isinstance(positive, int)
        and not isinstance(positive, bool)
        and positive in {0, 1}
    ):
        return positive
    _invalid("model.positive_label_value must be 0 or 1 without label_mapping")


def _binary_labels(model: Mapping[str, Any]) -> tuple[int, str, str]:
    """Return the encoded positive class and the two original label values."""
    mapping = _mapping(model.get("label_mapping") or {}, "model.label_mapping")
    if not mapping:
        positive = _positive_label_index(model)
        return positive, str(positive), str(1 - positive)
    if len(mapping) != 2:
        _invalid("binary decision_threshold requires exactly two label mappings")
    reverse: dict[int, str] = {}
    for original, encoded in mapping.items():
        if isinstance(encoded, bool) or not isinstance(encoded, int):
            _invalid("model.label_mapping values must be integer class indexes")
        if encoded in reverse:
            _invalid("model.label_mapping class indexes must be unique")
        reverse[encoded] = str(original)
    positive = _positive_label_index(model)
    if positive not in reverse:
        _invalid("model positive_label_value is not present in label_mapping")
    negative = next((index for index in reverse if index != positive), None)
    if negative is None:
        _invalid("binary decision_threshold requires a negative label")
    return positive, reverse[positive], reverse[negative]


def _decision_threshold(payload: Mapping[str, Any]) -> float | None:
    """Read the per-execution threshold injected by KnoVa's control service."""
    extensions = payload.get("extensions", {})
    if not isinstance(extensions, Mapping):
        return None
    raw = extensions.get("decision_threshold")
    if raw is None:
        return None
    model = _mapping(payload.get("model"), "model")
    if str(model.get("task_type") or "").upper() != "BINARY_CLASSIFICATION":
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    threshold = float(raw)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        return None
    _binary_labels(model)
    return threshold


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
) -> tuple[InferenceRequest, _ClickHouseResultSink, dict[str, str]]:
    payload = request.model_dump(mode="python")
    try:
        validate_inference_payload(payload)
    except ProtocolValidationError as exc:
        raise _InferenceConfigurationError(str(exc)) from None
    model = _mapping(payload.get("model"), "model")
    if str(model.get("task_type") or "").upper() == "BINARY_CLASSIFICATION":
        _positive_label_index(model)
    _decision_threshold(payload)
    input_value = _mapping(payload.get("input"), "input")
    if input_value.get("timeseries_pivot") is not None:
        _invalid("This runtime does not support timeseries pivot inference")
    output = _mapping(payload.get("output"), "output")
    _validate_result_filter(model, output)
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
    table_name = _native_table_name(table, database)
    username, password, _input_password_env = _credentials(
        datasource,
        inline_password_env="TRIBUTO_CLICKHOUSE_PASSWORD",
    )
    columns = list(
        dict.fromkeys(
            (
                entity_column,
                *features,
                *(
                    step.column
                    for step in transforms.steps
                    if isinstance(step, (FilterEq, FilterNotEq, FilterComparison))
                ),
            )
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
            "table": table_name,
            "host": host,
            "port": port,
            "database": database,
            "columns": columns,
            # KnoVa points at ClickHouse Distributed tables. Ray's ClickHouse
            # connector cannot split those tables by database partitions, so
            # read through the coordinator and let Ray parallelize downstream.
            "partitioning": {"mode": "single"},
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
    output_username, output_password, output_password_env = _credentials(
        output_datasource,
        inline_password_env=_OUTPUT_PASSWORD_ENV,
    )
    uri_host = (
        f"[{output_host}]"
        if ":" in output_host and not output_host.startswith("[")
        else output_host
    )
    sink_request = DataWriteTargetRequest(
        target_kind="clickhouse",
        target=(
            f"clickhouse://{uri_host}:{output_port}/"
            f"{output_database}/{output_table.split('.', 1)[1]}"
        ),
        binding_id=_CLICKHOUSE_WRITE_BINDING_ID,
        mode="append",
        options={
            "batch_rows": sink_batch_size,
            "concurrency": concurrency,
        },
        runtime_options={
            "user": output_username or "default",
            "credential_ref": f"env://{output_password_env}",
        },
    )
    core_request = InferenceRequest(
        model=model_reference,
        input=IngestionRequest(
            source=source,
            engine="ray",
            binding_id=_CLICKHOUSE_BINDING_ID,
            transforms=transforms,
            read_options=ReadOptions(
                batch_size=_CLICKHOUSE_READ_TASK_ROWS,
                target_split_size_bytes=_CLICKHOUSE_READ_TASK_BYTES,
                concurrency=concurrency,
            ),
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
        target=sink_request,
        request=request,
        entity_column=entity_column,
        feature_columns=features,
        feature_result_names=result_names,
        batch_size=sink_batch_size,
        concurrency=concurrency,
        explanation=explanation,
        model_reference=attribution_model_reference,
    )
    return (
        core_request,
        sink,
        {
            "TRIBUTO_CLICKHOUSE_USER": username,
            "TRIBUTO_CLICKHOUSE_PASSWORD": password,
            output_password_env: output_password,
        },
    )


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
    options = (
        {
            key: value
            for key, value in runtime.items()
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
        if isinstance(runtime, Mapping)
        else {}
    )
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
        200_000,
    )
    if minimum > maximum:
        minimum, maximum = 50_000, 200_000

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
    # A retry must reduce the failed attempt's actual predictor batch even
    # when the original request selected an explicit batch size.
    if batch_size_override is not None:
        predictor_batch = base
    sink_batch = _positive_int(
        options.get("sink_batch_size"),
        "extensions.tributo.inference_runtime.sink_batch_size",
        min(base, minimum, 50_000),
    )
    concurrency = _positive_int(
        execution.get("concurrency", options.get("max_predictor_actors")),
        "execution.concurrency",
        2,
    )
    return predictor_batch, sink_batch, concurrency


def _parameter_type(value: object) -> str:
    if isinstance(value, bool):
        return "UInt8"
    if isinstance(value, int):
        return "UInt64" if value >= 2**63 else "Int64"
    if isinstance(value, float):
        return "Float64"
    if isinstance(value, str):
        return "String"
    _invalid("input.query filter parameters must be scalar JSON values")


def _measure_input_rows(
    request: InferenceRequest,
    credentials: Mapping[str, str],
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
    row_limit: int | None = None
    for index, step in enumerate(request.input.transforms.steps):
        if isinstance(step, Limit):
            row_limit = step.count
            continue
        name = f"filter_{index}"
        column = _identifier(step.column, "input query filter column")
        if isinstance(step, FilterEq):
            operator = "="
        elif isinstance(step, FilterNotEq):
            operator = "!="
        elif isinstance(step, FilterComparison):
            operator = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[step.operator]
        else:
            _invalid("adaptive row counting supports scalar filters only")
        clauses.append(
            f"`{column}` {operator} {{{name}:{_parameter_type(step.value)}}}"
        )
        parameters[name] = step.value
    sql = f"SELECT count() FROM {table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    client = None
    try:
        try:
            client = clickhouse_connect.get_client(
                host=host,
                port=port,
                database=database,
                username=credentials["TRIBUTO_CLICKHOUSE_USER"],
                password=credentials["TRIBUTO_CLICKHOUSE_PASSWORD"],
            )
        except Exception:
            raise _InferenceExecutionError(
                "Input datasource could not be reached",
                code="DATA_SOURCE_UNREACHABLE",
                phase="READ_INPUT",
            ) from None
        schema_rows = client.query(f"DESCRIBE TABLE `{table}`").result_rows
        available = {row[0]: row[1] for row in schema_rows}
        if set(options["columns"]) - set(available):
            _invalid(
                "Input table is missing declared entity, feature, or filter columns",
                code="INPUT_COLUMN_MISSING",
                phase="READ_INPUT",
            )
        for step in request.input.transforms.steps:
            if isinstance(step, (FilterEq, FilterNotEq, FilterComparison)):
                native_type = (
                    available[step.column]
                    .removeprefix("Nullable(")
                    .removesuffix(")")
                    .removeprefix("LowCardinality(")
                    .removesuffix(")")
                )
                is_text = native_type == "String" or native_type.startswith(
                    "FixedString("
                )
                if isinstance(step.value, str) != is_text:
                    _invalid(
                        "Query parameter and source column types require "
                        "unsupported implicit coercion",
                        phase="READ_INPUT",
                    )
        measured = int(client.query(sql, parameters=parameters).first_row[0])
        return min(measured, row_limit) if row_limit is not None else measured
    except (_InferenceExecutionError, _InferenceConfigurationError):
        raise
    except Exception:
        raise _InferenceExecutionError(
            "Input row count query failed",
            code="INPUT_QUERY_FAILED",
            phase="READ_INPUT",
        ) from None
    finally:
        if client is not None:
            client.close()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value) if value.is_finite() else None
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
        model_feature_names: tuple[str, ...] | None = None,
    ) -> None:
        import xgboost

        self._runtime: Any | None = None
        self._feature_columns = feature_columns
        self._model_feature_names = model_feature_names or feature_columns
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
        if model_names and model_names != self._model_feature_names:
            raise ValueError("XGBoost feature names do not match inference columns")

    def __call__(self, frame: Any) -> Any:
        import numpy as np
        import xgboost

        values = frame.loc[:, list(self._feature_columns)].to_numpy(dtype=np.float32)
        matrix = xgboost.DMatrix(
            values,
            feature_names=list(self._model_feature_names),
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
        from botocore.exceptions import ClientError

        try:
            response = boto3.client(
                "s3",
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            ).get_object(Bucket=parsed.netloc, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "NoSuchKey",
                "NoSuchBucket",
                "404",
            }:
                raise FileNotFoundError("Declared S3 artifact was not found") from None
            raise
        body = response["Body"]
        try:
            return bytes(body.read())
        finally:
            body.close()
    path = Path(uri.removeprefix("file://"))
    return path.read_bytes()


def _validate_result_filter(
    model: Mapping[str, Any], output: Mapping[str, Any]
) -> None:
    result_filter = output.get("result_filter")
    if result_filter is None:
        return
    if str(model.get("task_type") or "").upper() not in {
        "BINARY_CLASSIFICATION",
        "MULTICLASS_CLASSIFICATION",
    }:
        _invalid("output.result_filter is only supported for classification")
    result_filter = _mapping(result_filter, "output.result_filter")
    if str(result_filter.get("mode") or "").upper() not in {"INCLUDE", "EXCLUDE"}:
        _invalid("output.result_filter.mode must be INCLUDE or EXCLUDE")
    labels = result_filter.get("labels")
    mapping = model.get("label_mapping") or {}
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or label not in mapping for label in labels
    ):
        _invalid("output.result_filter.labels must belong to model.label_mapping")


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
    decision_threshold = _decision_threshold(task)
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
    if decision_threshold is not None:
        if probability_rows is None:
            _invalid("decision_threshold requires binary probability output")
        positive, positive_label, negative_label = _binary_labels(model)
        try:
            label_values = [
                (
                    positive_label
                    if float(row[positive]) >= decision_threshold
                    else negative_label
                )
                for row in probability_rows
            ]
        except (IndexError, TypeError, ValueError) as exc:
            raise _InferenceConfigurationError(
                "binary probability output does not contain the positive label"
            ) from exc
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
        if decision_threshold is not None:
            extra["decision_threshold"] = decision_threshold
        if explanation is not None:
            if shap_values is None or shap_bases is None or shap_groups is None:
                raise ValueError("TREE_SHAP result columns are missing")
            group = int(shap_groups.iloc[row_index])
            explained_index = (
                _positive_label_index(model)
                if task_type == "BINARY_CLASSIFICATION"
                else group
            )
            # XGBoost binary contributions explain the encoded class-1 margin.
            # Class 0 has the opposite margin: sigmoid(-m) == 1 - sigmoid(m).
            sign = (
                -1.0
                if task_type == "BINARY_CLASSIFICATION" and explained_index == 0
                else 1.0
            )
            explanation_payload: dict[str, Any] = {
                "method": "TREE_SHAP",
                "output_space": "RAW_MARGIN",
                "exactness": ("approximate" if explanation["approximate"] else "exact"),
                "approximate": explanation["approximate"],
                "base_value": sign * float(shap_bases.iloc[row_index]),
                "feature_contributions": [
                    {
                        "feature_name": name,
                        "shap_value": sign * float(value),
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
                _json_value(extra),
                allow_nan=False,
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
            # KnoVa's result table declares ClickHouse DateTime without a
            # timezone. Preserve the UTC instant but emit a timezone-naive
            # Arrow timestamp so the official ClickHouse binding can validate
            # and write the column losslessly.
            "inferred_at": datetime.now(UTC).replace(tzinfo=None),
        }
    )
    output = _mapping(task.get("output"), "output")
    _validate_result_filter(model, output)
    result_filter = output.get("result_filter")
    if isinstance(result_filter, Mapping):
        wanted = {str(value) for value in result_filter["labels"]}
        mode = str(result_filter.get("mode") or "").upper()
        if mode == "INCLUDE":
            result = result[result["pred_label"].isin(wanted)]
        elif mode == "EXCLUDE":
            result = result[~result["pred_label"].isin(wanted)]
        else:
            _invalid("output.result_filter.mode must be INCLUDE or EXCLUDE")
    return result


def _stable_result_batch(frame: Any) -> Any:
    """Emit the one Arrow schema accepted by the ClickHouse result writer.

    Ray can split one ClickHouse read into many small blocks.  Without this
    explicit boundary, a block whose nullable output is entirely null becomes
    Arrow ``null`` while another becomes ``string`` or ``float64``.  The
    ClickHouse provider correctly rejects that schema drift at write time.
    """
    import pyarrow as pa

    missing = [column for column in _RESULT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"inference result is missing columns: {', '.join(missing)}")
    schema = pa.schema(
        (
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("pred_label", pa.string(), nullable=True),
            pa.field("pred_probability", pa.float64(), nullable=True),
            pa.field("pred_probabilities", pa.string(), nullable=True),
            pa.field("pred_value", pa.float64(), nullable=True),
            pa.field("pred_extra", pa.string(), nullable=False),
            pa.field("execution_id", pa.string(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("version_id", pa.string(), nullable=False),
            pa.field("tenant_id", pa.string(), nullable=False),
            pa.field("inferred_at", pa.timestamp("us"), nullable=False),
        )
    )
    return pa.Table.from_pandas(
        frame.loc[:, list(_RESULT_COLUMNS)],
        schema=schema,
        preserve_index=False,
        safe=True,
    )


def _check_control(reporter: Any, phase: str) -> None:
    check = getattr(reporter, "check_control", None)
    if callable(check):
        check(phase)


def _progress_protocol_batch(frame: Any, *, progress_queue: Any, **kwargs: Any) -> Any:
    result = _protocol_batch(frame, **kwargs)
    progress_queue.put(len(frame))
    return result


class _InferenceProgress:
    """Ray workers emit counters; only the driver publishes protocol events."""

    def __init__(self, reporter: Any, total: int, interval: int) -> None:
        from ray.util.queue import Queue

        self.queue = Queue(actor_options={"num_cpus": 0})
        self.reporter, self.total, self.interval = reporter, total, interval
        self.processed = self.reported = 0
        self.error = None
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        from ray.util.queue import Empty

        try:
            while True:
                try:
                    rows = self.queue.get(timeout=0.25)
                except Empty:
                    _check_control(self.reporter, "PREDICT")
                    continue
                if rows is None:
                    break
                self.processed += rows
                if self.processed - self.reported >= self.interval:
                    self.reporter.publish(
                        "PROGRESS",
                        {
                            "processed_rows": self.processed,
                            "result_rows": 0,
                            "total_rows": self.total,
                            "percent": min(
                                99.9,
                                round(100 * self.processed / max(1, self.total), 1),
                            ),
                        },
                        phase="PREDICT",
                    )
                    self.reported = self.processed
        except Exception as exc:
            self.error = exc

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=10)
        self.queue.shutdown()
        if self.thread.is_alive():
            raise _InferenceExecutionError("Progress aggregation did not finish")
        if self.error is not None:
            raise self.error


class _ClickHouseResultSink:
    api_version = 1

    def __init__(
        self,
        *,
        target: DataWriteTargetRequest,
        request: InferenceExecutionRequest,
        entity_column: str,
        feature_columns: tuple[str, ...],
        feature_result_names: tuple[str, ...],
        batch_size: int,
        concurrency: int,
        explanation: Mapping[str, Any] | None,
        model_reference: BundleModelReference | ArtifactModelReference,
    ) -> None:
        self._target = target
        self._task = request.model_dump(mode="python")
        self._entity_column = entity_column
        self._feature_columns = feature_columns
        self._feature_result_names = feature_result_names
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._explanation = dict(explanation) if explanation is not None else None
        self._model_reference = model_reference.model_dump(mode="python")
        self._model_feature_names = feature_columns
        self._reporter = None
        self._progress = None
        self._processed_rows = None
        self._control_error = None
        self._total_rows = 0

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

        _check_control(self._reporter, "PREDICT")
        if self._reporter is not None:
            interval = (
                self._task.get("execution", {}).get("progress_report_interval_rows")
                or 5000
            )
            self._progress = _InferenceProgress(
                self._reporter, self._total_rows, interval
            )
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
                        "model_feature_names": self._model_feature_names,
                    },
                )
            transformed = dataset.map_batches(
                _progress_protocol_batch
                if self._progress is not None
                else _protocol_batch,
                batch_format="pandas",
                max_retries=0,
                batch_size=min(
                    self._batch_size,
                    self._task.get("execution", {}).get("progress_report_interval_rows")
                    or self._batch_size,
                ),
                fn_kwargs={
                    **(
                        {"progress_queue": self._progress.queue}
                        if self._progress is not None
                        else {}
                    ),
                    "task": self._task,
                    "entity_column": self._entity_column,
                    "feature_columns": self._feature_columns,
                    "feature_result_names": self._feature_result_names,
                },
            )
            transformed = transformed.map_batches(
                _stable_result_batch,
                batch_format="pandas",
            )
            transformed = transformed.materialize()
        except Exception as exc:
            raise ResultMaterializationError(type(exc).__name__) from None
        finally:
            if self._progress is not None:
                try:
                    self._progress.close()
                    self._processed_rows = self._progress.processed
                except ExecutionControlError as exc:
                    self._control_error = exc
                    raise
        try:
            _check_control(self._reporter, "WRITE_RESULT")
            if self._reporter is not None:
                self._reporter.publish(
                    "PROGRESS",
                    {
                        "processed_rows": self._processed_rows or 0,
                        "result_rows": 0,
                        "total_rows": self._total_rows,
                        "percent": 99.9,
                    },
                    phase="WRITE_RESULT",
                )
            write_receipt = default_write_gateway().execute(
                self._target.as_write_request(),
                RayDataHandle(transformed),
            )
        except ExecutionControlError as exc:
            self._control_error = exc
            raise
        except Exception as exc:
            raise ResultWriteError(
                f"ClickHouse result write failed ({type(exc).__name__})"
            ) from None
        if write_receipt.rows_written is None:
            raise ResultWriteError("ClickHouse result write returned no row count")
        result_id = hashlib.sha256(
            f"{run_id}|{plan_digest}|{self._target.target}".encode()
        ).hexdigest()
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id=result_id,
            uri=self._target.target,
            rows_written=write_receipt.rows_written,
            metadata={
                "target_kind": "clickhouse",
                "binding_id": write_receipt.binding_id,
                "committed": str(write_receipt.committed).lower(),
            },
        )


def execute_inference(request: InferenceExecutionRequest, reporter: Any) -> Any:
    """Execute one validated request through Tributo's public inference API."""
    configure = getattr(reporter, "configure_timeout", None)
    if callable(configure):
        configure(request.execution.get("max_execution_time_seconds"))
    reporter.phase("LOAD_MODEL")
    _check_control(reporter, "LOAD_MODEL")
    core_request, sink, credentials = _build_request(request)
    core_request = _prepare_model(core_request, sink, request)
    prepared_core, prepared_sink = core_request, sink
    _check_control(reporter, "READ_INPUT")
    measured_rows = _measure_input_rows(core_request, credentials)
    if measured_rows == 0:
        raise _EmptyInferenceInputError(
            "Inference input query returned zero rows",
            code="EMPTY_INPUT",
            phase="READ_INPUT",
        )
    reporter.publish("STARTED", {"total_rows": measured_rows}, phase="READ_INPUT")
    reporter.publish(
        "PROGRESS",
        {
            "processed_rows": 0,
            "result_rows": 0,
            "total_rows": measured_rows,
            "percent": 0.0,
        },
        phase="PREDICT",
    )
    retries = 0
    next_batch = None
    while True:
        core_request, sink, credentials = _build_request(
            request, measured_rows=measured_rows, batch_size_override=next_batch
        )
        core_request = core_request.model_copy(
            update={
                "model": prepared_core.model,
                "input_binding": prepared_core.input_binding,
            }
        )
        sink._feature_columns = prepared_sink._feature_columns
        sink._feature_result_names = prepared_sink._feature_result_names
        sink._model_feature_names = prepared_sink._model_feature_names
        sink._reporter, sink._total_rows = reporter, measured_rows
        _check_control(reporter, "PREDICT")
        try:
            with _runtime_environment(credentials):
                result = run_inference(core_request, bound_sink=sink)
        except ExecutionControlError:
            raise
        except Exception:
            raise _InferenceExecutionError(
                "Inference model resolution failed",
                code="MODEL_LOAD_FAILED",
                phase="LOAD_MODEL",
            ) from None
        if sink._control_error is not None:
            raise sink._control_error
        _check_control(reporter, "WRITE_RESULT")
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
                    "message": "Memory pressure detected; retrying inference "
                    "with batch_size={next_batch}"
                },
                phase="PREDICT",
            )
            continue
        if result.status != "succeeded":
            phase, code = {
                "acquisition": ("READ_INPUT", "INPUT_QUERY_FAILED"),
                "sink": ("WRITE_RESULT", "RESULT_WRITE_FAILED"),
                "materialization": ("PREDICT", "PREDICTION_FAILED"),
            }.get(getattr(failure, "phase", ""), ("PREDICT", "PREDICTION_FAILED"))
            raise _InferenceExecutionError(
                "Tributo inference did not succeed", code=code, phase=phase
            )
        break
    processed = (
        sink._processed_rows if sink._processed_rows is not None else measured_rows
    )
    final = {
        "processed_rows": processed,
        "result_rows": result.output_rows or 0,
        "total_rows": processed,
        "percent": 100.0,
    }
    reporter.publish("PROGRESS", final, phase="WRITE_RESULT")
    reporter.publish(
        "COMPLETED",
        {
            **final,
            "result_summary": {
                "status": result.status,
                "output_rows": result.output_rows,
                "result_uri": result.sink_receipt.uri if result.sink_receipt else None,
            },
            "inference_result": result.model_dump(mode="json"),
        },
        phase="COMPLETED",
    )
    return result


__all__ = ["execute_inference"]
