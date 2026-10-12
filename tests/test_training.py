from __future__ import annotations

import runpy
from decimal import Decimal
from pathlib import Path
from traceback import format_exception
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from tributo.data import (
    FilterComparison,
    FilterEq,
    FilterNotEq,
    Limit,
    ProviderSourceConfig,
    RenameColumns,
    SelectColumns,
)

from tributo_knova import training
from tributo_knova.protocol import TrainingExecutionRequest

_SECRET = "training-password-must-not-leak"


def test_string_target_mapping_preserves_feature_order_and_input() -> None:
    batch = pd.DataFrame({"f2": [3.5, 2.0], "label": ["正类", "负类"], "f1": [1, 4]})
    actual = training._encode_target_batch(
        batch, label_name="label", label_mapping={"负类": 0, "正类": 1}, num_class=2
    )
    assert actual["label"].tolist() == [1, 0]
    assert list(actual.columns) == list(batch.columns)
    assert actual["f2"].tolist() == batch["f2"].tolist()
    assert batch["label"].tolist() == ["正类", "负类"]


def test_numeric_targets_are_already_encoded_with_display_mapping() -> None:
    batch = pd.DataFrame({"label": [0, 2, 1]})
    actual = training._encode_target_batch(
        batch,
        label_name="label",
        label_mapping={"甲": 0, "乙": 1, "丙": 2},
        num_class=3,
    )
    assert actual["label"].tolist() == [0, 2, 1]


@pytest.mark.parametrize(
    "values",
    [
        [Decimal("0"), Decimal("1")],
        pd.Categorical([0, 1]),
        pd.Series([0, 1], dtype="Int64"),
    ],
)
def test_encoded_numeric_extension_types(values: object) -> None:
    actual = training._encode_target_batch(
        pd.DataFrame({"label": values}),
        label_name="label",
        label_mapping={"no": 0, "yes": 1},
        num_class=2,
    )
    assert actual["label"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "labels", [["unknown-private-value"], [None], [float("nan")], [2], [0.5]]
)
def test_invalid_target_values_have_safe_error(labels: list[object]) -> None:
    with pytest.raises(training._TrainingConfigurationError) as raised:
        training._encode_target_batch(
            pd.DataFrame({"label": labels}),
            label_name="label",
            label_mapping={"no": 0, "yes": 1},
            num_class=2,
        )
    assert "unknown-private-value" not in str(raised.value)
    assert "target" in str(raised.value)


@pytest.mark.parametrize(
    "mapping", [{"a": 0, "b": 0}, {"a": 1, "b": 2}, {"a": False, "b": True}, {"a": 0}]
)
def test_target_mapping_rejects_invalid_class_codes(mapping: dict[str, int]) -> None:
    with pytest.raises(training._TrainingConfigurationError, match="label_mapping"):
        training._target_encoder(
            {"task_type": "BINARY_CLASSIFICATION", "label_mapping": mapping},
            "label",
            {},
            None,
        )


def test_label_encoding_happens_after_raw_class_sampling() -> None:
    calls = []

    class Dataset:
        def map_batches(self, fn, **kwargs):
            calls.append(("encode", kwargs))
            return self

    dataset = Dataset()

    def sample(value):
        assert value is dataset
        calls.append(("sample", {}))
        return value

    encoder = training._RayLabelEncoder("label", {"no": 0, "yes": 1}, 2, sample)
    assert encoder(dataset) is dataset
    assert [call[0] for call in calls] == ["sample", "encode"]
    assert calls[1][1]["fn_kwargs"]["label_mapping"] == {"no": 0, "yes": 1}


def test_confusion_matrix_restores_original_labels_without_mutation() -> None:
    matrix = {"labels": ["0", "1", "2"], "matrix": [[1, 0, 0], [0, 2, 0], [0, 0, 3]]}
    result = SimpleNamespace(
        execution=SimpleNamespace(
            metrics={"evaluation_details": {"confusion_matrix": matrix}}
        )
    )
    actual = training._evaluation_details(
        result,
        "MULTICLASS_CLASSIFICATION",
        {"label_mapping": {"甲": 0, "乙": 1, "丙": 2}},
    )
    assert actual["confusion_matrix"]["labels"] == ["甲", "乙", "丙"]
    assert actual["confusion_matrix"]["matrix"] == matrix["matrix"]
    assert matrix["labels"] == ["0", "1", "2"]


@pytest.mark.parametrize("positive,expected", [("no", 0), ("yes", 1), (None, 1)])
def test_training_positive_label_is_forwarded(
    positive: str | None, expected: int
) -> None:
    request = _request()
    request.target["positive_label_value"] = positive
    execution = training._build_execution(request)[0]
    assert (
        execution.algorithm_request.algorithm_config["training"]["positive_label"]
        == expected
    )


@pytest.mark.parametrize("positive", ["unknown-private-class", True, [], {}])
def test_training_unknown_positive_label_is_rejected(positive: object) -> None:
    request = _request()
    request.target["positive_label_value"] = positive
    with pytest.raises(
        training._TrainingConfigurationError, match="positive_label_value"
    ) as error:
        training._build_execution(request)
    assert "unknown-private-class" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("missing_value_strategy", "MEAN"),
        ("outlier_strategy", "CLIP"),
        ("scaling_method", "STANDARD"),
        ("encoding_method", "ONE_HOT"),
    ],
)
def test_explicit_unimplemented_preprocessing_is_not_silently_ignored(
    field: str, value: str
) -> None:
    request = _request(feature_engineering={"default_" + field: value})
    with pytest.raises(training._TrainingConfigurationError, match=field):
        training._build_execution(request)
    request.features[0]["treatment"] = {field: "AUTO"}
    request.features[1]["treatment"] = {field: "AUTO"}
    training._build_execution(request)


def test_timeseries_query_is_not_silently_read_as_plain_table() -> None:
    with pytest.raises(
        training._TrainingConfigurationError, match="entity_sampling_query|DIRECT_QUERY"
    ):
        training._build_execution(_request(data_query={"mode": "TIMESERIES_PIVOT"}))


class _Reporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any], str | None]] = []

    def phase(self, phase: str) -> None:
        self.events.append(("PHASE", {}, phase))

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        phase: str | None = None,
    ) -> None:
        self.events.append((event_type, payload, phase))


class _Dispatcher:
    def __init__(self, result: object | None = None) -> None:
        self.result = result or SimpleNamespace(
            execution=SimpleNamespace(
                status="succeeded",
                metrics={
                    "sample_rows": {
                        "total": 2_000,
                        "train": 1_400,
                        "validation": 300,
                        "test": 300,
                    },
                    "evaluation": {
                        "auc": 0.91,
                        "f1": 0.82,
                        "invalid": float("nan"),
                        "flag": True,
                    },
                    "evaluation_details": {
                        "confusion_matrix": {
                            "tp": 120,
                            "fp": 30,
                            "fn": 24,
                            "tn": 126,
                        },
                        "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                        "threshold_analysis": None,
                    },
                    "feature_importance": (
                        {
                            "rank": 1,
                            "model_feature_name": "t0__spend",
                            "importance_score": 0.7,
                        },
                        {
                            "rank": 2,
                            "model_feature_name": "t0__active_days",
                            "importance_score": 0.3,
                        },
                    ),
                },
                outputs={
                    "bundle_id": "bundle-1",
                    "bundle_uri": "/mnt/train/models/job-1/bundle-1",
                    "manifest_sha256": "a" * 64,
                },
            )
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        reporter = kwargs["reporter"]
        reporter.phase("DATA_SPLITTING")
        reporter.phase("EXECUTING")
        reporter.publish(
            "METRICS",
            {
                "current_round": 1,
                "total_rounds": 75,
                "progress_percent": 1.3,
                "metrics": [
                    {"metric_name": "loss", "train": 0.4, "eval": 0.45},
                    {"metric_name": "auc", "train": 0.8, "eval": 0.77},
                ],
            },
            phase="EXECUTING",
        )
        reporter.phase("MATERIALIZING")
        return self.result


def _request(**updates: Any) -> TrainingExecutionRequest:
    payload: dict[str, Any] = {
        "protocol_version": "2.0",
        "job_id": "job-1",
        "model_id": "model-1",
        "version_id": "version-1",
        "tenant_id": "tenant-1",
        "algorithm": {
            "category": "CLASSIFICATION",
            "algorithm_key": "xgboost",
            "hyper_params": {
                "max_depth": 6,
                "learning_rate": 0.08,
                "n_estimators": 75,
            },
        },
        "datasource": {
            "type": "CLICKHOUSE",
            "host": "clickhouse",
            "port": 8123,
            "database_name": "analytics",
            "username": "reader",
            "password": _SECRET,
            "properties": {
                "native_table": "analytics.churn_training_features",
            },
        },
        "tables": [
            {
                "table_alias": "t0",
                "database_name": "analytics",
                "table_name": "churn_training_features",
                "role": "PRIMARY",
            }
        ],
        "data_query": {
            "mode": "DIRECT_QUERY",
            "query": {
                "sql": (
                    "SELECT t0.active_days AS t0__active_days, "
                    "t0.spend AS t0__spend, t0.is_churn AS t0__is_churn "
                    "FROM analytics.churn_training_features t0"
                ),
                "params": {},
            },
        },
        "features": [
            {
                "feature_id": "f001",
                "display_name": "Active days",
                "feature_role": "REGULAR",
                "result_column": "t0__active_days",
                "origin": {
                    "table_alias": "t0",
                    "column_name": "active_days",
                    "column_type": "Int32",
                },
            },
            {
                "feature_id": "f002",
                "display_name": "Spend",
                "feature_role": "REGULAR",
                "result_column": "t0__spend",
                "origin": {
                    "table_alias": "t0",
                    "column_name": "spend",
                    "column_type": "Float64",
                },
            },
        ],
        "target": {
            "display_name": "Churn",
            "result_column": "t0__is_churn",
            "origin": {
                "table_alias": "t0",
                "column_name": "is_churn",
                "column_type": "String",
            },
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"yes": 1, "no": 0},
        },
        "data_split": {
            "strategy": "RANDOM",
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "stratify": False,
            "cross_validation": {"enabled": False},
        },
        "evaluation": {
            "primary_metric": "auc",
            "additional_metrics": [],
            "realtime_metrics": ["auc"],
            "artifacts": {
                "roc_curve": True,
                "threshold_analysis": True,
                "feature_importance": True,
                "correlation_matrix": False,
            },
        },
        "storage_context": {
            "type": "nfs",
            "bucket": "/mnt/train",
            "prefix": "models/job-1/",
            "properties": {},
        },
        "extensions": {
            "tributo": {
                "training_runtime": {
                    "ray": {
                        "num_workers": 3,
                        "cpus_per_worker": 2,
                        "use_gpu": True,
                    }
                }
            }
        },
    }
    # Keep helper overrides concise while supplying unrelated required fields.
    for key in ("algorithm", "datasource", "target"):
        if key in updates:
            updates[key] = {**payload[key], **updates[key]}
    if "target" in updates:
        updates["target"]["origin"] = {
            **payload["target"]["origin"],
            **updates["target"].get("origin", {}),
        }
        if updates["target"].get("task_type") == "REGRESSION":
            payload["algorithm"]["category"] = "REGRESSION"
            payload["algorithm"]["algorithm_key"] = "xgboost_reg"
        if updates["target"]["origin"]["column_name"] != "is_churn":
            payload["data_query"]["query"]["sql"] = payload["data_query"]["query"][
                "sql"
            ].replace(
                "t0.is_churn AS",
                "t0." + updates["target"]["origin"]["column_name"] + " AS",
            )
    payload.update(updates)
    return TrainingExecutionRequest.model_validate(payload)


@pytest.fixture(autouse=True)
def _published_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "bundle_id": "bundle-1",
        "canonical_uri": "/mnt/train/models/job-1/bundle-1",
        "created_at": "2026-09-10T12:00:00Z",
        "input_signature": {
            "input_fields": [{"name": "float_input", "dtype": "float32", "shape": []}]
        },
        "output_signature": {
            "output_fields": [
                {"name": "label", "dtype": "int64", "shape": []},
                {"name": "probabilities", "dtype": "float32", "shape": []},
            ]
        },
        "artifacts": [
            {
                "name": "onnx-model",
                "format": "onnx",
                "files": [
                    {
                        "relative_path": "model.onnx",
                        "sha256": "b" * 64,
                        "size_bytes": 120,
                    }
                ],
            },
            {
                "name": "native-model",
                "format": "ubj",
                "files": [
                    {
                        "relative_path": "model.ubj",
                        "sha256": "c" * 64,
                        "size_bytes": 80,
                    },
                    {
                        "relative_path": "feature_names.json",
                        "sha256": "d" * 64,
                        "size_bytes": 20,
                    },
                ],
            },
        ],
    }
    monkeypatch.setattr(training, "load_bundle", lambda _ref: manifest)


def test_execute_training_maps_knova_to_public_tributo_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    reporter = _Reporter()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)

    result = training.execute_training(_request(), reporter)

    assert result is dispatcher.result
    assert len(dispatcher.calls) == 1
    call = dispatcher.calls[0]
    assert call["worker_count"] == 3
    assert call["resources"].num_cpus == 2
    assert call["resources"].num_gpus == 1
    assert call["feature_names"] == ("t0__active_days", "t0__spend")
    assert call["label_name"] == "t0__is_churn"
    assert call["task_type"] == "BINARY_CLASSIFICATION"
    assert call["num_class"] is None
    assert call["evaluation_artifacts"] == {
        "roc_curve": True,
        "threshold_analysis": True,
        "feature_importance": True,
        "correlation_matrix": False,
    }
    assert call["algorithm_config"] == {
        "data": {"label_col": "t0__is_churn"},
        "model": {
            "max_depth": 6,
            "learning_rate": 0.08,
            "objective": "binary:logistic",
            "eval_metric": ("logloss", "auc"),
        },
        "training": {
            "num_rounds": 75,
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 42,
            "split_strategy": "RANDOM",
            "stratify": False,
            "positive_label": 1,
        },
        "ray": {"storage_path": "/mnt/train"},
        "output": {"bundle_uri": "/mnt/train/models/job-1"},
    }

    ingestion_request = call["ingestion_request"]
    assert ingestion_request.engine == "tributo.ray_data"
    assert ingestion_request.binding_id == "ray_clickhouse.ray.clickhouse"
    source = ingestion_request.source
    assert isinstance(source, ProviderSourceConfig)
    assert source.provider == "tributo.clickhouse"
    assert source.options["table"] == "churn_training_features"
    assert source.options["columns"] == ["active_days", "spend", "is_churn"]
    assert source.options["partitioning"] == {"mode": "single"}
    assert "password" not in source.options
    assert "sql" not in source.options
    assert "secret_raw_sql" not in repr(ingestion_request)
    assert _SECRET not in source.uri

    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "PHASE",
        "PHASE",
        "METRICS",
        "PHASE",
        "COMPLETED",
    ]
    assert reporter.events[0][2] == "PREPARING"
    assert reporter.events[3][1]["total_rounds"] == 75
    assert reporter.events[3][1]["metrics"]
    completed = reporter.events[5][1]
    assert completed["training_result"] == {
        "algorithm_key": "xgboost",
        "task_type": "BINARY_CLASSIFICATION",
        "model_features": [
            {
                "model_feature_index": 0,
                "model_feature_name": "t0__active_days",
                "feature_id": "f001",
                "transformation": "PASSTHROUGH",
            },
            {
                "model_feature_index": 1,
                "model_feature_name": "t0__spend",
                "feature_id": "f002",
                "transformation": "PASSTHROUGH",
            },
        ],
        "evaluation": {
            "eval_type": "BINARY_CLASSIFICATION",
            "sample_rows": 300,
            "metrics": [
                {"metric_name": "auc", "value": 0.91},
                {"metric_name": "f1", "value": 0.82},
            ],
            "details": {
                "confusion_matrix": {
                    "tp": 120,
                    "fp": 30,
                    "fn": 24,
                    "tn": 126,
                },
                "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
                "threshold_analysis": None,
            },
        },
        "feature_analysis": {
            "importance_ranking": [
                {
                    "rank": 1,
                    "feature_id": "f002",
                    "model_feature_name": "t0__spend",
                    "importance_score": 0.7,
                },
                {
                    "rank": 2,
                    "feature_id": "f001",
                    "model_feature_name": "t0__active_days",
                    "importance_score": 0.3,
                },
            ],
            "correlation_matrix": None,
        },
        "tuning_result": None,
    }
    assert completed["result_summary"] == {
        "primary_metric": {"name": "auc", "value": 0.91},
        "sample_rows": {
            "total": 2_000,
            "train": 1_400,
            "validation": 300,
            "test": 300,
        },
    }
    artifact_manifest = completed["artifact_manifest"]
    assert artifact_manifest["model_id"] == "model-1"
    assert artifact_manifest["version_id"] == "version-1"
    assert artifact_manifest["storage"] == {
        "type": "nas",
        "bucket": "/mnt/train",
        "prefix": "models/job-1/bundle-1/",
        "properties": {},
    }
    alternatives = artifact_manifest["model_artifacts"]["model_weights"]["alternatives"]
    assert [alternative["format"] for alternative in alternatives] == [
        "onnx",
        "xgboost",
    ]
    assert alternatives[0]["files"][0]["path"] == ("artifacts/onnx-model/model.onnx")
    assert alternatives[1]["files"][0]["path"] == ("artifacts/native-model/model.ubj")
    assert artifact_manifest["total_size_bytes"] == 220


def test_default_data_split_keeps_validation_and_test_sets() -> None:
    assert training._data_split_config({}) == {
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "split_strategy": "RANDOM",
        "stratify": False,
    }


def test_s3_maps_explicit_ray_storage_and_bundle_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    storage = {
        "type": "s3",
        "bucket": "knova-models",
        "prefix": "tenant-a/job-1/",
        "properties": {"ray_storage_path": "/mnt/ray-checkpoints"},
    }

    monkeypatch.setattr(
        training,
        "load_bundle",
        lambda _ref: {
            "bundle_id": "bundle-1",
            "canonical_uri": "s3://knova-models/tenant-a/job-1/bundle-1",
            "created_at": "2026-09-10T12:00:00Z",
            "input_signature": {},
            "output_signature": {},
            "artifacts": [
                {
                    "name": "onnx-model",
                    "format": "onnx",
                    "files": [
                        {
                            "relative_path": "model.onnx",
                            "sha256": "b" * 64,
                            "size_bytes": 10,
                        }
                    ],
                },
                {
                    "name": "native-model",
                    "format": "ubj",
                    "files": [
                        {
                            "relative_path": "model.ubj",
                            "sha256": "c" * 64,
                            "size_bytes": 10,
                        }
                    ],
                },
            ],
        },
    )
    dispatcher.result.execution.outputs["bundle_uri"] = (
        "s3://knova-models/tenant-a/job-1/bundle-1"
    )

    reporter = _Reporter()
    training.execute_training(_request(storage_context=storage), reporter)

    config = dispatcher.calls[0]["algorithm_config"]
    assert config["ray"] == {"storage_path": "/mnt/ray-checkpoints"}
    assert config["output"] == {"bundle_uri": "s3://knova-models/tenant-a/job-1"}
    assert reporter.events[-1][1]["artifact_manifest"]["storage"] == {
        "type": "s3",
        "bucket": "knova-models",
        "prefix": "tenant-a/job-1/bundle-1/",
        "properties": {},
    }


def test_s3_defaults_ray_storage_below_requested_prefix() -> None:
    storage = {
        "type": "s3",
        "bucket": "knova-models",
        "prefix": "tenant-a/job-1/",
        "properties": {},
    }

    assert training._storage_config({"storage_context": storage}) == (
        "s3://knova-models/tenant-a/job-1/_ray-checkpoints",
        "s3://knova-models/tenant-a/job-1",
    )


def test_completed_event_rejects_mismatched_bundle_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    reporter = _Reporter()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    monkeypatch.setattr(
        training,
        "load_bundle",
        lambda _ref: {
            "bundle_id": "a-different-bundle",
            "canonical_uri": "/mnt/train/models/job-1/bundle-1",
            "artifacts": [],
        },
    )

    with pytest.raises(
        RuntimeError, match="published Bundle identity validation failed"
    ):
        training.execute_training(_request(), reporter)

    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "PHASE",
        "PHASE",
        "METRICS",
        "PHASE",
    ]


@pytest.mark.parametrize(
    ("storage", "canonical_uri"),
    [
        (
            {
                "type": "s3",
                "bucket": "knova-models",
                "prefix": "tenant-a/job-1/",
                "properties": {"ray_storage_path": "/mnt/ray-checkpoints"},
            },
            "s3://knova-models/tenant-a/job-10/bundle-1",
        ),
        (
            {
                "type": "nas",
                "bucket": "/mnt/train",
                "prefix": "models/job-1/",
                "properties": {},
            },
            "/mnt/train/models/job-10/bundle-1",
        ),
    ],
)
def test_artifact_storage_rejects_sibling_prefixes(
    storage: dict[str, Any], canonical_uri: str
) -> None:
    with pytest.raises(RuntimeError, match="outside the requested"):
        training._artifact_storage(
            _request(storage_context=storage),
            canonical_uri,
        )


def test_completed_event_maps_real_primary_metric_without_fabrication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    dispatcher.result.execution.metrics["evaluation"] = {
        "auc": 0.91,
        "loss": 0.2,
        "invalid": float("nan"),
        "flag": True,
    }
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    reporter = _Reporter()

    training.execute_training(
        _request(evaluation={"primary_metric": "AUC"}),
        reporter,
    )

    completed = reporter.events[-1][1]
    assert completed["result_summary"]["primary_metric"] == {
        "name": "auc",
        "value": 0.91,
    }
    assert completed["training_result"]["evaluation"]["metrics"] == [
        {"metric_name": "auc", "value": 0.91},
        {"metric_name": "loss", "value": 0.2},
    ]


@pytest.mark.parametrize(
    ("task_type", "label_mapping", "expected"),
    [
        (
            "BINARY_CLASSIFICATION",
            {"yes": 1, "no": 0},
            {"objective": "binary:logistic"},
        ),
        (
            "MULTICLASS_CLASSIFICATION",
            {"red": 0, "green": 1, "blue": 2},
            {"objective": "multi:softprob", "num_class": 3},
        ),
        ("REGRESSION", None, {"objective": "reg:squarederror"}),
    ],
)
def test_task_type_and_label_mapping_select_xgboost_objective(
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    label_mapping: dict[str, int] | None,
    expected: dict[str, Any],
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    target = {
        "origin": {"column_name": "label"},
        "task_type": task_type,
        "label_mapping": label_mapping,
    }

    training.execute_training(_request(target=target), _Reporter())

    model = dispatcher.calls[0]["algorithm_config"]["model"]
    assert {key: model[key] for key in expected} == expected


def test_unsupported_algorithm_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    algorithm = {"algorithm_key": "lightgbm", "hyper_params": {}}

    with pytest.raises(ValueError, match="only algorithm_key=xgboost") as captured:
        training.execute_training(_request(algorithm=algorithm), _Reporter())

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_clickhouse_requires_native_table_and_never_falls_back_to_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }

    with pytest.raises(ValueError, match="tables") as captured:
        training.execute_training(
            _request(datasource=datasource, tables=[]), _Reporter()
        )

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_clickhouse_maps_knova_simple_direct_query_to_native_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }
    tables = [
        {
            "table_alias": "t0",
            "database_name": "analytics",
            "table_name": "churn_training_features",
            "role": "PRIMARY",
        }
    ]
    data_query = {
        "mode": "DIRECT_QUERY",
        "query": {
            "sql": (
                "SELECT t0.active_days AS t0__active_days, "
                "t0.spend AS t0__spend, t0.is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features AS t0"
            ),
            "params": {},
        },
    }

    training.execute_training(
        _request(datasource=datasource, tables=tables, data_query=data_query),
        _Reporter(),
    )

    ingestion_request = dispatcher.calls[0]["ingestion_request"]
    assert ingestion_request.source.options["table"] == "churn_training_features"
    assert "sql" not in ingestion_request.source.options


def _filtered_training_request(
    suffix: str,
    params: dict[str, str],
    *,
    native_table: bool = False,
) -> TrainingExecutionRequest:
    payload = _request().model_dump(mode="python")
    if not native_table:
        payload["datasource"]["properties"] = {}
    payload["tables"] = [
        {
            "table_alias": "t0",
            "database_name": "analytics",
            "table_name": "churn_training_features",
            "role": "PRIMARY",
        }
    ]
    payload["data_query"] = {
        "mode": "DIRECT_QUERY",
        "query": {
            "sql": (
                "SELECT t0.active_days AS t0__active_days, t0.spend AS t0__spend, "
                "t0.is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features AS t0 " + suffix
            ),
            "params": params,
        },
    }
    return TrainingExecutionRequest.model_validate(payload)


@pytest.mark.parametrize("native_table", [False, True])
def test_clickhouse_preserves_parameterized_filter_on_non_feature_column(
    monkeypatch: pytest.MonkeyPatch,
    native_table: bool,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    # Same query shape as the browser's sourceTable user_id <= 10000 filter.
    request = _filtered_training_request(
        "WHERE t0.user_id <= {p0:UInt64} LIMIT 834",
        {"p0": "10000"},
        native_table=native_table,
    )
    training.execute_training(request, _Reporter())

    ingestion = dispatcher.calls[0]["ingestion_request"]
    assert ingestion.source.options["columns"] == [
        "active_days",
        "spend",
        "is_churn",
        "user_id",
    ]
    assert ingestion.transforms.steps[:-1] == (
        FilterComparison(column="user_id", operator="lte", value=10000),
        Limit(count=834),
        SelectColumns(columns=("active_days", "spend", "is_churn")),
    )
    assert "sql" not in ingestion.source.options
    assert "params" not in ingestion.source.options


@pytest.mark.parametrize(
    "operator,transform",
    [
        ("=", FilterEq(column="user_id", value=10000)),
        ("!=", FilterNotEq(column="user_id", value=10000)),
        ("<>", FilterNotEq(column="user_id", value=10000)),
        ("<", FilterComparison(column="user_id", operator="lt", value=10000)),
        (">", FilterComparison(column="user_id", operator="gt", value=10000)),
        (">=", FilterComparison(column="user_id", operator="gte", value=10000)),
    ],
)
def test_clickhouse_preserves_all_scalar_conditions_and_literal_values(
    monkeypatch: pytest.MonkeyPatch,
    operator: str,
    transform: object,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    literal = "not SQL'); DROP TABLE training; --"
    request = _filtered_training_request(
        f"WHERE (t0.user_id {operator} {{p0:UInt64}}) AND t0.segment = {{p1:String}}",
        {"p0": "10000", "p1": literal},
    )
    training.execute_training(request, _Reporter())
    ingestion = dispatcher.calls[0]["ingestion_request"]
    assert ingestion.transforms.steps[:2] == (
        transform,
        FilterEq(column="segment", value=literal),
    )
    assert literal not in str(ingestion.source.options)


@pytest.mark.parametrize(
    "suffix,params",
    [
        ("WHERE t0.user_id <= {p0:UInt64}", {}),
        ("WHERE t0.user_id <= {p0:UInt64}", {"p0": "10000", "unused": "1"}),
        ("WHERE t0.user_id <= {p0:UInt64}", {"p0": _SECRET}),
        ("WHERE t0.user_id <= {p0:Unsupported}", {"p0": "10000"}),
        ("WHERE t1.user_id <= {p0:UInt64}", {"p0": "10000"}),
        ("WHERE t0.user_id <= {p0:UInt64} OR 1=1", {"p0": "10000"}),
        ("WHERE t0.user_id <= {p0:UInt64} ORDER BY t0.user_id", {"p0": "10000"}),
        ("WHERE t0.user_id <= {p0:UInt64}; SELECT 1", {"p0": "10000"}),
        ("WHERE t0.user_id <= {p0:UInt64} -- ignored", {"p0": "10000"}),
        ("WHERE (t0.user_id <= {p0:UInt64}", {"p0": "10000"}),
        ("WHERE t0.user_id <= {p0:UInt64} LIMIT 1, 2", {"p0": "10000"}),
        ("JOIN analytics.other AS t1 ON t0.user_id=t1.user_id", {}),
    ],
)
@pytest.mark.parametrize("native_table", [False, True])
def test_clickhouse_rejects_unsupported_filters_without_dropping_them(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    params: dict[str, str],
    native_table: bool,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    with pytest.raises(ValueError) as error:
        training.execute_training(
            _filtered_training_request(suffix, params, native_table=native_table),
            _Reporter(),
        )
    assert dispatcher.calls == []
    assert _SECRET not in str(error.value)


def test_clickhouse_sampling_error_does_not_reflect_untrusted_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    with pytest.raises(ValueError, match="strategy.*unsupported") as error:
        training.execute_training(
            _request(data_sampling={"class_balance": {"strategy": _SECRET}}),
            _Reporter(),
        )
    assert _SECRET not in str(error.value)
    assert dispatcher.calls == []


def test_clickhouse_accepts_unqualified_native_table_with_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    payload = _filtered_training_request(
        "WHERE t0.user_id <= {p0:UInt64}",
        {"p0": "10000"},
        native_table=True,
    ).model_dump(mode="python")
    payload["datasource"]["properties"]["native_table"] = "churn_training_features"
    training.execute_training(
        TrainingExecutionRequest.model_validate(payload), _Reporter()
    )
    assert dispatcher.calls[0]["ingestion_request"].transforms.steps[0] == (
        FilterComparison(column="user_id", operator="lte", value=10000)
    )


def test_clickhouse_sampling_error_does_not_reflect_class_amount_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    with pytest.raises(ValueError, match="class_amounts value") as error:
        training.execute_training(
            _request(
                data_sampling={
                    "class_balance": {
                        "strategy": "CUSTOM_AMOUNT",
                        "class_amounts": {_SECRET: 0},
                    }
                }
            ),
            _Reporter(),
        )
    assert _SECRET not in str(error.value)
    assert dispatcher.calls == []


@pytest.mark.parametrize(
    "suffix",
    [
        "HAVING active_days > 0",
        "QUALIFY active_days > 0",
        "GROUP BY active_days",
        "WHERE active_days > 0",
        "ORDER BY active_days",
        "LIMIT 10",
    ],
)
def test_clickhouse_legacy_native_table_cannot_bypass_query_conditions(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    with pytest.raises(ValueError, match="tables"):
        training.execute_training(
            _request(
                tables=[],
                data_query={
                    "mode": "DIRECT_QUERY",
                    "query": {
                        "sql": "SELECT active_days FROM features " + suffix,
                    },
                },
            ),
            _Reporter(),
        )
    assert dispatcher.calls == []


def test_clickhouse_maps_custom_class_amounts_to_ray_dataset_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }
    tables = [
        {
            "table_alias": "t0",
            "role": "PRIMARY",
            "database_name": "analytics",
            "table_name": "churn_training_features",
        }
    ]
    data_query = {
        "mode": "DIRECT_QUERY",
        "query": {
            "sql": (
                "(SELECT active_days AS t0__active_days, spend AS t0__spend, "
                "is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features "
                "WHERE is_churn = {p0:UInt8} LIMIT 80)\n"
                "UNION ALL\n"
                "(SELECT active_days AS t0__active_days, spend AS t0__spend, "
                "is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features "
                "WHERE is_churn = {p1:UInt8} LIMIT 20)"
            ),
            "params": {"p0": "0", "p1": "1"},
        },
    }
    data_sampling = {
        "class_balance": {
            "strategy": "CUSTOM_AMOUNT",
            "class_amounts": {"0": 80, "1": 20},
        },
        "random_seed": 42,
    }

    training.execute_training(
        _request(
            datasource=datasource,
            tables=tables,
            data_query=data_query,
            data_sampling=data_sampling,
        ),
        _Reporter(),
    )

    call = dispatcher.calls[0]
    ingestion_request = call["ingestion_request"]
    assert ingestion_request.source.options["table"] == "churn_training_features"
    transform = call["ray_dataset_transform"]
    assert isinstance(transform, training._RayInputValidation)
    encoder = transform.transform
    assert isinstance(encoder, training._RayLabelEncoder)
    sampler = encoder.sampler
    assert isinstance(sampler, training._RayClassAmountSampler)
    assert sampler.label_name == "t0__is_churn"
    assert sampler.limits == ((0, 80), (1, 20))


def test_clickhouse_rejects_custom_class_limit_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }
    tables = [
        {
            "table_alias": "t0",
            "role": "PRIMARY",
            "database_name": "analytics",
            "table_name": "churn_training_features",
        }
    ]
    data_query = {
        "mode": "DIRECT_QUERY",
        "query": {
            "sql": (
                "(SELECT active_days AS t0__active_days, spend AS t0__spend, "
                "is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features "
                "WHERE is_churn = {p0:UInt8} LIMIT 79)\n"
                "UNION ALL\n"
                "(SELECT active_days AS t0__active_days, spend AS t0__spend, "
                "is_churn AS t0__is_churn "
                "FROM analytics.churn_training_features "
                "WHERE is_churn = {p1:UInt8} LIMIT 20)"
            ),
            "params": {"p0": "0", "p1": "1"},
        },
    }
    data_sampling = {
        "class_balance": {
            "strategy": "CUSTOM_AMOUNT",
            "class_amounts": {"0": 80, "1": 20},
        }
    }

    with pytest.raises(ValueError, match="LIMIT does not match class_amounts"):
        training.execute_training(
            _request(
                datasource=datasource,
                tables=tables,
                data_query=data_query,
                data_sampling=data_sampling,
            ),
            _Reporter(),
        )

    assert dispatcher.calls == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM analytics.churn_training_features WHERE active_days > 1",
        (
            "SELECT * FROM analytics.churn_training_features "
            "JOIN analytics.other USING (id)"
        ),
        (
            "SELECT active_days + 1 AS t0__active_days, spend AS t0__spend, "
            "is_churn AS t0__is_churn FROM analytics.churn_training_features"
        ),
    ],
)
def test_clickhouse_rejects_nontrivial_direct_query_without_native_table(
    monkeypatch: pytest.MonkeyPatch, sql: str
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }
    tables = [
        {
            "table_alias": "t0",
            "role": "PRIMARY",
            "database_name": "analytics",
            "table_name": "churn_training_features",
        }
    ]

    with pytest.raises(ValueError, match="native_table|plain column") as captured:
        training.execute_training(
            _request(
                datasource=datasource,
                tables=tables,
                data_query={
                    "mode": "DIRECT_QUERY",
                    "query": {"sql": sql, "params": {}},
                },
            ),
            _Reporter(),
        )

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


@pytest.mark.parametrize(
    ("error", "summary"),
    [
        (
            ValueError("training dataset is empty"),
            "Training input is empty after filtering",
        ),
        (
            ValueError("Couldn't create non-empty splits with the given proportions."),
            "Training input is too small for the configured "
            "train/validation/test split",
        ),
        (
            ValueError("training dataset is empty " + _SECRET),
            "XGBoost training execution failed (ValueError)",
        ),
        (
            RuntimeError("training dataset is empty"),
            "XGBoost training execution failed (RuntimeError)",
        ),
    ],
)
def test_known_data_failures_have_fixed_safe_reasons(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    summary: str,
) -> None:
    def fail(**_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(training, "run_xgboost_training", fail)
    with pytest.raises(training._TrainingExecutionError) as captured:
        training.execute_training(_request(), _Reporter())
    assert str(captured.value) == summary
    assert _SECRET not in "".join(format_exception(captured.value))


def test_dispatcher_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingDispatcher:
        def __call__(self, **_kwargs: Any) -> None:
            raise RuntimeError(_SECRET)

    monkeypatch.setattr(training, "run_xgboost_training", _FailingDispatcher())
    monkeypatch.setattr(training.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(training.logger, "propagate", False)

    with pytest.raises(RuntimeError) as captured:
        training.execute_training(_request(), _Reporter())

    assert _SECRET not in str(captured.value)
    assert _SECRET not in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "test_training.py" in caplog.text
    assert "__call__" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    ("target", "error_type"),
    [
        ("_build_execution", RuntimeError),
        ("_build_execution", training._TrainingConfigurationError),
        ("run_xgboost_training", training._TrainingExecutionError),
    ],
)
def test_training_failure_logs_do_not_render_exception_chains(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    target: str,
    error_type: type[Exception],
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        try:
            raise RuntimeError(_SECRET)
        except RuntimeError as cause:
            raise error_type("safe outer failure") from cause

    monkeypatch.setattr(training, target, fail)
    monkeypatch.setattr(training.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(training.logger, "propagate", False)

    with pytest.raises((ValueError, RuntimeError)) as captured:
        training.execute_training(_request(), _Reporter())

    assert _SECRET not in caplog.text
    assert _SECRET not in "".join(format_exception(captured.value))
    assert "Traceback (most recent call last)" in caplog.text
    assert "test_training.py" in caplog.text
    assert "in fail" in caplog.text
    assert error_type.__name__ in caplog.text
    assert "operation_id=job-1" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_training_failure_logs_do_not_include_source_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    module = tmp_path / "failing_training.py"
    module.write_text(
        f"def fail(**kwargs):\n    raise RuntimeError({_SECRET!r})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training, "run_xgboost_training", runpy.run_path(module)["fail"]
    )
    monkeypatch.setattr(training.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(training.logger, "propagate", False)

    with pytest.raises(RuntimeError):
        training.execute_training(_request(), _Reporter())

    assert _SECRET not in caplog.text
    assert "failing_training.py" in caplog.text
    assert "line 2, in fail" in caplog.text


def test_protocol_patience_and_timeout_are_mapped_with_precedence():
    request = _request(
        resource_limits={"early_stopping_patience": 3, "max_training_time_seconds": 7}
    )
    request.algorithm["hyper_params"]["early_stopping_rounds"] = 20
    config = training._build_execution(request)[0].algorithm_request.algorithm_config[
        "training"
    ]
    assert config["early_stopping_rounds"] == 3
    assert config["max_training_time_seconds"] == 7


def test_credential_reference_overrides_inline_password(monkeypatch):
    monkeypatch.setenv("KNOVA_TEST_CREDENTIAL", "reference-only-fixture")
    request = _request()
    request.datasource["credential_ref"] = "env://KNOVA_TEST_CREDENTIAL"
    assert training._build_execution(request)[4][1] == "reference-only-fixture"


def test_query_projection_reordering_keeps_opaque_model_feature_names():
    request = _request()
    request.data_query["query"]["sql"] = (
        "SELECT t0.is_churn AS t0__is_churn, t0.spend AS t0__spend, "
        "t0.active_days AS t0__active_days FROM analytics.churn_training_features t0"
    )
    execution, inputs, *_ = training._build_execution(request)
    assert execution.algorithm_request.input_binding.feature_names == (
        "t0__active_days",
        "t0__spend",
    )
    transform = inputs.values[training._INPUT_REFERENCE].request.transforms.steps[-1]
    assert isinstance(transform, RenameColumns)
    assert [(r.source, r.target) for r in transform.renames] == [
        ("active_days", "t0__active_days"),
        ("spend", "t0__spend"),
        ("is_churn", "t0__is_churn"),
    ]


def test_wrong_sql_result_alias_is_rejected():
    request = _request()
    request.data_query["query"]["sql"] = request.data_query["query"]["sql"].replace(
        "AS t0__spend", "AS wrong_alias"
    )
    with pytest.raises(
        training._TrainingConfigurationError, match="columns do not match"
    ):
        training._build_execution(request)


def test_entity_column_is_required_in_query_and_native_read():
    request = _request()
    request.data_query["entity_key"] = {
        "origin": {
            "table_alias": "t0",
            "column_name": "user_id",
            "column_type": "UInt64",
        },
        "result_column": "opaque_entity",
    }
    with pytest.raises(
        training._TrainingConfigurationError, match="columns do not match"
    ):
        training._build_execution(request)
    request.data_query["query"]["sql"] = request.data_query["query"]["sql"].replace(
        "SELECT ", "SELECT t0.user_id AS opaque_entity, "
    )
    plan = training._build_execution(request)
    assert plan[1].values[training._INPUT_REFERENCE].request.source.options[
        "columns"
    ] == ["active_days", "spend", "is_churn", "user_id"]
    assert isinstance(plan[-1], training._RayInputValidation)


def test_missing_loaded_result_column_uses_data_error():
    transform = training._build_execution(_request())[-1]
    dataset = SimpleNamespace(schema=lambda: SimpleNamespace(names=["wrong_column"]))
    with pytest.raises(training._TrainingDataError, match="t0__active_days") as caught:
        transform(dataset)
    assert caught.value.code == "DATA_QUERY_FAILED"
    assert caught.value.phase == "LOADING_DATA"


def test_auto_tuning_falls_back_with_protocol_warning(monkeypatch):
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_xgboost_training", dispatcher)
    reporter = _Reporter()
    training.execute_training(
        _request(
            tuning={
                "mode": "AUTO",
                "auto_config": {
                    "sampler": {"type": "RANDOM"},
                    "budget": {"max_trials": 2},
                    "search_space": {"max_depth": {"type": "INT", "min": 2, "max": 5}},
                },
            }
        ),
        reporter,
    )
    warnings = [payload for kind, payload, _ in reporter.events if kind == "LOG"]
    assert len(warnings) == 1 and warnings[0]["level"] == "warning"
    assert "MANUAL" in warnings[0]["message"]
    assert len(dispatcher.calls) == 1
    assert reporter.events[-1][1]["training_result"]["tuning_result"] is None


@pytest.mark.parametrize(
    "native_phase,error,code,phase",
    [
        ("PREPARING", RuntimeError(_SECRET), "DATA_QUERY_FAILED", "LOADING_DATA"),
        (
            "PREPARING",
            ConnectionError(_SECRET),
            "DATA_SOURCE_UNREACHABLE",
            "LOADING_DATA",
        ),
        (
            "DATA_SPLITTING",
            RuntimeError(_SECRET),
            "DATA_QUERY_FAILED",
            "DATA_SPLITTING",
        ),
        (
            "FEATURE_ENGINEERING",
            RuntimeError(_SECRET),
            "FEATURE_ENGINEERING_FAILED",
            "FEATURE_ENGINEERING",
        ),
        ("EXECUTING", RuntimeError(_SECRET), "TRAINING_FAILED", "TRAINING"),
        ("MATERIALIZING", RuntimeError(_SECRET), "EVALUATION_FAILED", "EVALUATING"),
    ],
)
def test_training_errors_keep_native_lifecycle_phase(
    monkeypatch, native_phase, error, code, phase
):
    reporter = _Reporter()

    def fail(**kwargs):
        kwargs["reporter"].phase(native_phase)
        # The real Redis reporter persists this protocol phase for driver errors.
        kwargs["reporter"]._current_phase = phase
        raise error

    monkeypatch.setattr(training, "run_xgboost_training", fail)
    with pytest.raises(training._TrainingExecutionError) as caught:
        training.execute_training(_request(), reporter)
    assert caught.value.code == code
    assert caught.value.phase == phase
    assert _SECRET not in "".join(format_exception(caught.value))


def test_published_artifact_validation_failure_is_model_export_failed(monkeypatch):
    monkeypatch.setattr(training, "run_xgboost_training", _Dispatcher())
    monkeypatch.setattr(training, "load_bundle", lambda _ref: {"bundle_id": "wrong"})
    with pytest.raises(training._TrainingExecutionError) as caught:
        training.execute_training(_request(), _Reporter())
    assert caught.value.code == "MODEL_EXPORT_FAILED"
    assert caught.value.phase == "EVALUATING"


def test_empty_training_input_has_specific_protocol_code(monkeypatch):
    def empty(**_kwargs):
        raise ValueError("training dataset is empty")

    monkeypatch.setattr(training, "run_xgboost_training", empty)
    with pytest.raises(training._TrainingExecutionError) as caught:
        training.execute_training(_request(), _Reporter())
    assert caught.value.code == "EMPTY_DATASET"
    assert caught.value.phase == "LOADING_DATA"


@pytest.mark.parametrize("wrapped", [False, True])
def test_lazy_source_failure_remains_data_query_error_after_training_starts(
    monkeypatch, wrapped
):
    from tributo.exceptions import DataSourceError

    reporter = _Reporter()

    def fail(**kwargs):
        kwargs["reporter"]._current_phase = "TRAINING"
        try:
            raise DataSourceError(_SECRET)
        except DataSourceError as cause:
            if wrapped:
                raise RuntimeError("safe wrapper") from cause
            raise

    monkeypatch.setattr(training, "run_xgboost_training", fail)
    with pytest.raises(training._TrainingExecutionError) as caught:
        training.execute_training(_request(), reporter)
    assert caught.value.code == "DATA_QUERY_FAILED"
    assert caught.value.phase == "LOADING_DATA"
    assert _SECRET not in "".join(format_exception(caught.value))
