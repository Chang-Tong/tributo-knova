from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tributo.data import ProviderSourceConfig

from tributo_knova import training
from tributo_knova.protocol import TrainingExecutionRequest

_SECRET = "training-password-must-not-leak"


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
                            "model_feature_name": "spend",
                            "importance_score": 0.7,
                        },
                        {
                            "rank": 2,
                            "model_feature_name": "active_days",
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
        "data_query": {
            "query": {"sql": "SELECT secret_raw_sql FROM forbidden"},
        },
        "features": [
            {
                "result_column": "t0__active_days",
                "origin": {"column_name": "active_days"},
            },
            {
                "result_column": "t1__spend",
                "origin": {"column_name": "spend"},
            },
        ],
        "target": {
            "result_column": "t0__is_churn",
            "origin": {"column_name": "is_churn"},
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
            "prefix": "models/job-1",
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
    monkeypatch.setattr(training, "run_training", dispatcher)

    result = training.execute_training(_request(), reporter)

    assert result is dispatcher.result
    assert len(dispatcher.calls) == 1
    call = dispatcher.calls[0]
    assert call["worker_count"] == 3
    assert call["resources"].num_cpus == 2
    assert call["resources"].num_gpus == 1
    assert call["feature_names"] == ("active_days", "spend")
    assert call["label_name"] == "is_churn"
    assert call["task_type"] == "BINARY_CLASSIFICATION"
    assert call["num_class"] is None
    assert call["evaluation_artifacts"] == {
        "roc_curve": True,
        "threshold_analysis": True,
        "feature_importance": True,
        "correlation_matrix": False,
    }
    assert call["algorithm_config"] == {
        "data": {"label_col": "is_churn"},
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
        },
        "ray": {"storage_path": "/mnt/train"},
        "output": {"bundle_uri": "/mnt/train/models/job-1"},
    }

    ingestion_request = call["ingestion_request"]
    assert ingestion_request.engine == "tributo.ray_data"
    assert ingestion_request.binding_id == "tributo.knova.ray.clickhouse"
    source = ingestion_request.source
    assert isinstance(source, ProviderSourceConfig)
    assert source.provider == "tributo.clickhouse"
    assert source.options["table"] == "analytics.churn_training_features"
    assert source.options["columns"] == ["active_days", "spend", "is_churn"]
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
                "model_feature_name": "active_days",
                "feature_id": "f001",
                "transformation": "PASSTHROUGH",
            },
            {
                "model_feature_index": 1,
                "model_feature_name": "spend",
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
                    "model_feature_name": "spend",
                    "importance_score": 0.7,
                },
                {
                    "rank": 2,
                    "feature_id": "f001",
                    "model_feature_name": "active_days",
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


def test_s3_requires_explicit_ray_storage_and_maps_bundle_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_training", dispatcher)
    storage = {
        "type": "s3",
        "bucket": "knova-models",
        "prefix": "tenant-a/job-1",
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

    storage["properties"] = {}
    with pytest.raises(ValueError, match="ray_storage_path") as captured:
        training.execute_training(_request(storage_context=storage), _Reporter())
    assert _SECRET not in str(captured.value)


def test_completed_event_rejects_mismatched_bundle_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    reporter = _Reporter()
    monkeypatch.setattr(training, "run_training", dispatcher)
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
                "prefix": "tenant-a/job-1",
                "properties": {"ray_storage_path": "/mnt/ray-checkpoints"},
            },
            "s3://knova-models/tenant-a/job-10/bundle-1",
        ),
        (
            {
                "type": "nas",
                "bucket": "/mnt/train",
                "prefix": "models/job-1",
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
    monkeypatch.setattr(training, "run_training", dispatcher)
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
    monkeypatch.setattr(training, "run_training", dispatcher)
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
    monkeypatch.setattr(training, "run_training", dispatcher)
    algorithm = {"algorithm_key": "lightgbm", "hyper_params": {}}

    with pytest.raises(ValueError, match="only algorithm_key=xgboost") as captured:
        training.execute_training(_request(algorithm=algorithm), _Reporter())

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_clickhouse_requires_native_table_and_never_falls_back_to_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }

    with pytest.raises(ValueError, match="native_table") as captured:
        training.execute_training(_request(datasource=datasource), _Reporter())

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_clickhouse_maps_knova_simple_direct_query_to_native_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "run_training", dispatcher)
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
    assert ingestion_request.source.options["table"] == (
        "analytics.churn_training_features"
    )
    assert "sql" not in ingestion_request.source.options


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
    monkeypatch.setattr(training, "run_training", dispatcher)
    datasource = {
        "type": "CLICKHOUSE",
        "host": "clickhouse",
        "database_name": "analytics",
        "password": _SECRET,
        "properties": {},
    }
    tables = [
        {
            "database_name": "analytics",
            "table_name": "churn_training_features",
        }
    ]

    with pytest.raises(ValueError, match="native_table|plain column") as captured:
        training.execute_training(
            _request(
                datasource=datasource,
                tables=tables,
                data_query={"query": {"sql": sql, "params": {}}},
            ),
            _Reporter(),
        )

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_dispatcher_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingDispatcher:
        def __call__(self, **_kwargs: Any) -> None:
            raise RuntimeError(_SECRET)

    monkeypatch.setattr(training, "run_training", _FailingDispatcher())

    with pytest.raises(RuntimeError) as captured:
        training.execute_training(_request(), _Reporter())

    assert _SECRET not in str(captured.value)
