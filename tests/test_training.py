from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tributo.algorithms import (
    AlgorithmOperation,
    ExecutionProfile,
    ExecutionRequest,
)
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.data import ProviderSourceConfig
from tributo.integrations.algorithm_inputs import IngestionInputInvocation

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
                outputs={
                    "bundle_id": "bundle-1",
                    "bundle_uri": "/mnt/train/models/job-1",
                },
            )
        )
        self.calls: list[
            tuple[ExecutionRequest, InputExecutionContext, InputResolutionContext]
        ] = []

    def execute(
        self,
        request: ExecutionRequest,
        context: InputExecutionContext,
        *,
        resolution_context: InputResolutionContext,
    ) -> object:
        self.calls.append((request, context, resolution_context))
        return self.result


def _request(**updates: Any) -> TrainingExecutionRequest:
    payload: dict[str, Any] = {
        "protocol_version": "2.0",
        "job_id": "job-1",
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


def test_execute_training_maps_knova_to_public_tributo_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    reporter = _Reporter()
    monkeypatch.setattr(training, "build_algorithm_dispatcher", lambda: dispatcher)

    result = training.execute_training(_request(), reporter)

    assert result is dispatcher.result
    assert len(dispatcher.calls) == 1
    execution, input_context, resolution_context = dispatcher.calls[0]
    assert execution.profile is ExecutionProfile.CLUSTER
    assert execution.worker_count == 3
    assert execution.resources_per_worker is not None
    assert execution.resources_per_worker.num_cpus == 2
    assert execution.resources_per_worker.num_gpus == 1

    algorithm = execution.algorithm_request
    assert algorithm.algorithm == "xgboost"
    assert algorithm.operation is AlgorithmOperation.FIT
    assert algorithm.implementation_id is None
    assert algorithm.input_binding.feature_names == ("active_days", "spend")
    assert algorithm.input_binding.label_name == "is_churn"
    assert algorithm.algorithm_config == {
        "data": {"label_col": "is_churn"},
        "model": {
            "max_depth": 6,
            "learning_rate": 0.08,
            "objective": "binary:logistic",
        },
        "training": {"num_rounds": 75},
        "ray": {"storage_path": "/mnt/train"},
        "output": {"bundle_uri": "/mnt/train/models/job-1"},
    }

    invocation = next(iter(input_context.values.values()))
    assert isinstance(invocation, IngestionInputInvocation)
    assert next(iter(resolution_context.values.values())) is invocation
    assert invocation.handle_adapter_id is None
    assert invocation.request.engine == "tributo.ray_data"
    assert invocation.request.binding_id == "tributo.knova.ray.clickhouse"
    source = invocation.request.source
    assert isinstance(source, ProviderSourceConfig)
    assert source.provider == "tributo.clickhouse"
    assert source.options["table"] == "analytics.churn_training_features"
    assert source.options["columns"] == ["active_days", "spend", "is_churn"]
    assert "sql" not in source.options
    assert "secret_raw_sql" not in repr(invocation)
    assert _SECRET not in source.uri

    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "METRICS",
        "COMPLETED",
    ]
    assert reporter.events[0][2] == "PREPARING"
    assert reporter.events[1][1]["total_rounds"] == 75
    assert reporter.events[2][1]["training_result"]["bundle_id"] == "bundle-1"


def test_s3_requires_explicit_ray_storage_and_maps_bundle_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "build_algorithm_dispatcher", lambda: dispatcher)
    storage = {
        "type": "s3",
        "bucket": "knova-models",
        "prefix": "tenant-a/job-1",
        "properties": {"ray_storage_path": "/mnt/ray-checkpoints"},
    }

    training.execute_training(_request(storage_context=storage), _Reporter())

    config = dispatcher.calls[0][0].algorithm_request.algorithm_config
    assert config["ray"] == {"storage_path": "/mnt/ray-checkpoints"}
    assert config["output"] == {"bundle_uri": "s3://knova-models/tenant-a/job-1"}

    storage["properties"] = {}
    with pytest.raises(ValueError, match="ray_storage_path") as captured:
        training.execute_training(_request(storage_context=storage), _Reporter())
    assert _SECRET not in str(captured.value)


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
    monkeypatch.setattr(training, "build_algorithm_dispatcher", lambda: dispatcher)
    target = {
        "origin": {"column_name": "label"},
        "task_type": task_type,
        "label_mapping": label_mapping,
    }

    training.execute_training(_request(target=target), _Reporter())

    model = dispatcher.calls[0][0].algorithm_request.algorithm_config["model"]
    assert {key: model[key] for key in expected} == expected


def test_unsupported_algorithm_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "build_algorithm_dispatcher", lambda: dispatcher)
    algorithm = {"algorithm_key": "lightgbm", "hyper_params": {}}

    with pytest.raises(ValueError, match="only algorithm_key=xgboost") as captured:
        training.execute_training(_request(algorithm=algorithm), _Reporter())

    assert dispatcher.calls == []
    assert _SECRET not in str(captured.value)


def test_clickhouse_requires_native_table_and_never_falls_back_to_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    monkeypatch.setattr(training, "build_algorithm_dispatcher", lambda: dispatcher)
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


def test_dispatcher_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingDispatcher:
        def execute(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(_SECRET)

    monkeypatch.setattr(
        training, "build_algorithm_dispatcher", lambda: _FailingDispatcher()
    )

    with pytest.raises(RuntimeError) as captured:
        training.execute_training(_request(), _Reporter())

    assert _SECRET not in str(captured.value)
