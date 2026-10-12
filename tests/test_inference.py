from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tributo_knova import inference
from tributo_knova.protocol import InferenceExecutionRequest

_SECRET = "inference-password-must-not-leak"


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


def _request(**updates: Any) -> InferenceExecutionRequest:
    payload: dict[str, Any] = {
        "protocol_version": "2.0",
        "execution_id": "inference-1",
        "task_id": "task-1",
        "tenant_id": "tenant-1",
        "model": {
            "model_id": "model-1",
            "version_id": "version-1",
            "algorithm_key": "xgboost",
            "algorithm_category": "CLASSIFICATION",
            "storage": {"type": "nas", "bucket": "/mnt/models", "prefix": "bundle-1/"},
            "model_artifacts": {
                "model_weights": {
                    "required_for_inference": True,
                    "alternatives": [
                        {
                            "format": "onnx",
                            "files": [
                                {
                                    "path": "model.onnx",
                                    "size": 10,
                                    "hash": "sha256:" + "a" * 64,
                                }
                            ],
                        },
                        {
                            "format": "xgboost",
                            "files": [
                                {
                                    "path": "model.ubj",
                                    "size": 10,
                                    "hash": "sha256:" + "b" * 64,
                                }
                            ],
                        },
                    ],
                }
            },
            "task_type": "BINARY_CLASSIFICATION",
            "label_mapping": {"active": 0, "churn": 1},
            "positive_label_value": "churn",
            "bundle_uri": "/mnt/models/bundle-1",
        },
        "input": {
            "datasource": {
                "type": "CLICKHOUSE",
                "host": "clickhouse",
                "port": 8123,
                "database_name": "analytics",
                "username": "reader",
                "password": _SECRET,
                "properties": {},
            },
            "tables": [
                {
                    "table_alias": "t0",
                    "database_name": "analytics",
                    "table_name": "features",
                    "role": "PRIMARY",
                }
            ],
            "entity_key": {
                "origin": {
                    "table_alias": "t0",
                    "column_name": "user_id",
                    "column_type": "Int64",
                },
                "result_column": "entity_id",
            },
            "query": {
                "sql": (
                    "SELECT t0.user_id AS entity_id, t0.spend AS t0__spend, "
                    "t0.active_days AS t0__active_days "
                    "FROM analytics.features t0 WHERE t0.stat_month = {month:String}"
                ),
                "params": {"month": "202601"},
            },
            "features": [
                {
                    "feature_id": "f1",
                    "display_name": "spend",
                    "feature_role": "REGULAR",
                    "origin": {
                        "table_alias": "t0",
                        "column_name": "spend",
                        "column_type": "Float64",
                    },
                    "result_column": "t0__spend",
                },
                {
                    "origin": {
                        "table_alias": "t0",
                        "column_name": "active_days",
                        "column_type": "Int32",
                    },
                    "feature_id": "f2",
                    "display_name": "active_days",
                    "feature_role": "REGULAR",
                    "result_column": "t0__active_days",
                },
            ],
        },
        "output": {
            "datasource": {
                "type": "CLICKHOUSE",
                "host": "clickhouse",
                "port": 8123,
                "database_name": "analytics",
                "username": "writer",
                "password": _SECRET,
                "properties": {},
            },
            "table_name": "ml_inference_result",
            "result_filter": None,
        },
        "execution": {"batch_size": 500, "concurrency": 3},
    }
    payload.update(updates)
    return InferenceExecutionRequest.model_validate(payload)


def test_build_request_uses_public_inference_contract_without_credentials() -> None:
    core, sink, credentials = inference._build_request(_request())

    assert core.run_id == "inference-1"
    assert core.model.kind == "artifact"
    assert core.input.engine == "tributo.ray_data"
    assert core.input.binding_id == "ray_clickhouse.ray.clickhouse"
    assert core.input.source.options["table"] == "features"
    assert core.input.source.options["columns"] == [
        "user_id",
        "spend",
        "active_days",
        "stat_month",
    ]
    assert core.input.source.options["partitioning"] == {"mode": "single"}
    assert core.input.read_options.batch_size == 200_000
    assert core.input.read_options.target_split_size_bytes == 64 * 1024 * 1024
    assert core.input.read_options.concurrency == 3
    assert core.input.transforms.steps[0].column == "stat_month"
    assert core.input.transforms.steps[0].value == "202601"
    assert core.input_binding.tensors[0].columns == ("spend", "active_days")
    assert core.input_binding.passthrough_columns == ("user_id",)
    assert core.execution.batch_size == 500
    assert core.execution.concurrency == 3
    assert core.result_sink.sink_id == "data-write-v1"
    assert core.result_sink.target == (
        "clickhouse://clickhouse:8123/analytics/ml_inference_result"
    )
    assert core.result_sink.binding_id == "tributo.ray.clickhouse"
    assert core.result_sink.options == {"batch_rows": 500, "concurrency": 3}
    assert core.result_sink.runtime_options == {
        "user": "writer",
        "credential_ref": "env://KNOVA_OUTPUT_CLICKHOUSE_PASSWORD",
    }
    assert sink.sink_id == "data-write-v1"
    assert credentials == {
        "TRIBUTO_CLICKHOUSE_USER": "reader",
        "TRIBUTO_CLICKHOUSE_PASSWORD": _SECRET,
        "KNOVA_OUTPUT_CLICKHOUSE_PASSWORD": _SECRET,
    }
    assert _SECRET not in core.model_dump_json()
    assert _SECRET not in repr(core)
    assert _SECRET not in repr(sink)


def test_standalone_onnx_maps_to_public_artifact_importer() -> None:
    request = _request()
    payload = request.model_dump(mode="python")
    payload["model"].pop("bundle_uri")
    payload["model"]["storage"] = {
        "type": "nfs",
        "bucket": "/mnt/models",
        "prefix": "model-1/",
        "properties": {},
    }
    payload["model"]["model_artifacts"] = {
        "model_weights": {
            "required_for_inference": True,
            "alternatives": [
                {
                    "format": "onnx",
                    "files": [
                        {
                            "path": "model.onnx",
                            "size": 10,
                            "hash": "sha256:" + "a" * 64,
                            "metadata": {
                                "input_names": ["X"],
                                "output_names": ["label", "probabilities"],
                            },
                        }
                    ],
                }
            ],
        }
    }

    core, _sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )

    assert core.model.kind == "artifact"
    assert core.model.uri == "/mnt/models/model-1/model.onnx"
    assert core.model.import_bundle_uri.endswith("/.tributo-imports/inference-1")
    assert core.input_binding.tensors[0].tensor_name == "X"
    assert core.model.options["input_fields"][0]["shape"] == ["batch", 2]


def test_tree_shap_keeps_onnx_prediction_and_selects_native_attribution() -> None:
    payload = _request().model_dump(mode="python")
    payload["extensions"] = {
        "explanation": {
            "enabled": True,
            "method": "TREE_SHAP",
            "approximate": False,
        }
    }

    core, sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )

    assert core.model.kind == "artifact"
    assert core.model.format_id == "onnx"
    assert sink._model_reference["kind"] == "artifact"
    assert sink._model_reference["format_id"] == "ubj"
    assert sink._explanation == {"method": "TREE_SHAP", "approximate": False}


def test_protocol_xgboost_artifact_maps_to_public_ubj_importer() -> None:
    payload = _request().model_dump(mode="python")
    payload["model"].pop("bundle_uri")
    payload["model"]["storage"] = {
        "type": "nas",
        "bucket": "/mnt/models",
        "prefix": "bundle-1/",
        "properties": {},
    }
    payload["model"]["model_artifacts"] = {
        "model_weights": {
            "required_for_inference": True,
            "alternatives": [
                {
                    "format": "onnx",
                    "files": [
                        {
                            "path": "artifacts/onnx-model/model.onnx",
                            "size": 10,
                            "hash": "sha256:" + "a" * 64,
                            "metadata": {},
                        }
                    ],
                },
                {
                    "format": "xgboost",
                    "files": [
                        {
                            "path": "artifacts/native-model/model.ubj",
                            "size": 10,
                            "hash": "sha256:" + "b" * 64,
                            "metadata": {"supports_tree_shap": True},
                        },
                        {
                            "path": "artifacts/native-model/feature_names.json",
                            "size": 10,
                            "hash": "sha256:" + "c" * 64,
                            "metadata": {},
                        },
                    ],
                },
            ],
        }
    }
    payload["extensions"] = {
        "explanation": {
            "enabled": True,
            "method": "TREE_SHAP",
            "approximate": False,
        }
    }

    core, sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )

    assert core.model.format_id == "onnx"
    assert sink._model_reference["format_id"] == "ubj"
    assert sink._model_reference["uri"].endswith("/artifacts/native-model/model.ubj")


def test_complex_query_is_rejected_without_executing_raw_sql() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"]["sql"] = (
        "SELECT * FROM analytics.features t0 "
        "JOIN analytics.other t1 ON t0.user_id=t1.user_id"
    )

    with pytest.raises(ValueError, match="single-table SELECT") as captured:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))

    assert _SECRET not in str(captured.value)


def test_ordered_query_filter_preserves_declared_string_type() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"] = {
        "sql": (
            "SELECT t0.user_id AS entity_id, t0.spend AS t0__spend, "
            "t0.active_days AS t0__active_days "
            "FROM analytics.features AS t0 WHERE t0.user_id <= {p0:String}"
        ),
        "params": {"p0": "1000000"},
    }

    core, _sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )

    step = core.input.transforms.steps[0]
    assert step.type == "filter_comparison"
    assert step.column == "user_id"
    assert step.operator == "lte"
    assert step.value == "1000000"


def test_generated_row_limit_is_preserved_as_ordered_ray_transform() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"] = {
        "sql": (
            "SELECT * FROM (SELECT t0.user_id AS entity_id, t0.spend AS "
            "t0__spend, t0.active_days AS t0__active_days "
            "FROM analytics.features AS t0 LIMIT 2000) sub"
        ),
        "params": {},
    }

    core, _sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )

    assert [step.type for step in core.input.transforms.steps] == ["limit"]
    assert core.input.transforms.steps[0].count == 2000


@pytest.mark.parametrize(
    "ordering",
    [
        "ORDER BY t0.user_id",
        "ORDER BY t0.user_id ASC",
        "ORDER BY t0.user_id DESC",
        "order\nby t0.user_id desc",
        "ORDER BY t0.user_id ASC, t0.spend DESC",
        "ORDER BY " + _SECRET,
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_order_by_limit_is_rejected_instead_of_changing_selected_entities(
    ordering: str,
    nested: bool,
) -> None:
    payload = _request().model_dump(mode="python")
    sql = (
        "SELECT t0.user_id AS entity_id, t0.spend AS t0__spend, t0.active_days "
        "AS t0__active_days FROM analytics.features t0 "
        "WHERE t0.stat_month = {month:String} " + ordering + " LIMIT 2000"
    )
    if nested:
        sql = "SELECT * FROM (" + sql + ") sub"
    payload["input"]["query"]["sql"] = sql

    with pytest.raises(
        inference._InferenceConfigurationError, match="ORDER BY"
    ) as error:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))

    assert str(error.value) == (
        "input.query ORDER BY is not supported by this runtime; "
        "disable deterministic ordering"
    )
    assert _SECRET not in str(error.value)


def test_order_by_without_limit_is_rejected_before_any_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"]["sql"] += " ORDER BY t0.user_id DESC"

    def unexpected_call(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unsupported ORDER BY must fail before counting or inference")

    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_args: core)
    monkeypatch.setattr(inference, "_measure_input_rows", unexpected_call)
    monkeypatch.setattr(inference, "run_inference", unexpected_call)
    with pytest.raises(inference._InferenceConfigurationError, match="ORDER BY"):
        inference.execute_inference(
            InferenceExecutionRequest.model_validate(payload),
            _Reporter(),
        )


def test_arbitrary_nested_query_is_rejected() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"] = {
        "sql": (
            "SELECT * FROM (SELECT t0.user_id AS entity_id, t0.spend AS "
            "t0__spend, t0.active_days AS t0__active_days "
            "FROM analytics.features AS t0) sub"
        ),
        "params": {},
    }

    with pytest.raises(ValueError, match="single-table SELECT"):
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


def test_limit_with_offset_is_rejected_instead_of_changing_row_identity() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"] = {
        "sql": (
            "SELECT t0.user_id AS entity_id, t0.spend AS t0__spend, "
            "t0.active_days AS t0__active_days "
            "FROM analytics.features AS t0 LIMIT 2000 OFFSET 10"
        ),
        "params": {},
    }

    with pytest.raises(ValueError, match="LIMIT must be"):
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


def test_execute_inference_scopes_credentials_and_reports_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _Reporter()
    captured: list[tuple[Any, Any, str | None, str | None]] = []
    monkeypatch.setenv("TRIBUTO_CLICKHOUSE_PASSWORD", "previous")

    class Result:
        status = "succeeded"
        failure = None
        output_rows = 7
        sink_receipt = SimpleNamespace(uri="clickhouse://clickhouse:8123/analytics.out")

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {"status": self.status, "output_rows": self.output_rows}

    def run(core: Any, *, bound_sink: Any) -> Result:
        captured.append(
            (
                core,
                bound_sink,
                os.environ.get("TRIBUTO_CLICKHOUSE_PASSWORD"),
                os.environ.get("KNOVA_OUTPUT_CLICKHOUSE_PASSWORD"),
            )
        )
        return Result()

    monkeypatch.setattr(inference, "run_inference", run)
    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_args: core)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 7)

    result = inference.execute_inference(_request(), reporter)

    assert result.status == "succeeded"
    assert captured[0][2] == _SECRET
    assert captured[0][3] == _SECRET
    assert os.environ["TRIBUTO_CLICKHOUSE_PASSWORD"] == "previous"
    assert "KNOVA_OUTPUT_CLICKHOUSE_PASSWORD" not in os.environ
    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "STARTED",
        "PROGRESS",
        "PROGRESS",
        "COMPLETED",
    ]
    assert reporter.events[2][1] == {
        "processed_rows": 0,
        "result_rows": 0,
        "total_rows": 7,
        "percent": 0.0,
    }
    assert reporter.events[-1][1]["result_summary"]["output_rows"] == 7
    assert reporter.events[-1][1]["processed_rows"] == 7
    assert reporter.events[-1][1]["result_rows"] == 7
    assert reporter.events[-1][1]["total_rows"] == 7


def test_execution_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_SECRET)

    monkeypatch.setattr(inference, "run_inference", fail)
    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_args: core)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 1)

    with pytest.raises(RuntimeError) as captured:
        inference.execute_inference(_request(), _Reporter())

    assert _SECRET not in str(captured.value)


def test_empty_query_is_rejected_before_model_execution_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_args: core)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 0)

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Empty input must not launch prediction or write output")

    monkeypatch.setattr(inference, "run_inference", unexpected)
    reporter = _Reporter()
    with pytest.raises(inference._EmptyInferenceInputError, match="zero rows"):
        inference.execute_inference(_request(), reporter)
    assert [event[0] for event in reporter.events] == ["PHASE"]


def test_measured_rows_select_adaptive_batch_policy() -> None:
    payload = _request().model_dump(mode="python")
    payload["extensions"] = {
        "tributo": {
            "inference_runtime": {
                "adaptive_target_batches": 20,
                "adaptive_min_batch_size": 50_000,
                "adaptive_max_batch_size": 1_000_000,
                "max_predictor_actors": 6,
            }
        }
    }
    payload["execution"].pop("concurrency")

    core, sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload),
        measured_rows=3_000_000,
    )

    assert core.execution.batch_size == 150_000
    assert core.execution.concurrency == 6
    assert core.input.source.options["partitioning"] == {"mode": "single"}
    assert sink._batch_size == 50_000


def test_default_concurrency_leaves_capacity_for_ray_input_tasks() -> None:
    payload = _request().model_dump(mode="python")
    payload["execution"].pop("concurrency")

    core, sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload),
        measured_rows=2_000,
    )

    assert core.execution.concurrency == 2
    assert sink._concurrency == 2


def test_default_adaptive_batch_and_read_tasks_are_capped_at_200k_rows() -> None:
    core, _sink, _credentials = inference._build_request(
        _request(),
        measured_rows=10_000_000,
    )

    assert core.execution.batch_size == 200_000
    assert core.input.source.options["partitioning"] == {"mode": "single"}
    assert core.input.read_options.batch_size == 200_000
    assert core.input.read_options.target_split_size_bytes == 64 * 1024 * 1024


@pytest.mark.parametrize("predictor_batch_size", [None, 100_000])
def test_memory_pressure_retries_with_smaller_batch(
    monkeypatch: pytest.MonkeyPatch,
    predictor_batch_size: int | None,
) -> None:
    reporter = _Reporter()
    batches: list[int] = []

    class Result:
        def __init__(self, *, succeeded: bool) -> None:
            self.status = "succeeded" if succeeded else "failed"
            self.failure = (
                None
                if succeeded
                else SimpleNamespace(
                    phase="materialization",
                    error_type="OutOfMemoryError",
                )
            )
            self.output_rows = 12 if succeeded else None
            self.sink_receipt = (
                SimpleNamespace(uri="clickhouse://clickhouse:8123/analytics.out")
                if succeeded
                else None
            )

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {"status": self.status, "output_rows": self.output_rows}

    def run(core: Any, *, bound_sink: Any) -> Result:
        del bound_sink
        batches.append(core.execution.batch_size)
        return Result(succeeded=len(batches) > 2)

    monkeypatch.setattr(inference, "run_inference", run)
    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_args: core)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 12)

    request = _request(
        extensions={
            "tributo": {
                "inference_runtime": {"predictor_batch_size": predictor_batch_size}
            }
        }
    )
    initial_payload = request.model_dump(mode="python")
    result = inference.execute_inference(request, reporter)

    assert result.status == "succeeded"
    expected_first_batch = predictor_batch_size or 12
    assert batches == [
        expected_first_batch,
        expected_first_batch // 2,
        expected_first_batch // 4,
    ]
    assert request.model_dump(mode="python") == initial_payload
    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "STARTED",
        "PROGRESS",
        "LOG",
        "LOG",
        "PROGRESS",
        "COMPLETED",
    ]


def test_protocol_batch_maps_binary_result_schema() -> None:
    frame = pd.DataFrame(
        {
            "user_id": pd.Series([11, None], dtype=object),
            "spend": [10.5, 2.0],
            "active_days": [30, 1],
            "__knova_label": [0, 1],
            "__knova_probabilities": [[0.8, 0.2], [0.1, 0.9]],
        }
    )
    task = _request().model_dump(mode="python")

    result = inference._protocol_batch(
        frame,
        task=task,
        entity_column="user_id",
        feature_columns=("spend", "active_days"),
        feature_result_names=("t0__spend", "t0__active_days"),
    )

    assert result["entity_id"].tolist() == ["11", ""]
    assert result["pred_label"].tolist() == ["active", "churn"]
    assert result["pred_probability"].tolist() == [0.2, 0.9]
    assert result["inferred_at"].dt.tz is None
    assert json.loads(result["pred_extra"][0]) == {
        "inference_feature_values": [
            {"feature_name": "t0__spend", "value": 10.5},
            {"feature_name": "t0__active_days", "value": 30},
        ]
    }


def test_protocol_batch_applies_per_execution_decision_threshold() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [11, 12],
            "spend": [10.5, 2.0],
            "active_days": [30, 1],
            # The model's native labels used its default threshold.  The
            # execution threshold must be the authoritative final decision.
            "__knova_label": [1, 1],
            "__knova_probabilities": [[0.25, 0.75], [0.15, 0.85]],
        }
    )
    task = _request().model_dump(mode="python")
    task["extensions"] = {"decision_threshold": 0.8}
    task["output"]["result_filter"] = {"mode": "INCLUDE", "labels": ["churn"]}

    result = inference._protocol_batch(
        frame,
        task=task,
        entity_column="user_id",
        feature_columns=("spend", "active_days"),
        feature_result_names=("t0__spend", "t0__active_days"),
    )

    assert result["entity_id"].tolist() == ["12"]
    assert result["pred_label"].tolist() == ["churn"]
    assert result["pred_probability"].tolist() == [0.85]
    assert json.loads(result["pred_extra"].iloc[0])["decision_threshold"] == 0.8


@pytest.mark.parametrize("threshold", [True, -0.01, 1.01, float("nan")])
def test_malformed_optional_decision_threshold_is_ignored(threshold: object) -> None:
    payload = _request().model_dump(mode="python")
    payload["extensions"] = {"decision_threshold": threshold}

    inference._build_request(InferenceExecutionRequest.model_validate(payload))
    assert inference._decision_threshold(payload) is None


@pytest.mark.parametrize("threshold", [None, 0.5])
def test_unknown_positive_label_is_rejected_without_reflecting_value(
    threshold: float | None,
) -> None:
    payload = _request().model_dump(mode="python")
    payload["model"]["positive_label_value"] = _SECRET
    if threshold is not None:
        payload["extensions"] = {"decision_threshold": threshold}
    with pytest.raises(inference._InferenceConfigurationError) as captured:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))
    assert "positive_label_value" in str(captured.value)
    assert _SECRET not in str(captured.value)


@pytest.mark.parametrize("positive", [0, 1, "0", "1"])
def test_numeric_positive_label_without_mapping(positive: int | str) -> None:
    model = {"positive_label_value": positive, "label_mapping": {}}
    assert inference._positive_label_index(model) == int(positive)
    assert inference._binary_labels(model) == (
        int(positive),
        str(positive),
        str(1 - int(positive)),
    )


def test_stable_result_batch_keeps_nullable_columns_consistent() -> None:
    common = {
        "entity_id": ["11"],
        "pred_extra": ["{}"],
        "execution_id": ["iexec-1"],
        "model_id": ["model-1"],
        "version_id": ["ver-1"],
        "tenant_id": ["tenant-1"],
        "inferred_at": [pd.Timestamp("2026-09-21T00:00:00")],
    }
    all_null = pd.DataFrame(
        {
            **common,
            "pred_label": [None],
            "pred_probability": [None],
            "pred_probabilities": [None],
            "pred_value": [None],
        }
    )
    populated = pd.DataFrame(
        {
            **common,
            "pred_label": ["churn"],
            "pred_probability": [0.9],
            "pred_probabilities": ['{"active":0.1,"churn":0.9}'],
            "pred_value": [None],
        }
    )

    assert inference._stable_result_batch(all_null).schema == (
        inference._stable_result_batch(populated).schema
    )


def test_protocol_batch_embeds_explicit_tree_shap_exactness() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [11],
            "spend": [10.5],
            "active_days": [30],
            "__knova_label": [1],
            "__knova_probabilities": [[0.1, 0.9]],
            "__knova_shap_values": [[0.25, -0.5]],
            "__knova_shap_base": [0.75],
            "__knova_shap_group": [0],
        }
    )
    task = _request().model_dump(mode="python")
    task["extensions"] = {
        "explanation": {
            "enabled": True,
            "method": "TREE_SHAP",
            "approximate": True,
        }
    }

    result = inference._protocol_batch(
        frame,
        task=task,
        entity_column="user_id",
        feature_columns=("spend", "active_days"),
        feature_result_names=("t0__spend", "t0__active_days"),
    )

    explanation = json.loads(result["pred_extra"][0])["explanation"]
    assert explanation["exactness"] == "approximate"
    assert explanation["approximate"] is True
    assert explanation["output_space"] == "RAW_MARGIN"
    assert explanation["explained_class_label"] == "churn"
    assert explanation["feature_contributions"] == [
        {"feature_name": "t0__spend", "shap_value": 0.25},
        {"feature_name": "t0__active_days", "shap_value": -0.5},
    ]


@pytest.mark.parametrize("approximate", [False, True])
def test_tree_shap_batch_loads_ubj_and_preserves_rows(
    tmp_path: Path,
    approximate: bool,
) -> None:
    import numpy as np
    import xgboost

    values = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.float32)
    matrix = xgboost.DMatrix(values, label=labels, feature_names=["f1", "f2"])
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2},
        matrix,
        num_boost_round=2,
    )
    model_path = tmp_path / "model.ubj"
    booster.save_model(model_path)
    frame = pd.DataFrame(
        {
            "f1": values[:, 0],
            "f2": values[:, 1],
            "__knova_label": [0, 0, 1],
        }
    )
    worker = inference._TreeShapBatch(
        model_reference={"kind": "artifact", "uri": str(model_path)},
        feature_columns=("f1", "f2"),
        task_type="BINARY_CLASSIFICATION",
        approximate=approximate,
    )

    result = worker(frame)

    assert len(result) == len(frame)
    assert all(len(row) == 2 for row in result["__knova_shap_values"])
    assert result["__knova_shap_group"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("positive_index", [0, 1])
def test_binary_shap_reconstructs_selected_positive_class(
    tmp_path: Path, approximate: bool, positive_index: int
) -> None:
    import numpy as np
    import xgboost

    values = np.asarray([[i, i % 3] for i in range(30)], dtype=np.float32)
    matrix = xgboost.DMatrix(
        values,
        label=np.asarray([int(i >= 20) for i in range(30)]),
        feature_names=["spend", "active_days"],
    )
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2, "nthread": 1},
        matrix,
        num_boost_round=3,
    )
    probability = booster.predict(matrix)
    model_path = tmp_path / "model.ubj"
    booster.save_model(model_path)
    frame = pd.DataFrame(
        {
            "user_id": range(30),
            "spend": values[:, 0],
            "active_days": values[:, 1],
            "__knova_label": (probability >= 0.5).astype(int),
            "__knova_probabilities": list(
                np.column_stack([1 - probability, probability])
            ),
        }
    )
    task = _request().model_dump(mode="python")
    task["model"]["positive_label_value"] = ["active", "churn"][positive_index]
    task["extensions"] = {
        "explanation": {
            "enabled": True,
            "method": "TREE_SHAP",
            "approximate": approximate,
        }
    }
    attributed = inference._TreeShapBatch(
        model_reference={"kind": "artifact", "uri": str(model_path)},
        feature_columns=("spend", "active_days"),
        task_type="BINARY_CLASSIFICATION",
        approximate=approximate,
    )(frame)
    result = inference._protocol_batch(
        attributed,
        task=task,
        entity_column="user_id",
        feature_columns=("spend", "active_days"),
        feature_result_names=("spend", "active_days"),
    )
    for row in result.itertuples():
        explanation = json.loads(row.pred_extra)["explanation"]
        margin = explanation["base_value"] + sum(
            item["shap_value"] for item in explanation["feature_contributions"]
        )
        assert explanation["explained_class_index"] == positive_index
        assert 1 / (1 + np.exp(-margin)) == pytest.approx(
            row.pred_probability, abs=1e-6
        )


@pytest.mark.parametrize(
    "task_type,result_filter",
    [
        ("REGRESSION", {"mode": "INCLUDE", "labels": ["active"]}),
        ("BINARY_CLASSIFICATION", {"mode": "INCLUDE", "labels": ["not-a-label"]}),
        ("BINARY_CLASSIFICATION", {"mode": "invalid", "labels": ["active"]}),
    ],
)
def test_result_filter_rejects_invalid_task_mode_and_label(
    task_type: str, result_filter: dict[str, Any]
) -> None:
    payload = _request().model_dump(mode="python")
    payload["model"]["task_type"] = task_type
    if task_type == "REGRESSION":
        payload["model"]["algorithm_category"] = "REGRESSION"
    payload["output"]["result_filter"] = result_filter
    with pytest.raises(inference._InferenceConfigurationError, match="result_filter"):
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


@pytest.mark.parametrize("task_type", ["MULTICLASS_CLASSIFICATION", "REGRESSION"])
@pytest.mark.parametrize("approximate", [False, True])
def test_multiclass_and_regression_shap_match_native_margin_and_feature_order(
    tmp_path: Path, task_type: str, approximate: bool
) -> None:
    import numpy as np
    import xgboost

    multi = task_type == "MULTICLASS_CLASSIFICATION"
    values = np.asarray([[i, i % 3] for i in range(30)], dtype=np.float32)
    labels = np.arange(30) % 3 if multi else np.arange(30) / 7 - 2
    matrix = xgboost.DMatrix(
        values, label=labels, feature_names=["spend", "active_days"]
    )
    parameters = {
        "objective": "multi:softprob" if multi else "reg:squarederror",
        "max_depth": 2,
        "nthread": 1,
    }
    if multi:
        parameters["num_class"] = 3
    booster = xgboost.train(parameters, matrix, num_boost_round=3)
    prediction = booster.predict(matrix, strict_shape=True)
    margin = booster.predict(matrix, output_margin=True, strict_shape=True)
    model_path = tmp_path / "model.ubj"
    booster.save_model(model_path)
    task = _request().model_dump(mode="python")
    task["model"].update(
        task_type=task_type,
        label_mapping={"原始甲": 0, "原始乙": 1, "原始丙": 2} if multi else {},
    )
    task["extensions"] = {
        "explanation": {
            "enabled": True,
            "method": "TREE_SHAP",
            "approximate": approximate,
        }
    }
    frame = pd.DataFrame(
        {
            "user_id": range(30),
            "spend": values[:, 0],
            "active_days": values[:, 1],
        }
    )
    if multi:
        frame["__knova_label"] = prediction.argmax(axis=1)
        frame["__knova_probabilities"] = list(prediction)
    else:
        frame["__knova_prediction"] = prediction[:, 0]
    worker = inference._TreeShapBatch(
        model_reference={"kind": "artifact", "uri": str(model_path)},
        feature_columns=("spend", "active_days"),
        task_type=task_type,
        approximate=approximate,
    )
    result = inference._protocol_batch(
        worker(frame),
        task=task,
        entity_column="user_id",
        feature_columns=("spend", "active_days"),
        feature_result_names=("原始消费", "活跃天数"),
    )
    for index, row in enumerate(result.itertuples()):
        explanation = json.loads(row.pred_extra)["explanation"]
        assert [
            item["feature_name"] for item in explanation["feature_contributions"]
        ] == ["原始消费", "活跃天数"]
        actual_margin = explanation["base_value"] + sum(
            item["shap_value"] for item in explanation["feature_contributions"]
        )
        group = int(prediction[index].argmax()) if multi else 0
        assert actual_margin == pytest.approx(float(margin[index, group]), abs=1e-5)
        if multi:
            assert explanation["explained_class_label"] == row.pred_label
            assert json.loads(row.pred_probabilities) == pytest.approx(
                prediction[index]
            )
        else:
            assert actual_margin == pytest.approx(row.pred_value, abs=1e-5)
            assert "explained_class_index" not in explanation
    with pytest.raises(ValueError, match="feature names"):
        inference._TreeShapBatch(
            model_reference={"kind": "artifact", "uri": str(model_path)},
            feature_columns=("active_days", "spend"),
            task_type=task_type,
            approximate=approximate,
        )


def test_bound_sink_delegates_to_tributo_write_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _core, sink, _credentials = inference._build_request(_request())
    calls: list[tuple[str, dict[str, Any]]] = []
    writes: list[tuple[Any, Any]] = []

    class Dataset:
        def map_batches(self, _fn: Any, **kwargs: Any) -> Dataset:
            calls.append(("map_batches", kwargs))
            return self

        def materialize(self) -> Dataset:
            calls.append(("materialize", {}))
            return self

    class Gateway:
        def execute(self, request: Any, handle: Any) -> Any:
            writes.append((request, handle))
            return SimpleNamespace(
                rows_written=7,
                binding_id="tributo.ray.clickhouse",
                committed=True,
            )

    monkeypatch.setattr(inference, "default_write_gateway", Gateway)
    receipt = sink.write(Dataset(), run_id="inference-1", plan_digest="a" * 64)

    assert [name for name, _kwargs in calls] == [
        "map_batches",
        "map_batches",
        "materialize",
    ]
    write_request, handle = writes[0]
    assert write_request.binding_id == "tributo.ray.clickhouse"
    assert write_request.target == (
        "clickhouse://clickhouse:8123/analytics/ml_inference_result"
    )
    assert handle.dataset.__class__ is Dataset
    assert receipt.rows_written == 7
    assert receipt.metadata["binding_id"] == "tributo.ray.clickhouse"
    assert _SECRET not in receipt.model_dump_json()


def test_tree_shap_sink_releases_predictor_actors_before_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _request().model_dump(mode="python")
    payload["extensions"] = {"explanation": {"enabled": True, "method": "TREE_SHAP"}}
    _core, sink, _credentials = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )
    calls: list[str] = []

    class Dataset:
        def map_batches(self, fn: Any, **_kwargs: Any) -> Dataset:
            calls.append(f"map:{fn.__name__}")
            return self

        def materialize(self) -> Dataset:
            calls.append("materialize")
            return self

    class Gateway:
        def execute(self, _request: Any, _handle: Any) -> Any:
            calls.append("gateway")
            return SimpleNamespace(
                rows_written=1,
                binding_id="tributo.ray.clickhouse",
                committed=True,
            )

    import ray.data

    monkeypatch.setattr(inference, "default_write_gateway", Gateway)
    monkeypatch.setattr(
        ray.data,
        "ActorPoolStrategy",
        lambda *, size: ("actors", size),
    )

    sink.write(Dataset(), run_id="inference-1", plan_digest="a" * 64)

    assert calls == [
        "materialize",
        "map:_TreeShapBatch",
        "map:_protocol_batch",
        "map:_stable_result_batch",
        "materialize",
        "gateway",
    ]


def _real_model_request(tmp_path: Path) -> InferenceExecutionRequest:
    import hashlib

    from onnx import TensorProto, helper

    names = ["t0__active_days", "t0__spend"]
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["float_input", "weights"], ["prediction"])],
        "order-sensitive",
        [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [None, 1])],
        [helper.make_tensor("weights", TensorProto.FLOAT, [2, 1], [2.0, -3.0])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=9
    )
    helper.set_model_props(model, {"feature_names": json.dumps(names)})
    data = model.SerializeToString()
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(data)
    payload = _request().model_dump(mode="python")
    payload["model"].update(
        task_type="REGRESSION",
        algorithm_category="REGRESSION",
        label_mapping={},
        storage={"type": "nas", "bucket": str(tmp_path), "prefix": "model/"},
        model_artifacts={
            "model_weights": {
                "required_for_inference": True,
                "alternatives": [
                    {
                        "format": "onnx",
                        "files": [
                            {
                                "path": "model.onnx",
                                "size": len(data),
                                "hash": "sha256:" + hashlib.sha256(data).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        },
    )
    return InferenceExecutionRequest.model_validate(payload)


def test_verified_model_order_controls_predictions_when_declarations_reordered(
    tmp_path: Path,
) -> None:
    import numpy as np
    import onnxruntime

    request = _real_model_request(tmp_path)
    frame = pd.DataFrame({"spend": [10.0, 7.0], "active_days": [3.0, 9.0]})
    predictions = []
    for reverse in (False, True):
        payload = request.model_dump(mode="python")
        if reverse:
            payload["input"]["features"].reverse()
        req = InferenceExecutionRequest.model_validate(payload)
        core, sink, _ = inference._build_request(req)
        core = inference._prepare_model(core, sink, req)
        assert core.input_binding.tensors[0].columns == ("active_days", "spend")
        session = onnxruntime.InferenceSession(core.model.uri)
        predictions.append(
            session.run(
                None,
                {
                    "float_input": frame.loc[
                        :, list(core.input_binding.tensors[0].columns)
                    ].to_numpy(dtype=np.float32)
                },
            )[0]
        )
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_array_equal(predictions[0].ravel(), [-24, -3])


def test_corrupt_artifact_rejected_before_query_or_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = _real_model_request(tmp_path)
    (tmp_path / "model/model.onnx").write_bytes(b"corrupted")
    monkeypatch.setattr(
        inference,
        "_measure_input_rows",
        lambda *_: pytest.fail("must not query corrupt model"),
    )
    with pytest.raises(inference._InferenceConfigurationError) as error:
        inference.execute_inference(req, _Reporter())
    assert error.value.code == "MODEL_ARTIFACT_CORRUPTED"


def test_missing_model_and_feature_schema_mismatch_are_distinct(tmp_path: Path) -> None:
    request = _real_model_request(tmp_path)
    payload = request.model_dump(mode="python")
    payload["input"]["features"][0]["result_column"] = "wrong_column"
    payload["input"]["query"]["sql"] = payload["input"]["query"]["sql"].replace(
        "AS t0__spend", "AS wrong_column"
    )
    req = InferenceExecutionRequest.model_validate(payload)
    core, sink, _ = inference._build_request(req)
    with pytest.raises(inference._InferenceConfigurationError) as mismatch:
        inference._prepare_model(core, sink, req)
    assert mismatch.value.code == "MODEL_SCHEMA_MISMATCH"
    (tmp_path / "model/model.onnx").unlink()
    with pytest.raises(inference._InferenceConfigurationError) as missing:
        inference._prepare_model(core, sink, req)
    assert missing.value.code == "MODEL_ARTIFACT_NOT_FOUND"


@pytest.mark.parametrize(
    "mutation",
    [
        "part",
        "format",
        "duplicate_format",
        "duplicate_path",
        "absolute_path",
        "traversal",
        "missing_hash",
        "missing_size",
        "prefix",
    ],
)
def test_artifact_manifest_invalid_cases_cannot_use_bundle_escape(
    mutation: str,
) -> None:
    payload = _request().model_dump(mode="python")
    artifacts = payload["model"]["model_artifacts"]
    entry = artifacts["model_weights"]
    file = entry["alternatives"][0]["files"][0]
    if mutation == "part":
        artifacts["unknown"] = entry
    elif mutation == "format":
        entry["alternatives"][0]["format"] = "ubj"
    elif mutation == "duplicate_format":
        entry["alternatives"].append(entry["alternatives"][0])
    elif mutation == "duplicate_path":
        entry["alternatives"][1]["files"][0]["path"] = file["path"]
    elif mutation == "absolute_path":
        file["path"] = "/etc/passwd"
    elif mutation == "traversal":
        file["path"] = "../model.onnx"
    elif mutation == "missing_hash":
        file.pop("hash")
    elif mutation == "missing_size":
        file.pop("size")
    else:
        payload["model"]["storage"]["prefix"] = "no-slash"
    with pytest.raises(inference._InferenceConfigurationError) as error:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))
    assert error.value.code == "INVALID_PAYLOAD"


@pytest.mark.parametrize("required", [False, True])
def test_required_preprocessor_never_silently_skipped(required: bool) -> None:
    payload = _request().model_dump(mode="python")
    payload["model"]["model_artifacts"]["preprocessor"] = {
        "required_for_inference": required,
        "alternatives": [
            {
                "format": "sklearn",
                "files": [
                    {"path": "prep.joblib", "size": 1, "hash": "sha256:" + "c" * 64}
                ],
            }
        ],
    }
    if required:
        with pytest.raises(inference._InferenceConfigurationError) as error:
            inference._build_request(InferenceExecutionRequest.model_validate(payload))
        assert error.value.code == "MODEL_FORMAT_UNSUPPORTED"
    else:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


@pytest.mark.parametrize(
    "sql",
    [
        None,
        "SELECT t0.user_id AS entity_id, t0.spend * 100 AS t0__spend, "
        "t0.active_days AS t0__active_days FROM analytics.features t0",
        "SELECT DISTINCT t0.user_id AS entity_id, t0.spend AS t0__spend, "
        "t0.active_days AS t0__active_days FROM analytics.features t0",
    ],
)
def test_query_semantics_are_not_silently_discarded(sql: str | None) -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"] = None if sql is None else {"sql": sql}
    with pytest.raises(inference._InferenceConfigurationError):
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


@pytest.mark.parametrize(
    "sql_type,value,expected",
    [("Int64", "42", 42), ("String", "0042", "0042"), ("Float64", "4.2", 4.2)],
)
def test_query_parameter_types_survive_mapping(
    sql_type: str, value: str, expected: object
) -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"]["sql"] = payload["input"]["query"]["sql"].replace(
        "{month:String}", "{month:" + sql_type + "}"
    )
    payload["input"]["query"]["params"]["month"] = value
    core, _, _ = inference._build_request(
        InferenceExecutionRequest.model_validate(payload)
    )
    assert core.input.transforms.steps[0].value == expected
    assert type(core.input.transforms.steps[0].value) is type(expected)


def test_credential_reference_takes_precedence_and_missing_ref_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTRACT_SECRET", "preferred")
    ds = {"password": "inline", "credential_ref": "env://CONTRACT_SECRET"}
    assert inference._credentials(ds, inline_password_env="INLINE")[1] == "preferred"
    monkeypatch.delenv("CONTRACT_SECRET")
    with pytest.raises(inference._InferenceConfigurationError):
        inference._credentials(ds, inline_password_env="INLINE")


def test_decimal_echo_is_json_serializable_without_losing_precision() -> None:
    from decimal import Decimal

    values = {
        "precise": Decimal("123456789.123456789"),
        "nonfinite": [Decimal("NaN"), float("inf")],
    }
    assert json.loads(json.dumps(inference._json_value(values), allow_nan=False)) == {
        "precise": "123456789.123456789",
        "nonfinite": [None, None],
    }


@pytest.mark.parametrize("mapping", [None, {}])
def test_native_multiclass_without_label_mapping_uses_verified_class_count(
    tmp_path: Path, mapping: object
) -> None:
    import hashlib

    import numpy as np
    import xgboost

    values = np.asarray([[i, i % 3] for i in range(30)], dtype=np.float32)
    data = xgboost.DMatrix(
        values, label=np.arange(30) % 3, feature_names=["t0__spend", "t0__active_days"]
    )
    booster = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "nthread": 1},
        data,
        num_boost_round=2,
    )
    folder = tmp_path / "model"
    folder.mkdir()
    path = folder / "model.ubj"
    booster.save_model(path)
    payload = _request().model_dump(mode="python")
    payload["model"].update(
        task_type="MULTICLASS_CLASSIFICATION",
        label_mapping=mapping,
        storage={"type": "nas", "bucket": str(tmp_path), "prefix": "model/"},
        model_artifacts={
            "model_weights": {
                "required_for_inference": True,
                "alternatives": [
                    {
                        "format": "xgboost",
                        "files": [
                            {
                                "path": "model.ubj",
                                "size": path.stat().st_size,
                                "hash": "sha256:"
                                + hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        },
    )
    req = InferenceExecutionRequest.model_validate(payload)
    core, sink, _ = inference._build_request(req)
    core = inference._prepare_model(core, sink, req)
    assert core.model.kind == "bundle"
    manifest = json.loads((Path(core.model.uri) / "manifest.json").read_text())
    assert manifest["output_signature"]["output_fields"][-1]["shape"] == ["batch", 3]


@pytest.mark.parametrize(
    "before,after",
    [("analytics.features", "analytics.FEATURES"), ("t0.spend", "T0.spend")],
)
def test_sql_identifier_case_is_not_discarded(before: str, after: str) -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"]["sql"] = payload["input"]["query"]["sql"].replace(
        before, after
    )
    with pytest.raises(inference._InferenceConfigurationError):
        inference._build_request(InferenceExecutionRequest.model_validate(payload))


def test_uint64_upper_half_retains_unsigned_count_parameter() -> None:
    number = inference._typed_query_parameter(str(2**64 - 1), "UInt64")
    assert inference._parameter_type(number) == "UInt64"
    with pytest.raises(inference._InferenceConfigurationError):
        inference._typed_query_parameter(str(2**64), "UInt64")


def test_input_schema_check_returns_column_missing_before_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import clickhouse_connect

    queries = []

    class Client:
        def query(self, sql: str, **kwargs: Any) -> Any:
            queries.append(sql)
            return SimpleNamespace(
                result_rows=[
                    ("user_id", "Int64"),
                    ("spend", "Float64"),
                    ("stat_month", "String"),
                ]
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **_: Client())
    core, _, credentials = inference._build_request(_request())
    with pytest.raises(inference._InferenceConfigurationError) as error:
        inference._measure_input_rows(core, credentials)
    assert error.value.code == "INPUT_COLUMN_MISSING"
    assert len(queries) == 1


def test_unreachable_input_has_specific_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import clickhouse_connect

    def unreachable(**kwargs: Any) -> None:
        raise ConnectionError(_SECRET)

    monkeypatch.setattr(clickhouse_connect, "get_client", unreachable)
    core, _, credentials = inference._build_request(_request())
    with pytest.raises(inference._InferenceExecutionError) as error:
        inference._measure_input_rows(core, credentials)
    assert error.value.code == "DATA_SOURCE_UNREACHABLE"
    assert _SECRET not in str(error.value)


@pytest.mark.parametrize(
    "phase,code",
    [
        ("acquisition", "INPUT_QUERY_FAILED"),
        ("sink", "RESULT_WRITE_FAILED"),
        ("materialization", "PREDICTION_FAILED"),
    ],
)
def test_core_failure_keeps_protocol_phase_and_error_code(
    monkeypatch: pytest.MonkeyPatch, phase: str, code: str
) -> None:
    monkeypatch.setattr(inference, "_prepare_model", lambda core, *_: core)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_: 1)
    monkeypatch.setattr(
        inference,
        "run_inference",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="failed",
            failure=SimpleNamespace(phase=phase, error_type="ValueError"),
        ),
    )
    with pytest.raises(inference._InferenceExecutionError) as error:
        inference.execute_inference(_request(), _Reporter())
    assert error.value.code == code
    assert (
        error.value.phase
        == {
            "acquisition": "READ_INPUT",
            "sink": "WRITE_RESULT",
            "materialization": "PREDICT",
        }[phase]
    )


def test_native_artifact_uses_verified_schema_sidecar_when_booster_omits_names(
    tmp_path: Path,
) -> None:
    import hashlib

    import numpy as np
    import xgboost

    folder = tmp_path / "native"
    folder.mkdir()
    matrix = xgboost.DMatrix(
        np.asarray([[1, 2], [3, 4]], dtype=np.float32), label=[0, 1]
    )
    booster = xgboost.train({"objective": "binary:logistic", "nthread": 1}, matrix, 1)
    booster.save_model(folder / "model.ubj")
    (folder / "feature_names.json").write_text(
        json.dumps(["t0__active_days", "t0__spend"])
    )
    files = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.iterdir())
    ]
    payload = _request().model_dump(mode="python")
    payload["model"].update(
        storage={"type": "nas", "bucket": str(tmp_path), "prefix": "native/"},
        model_artifacts={
            "model_weights": {
                "required_for_inference": True,
                "alternatives": [{"format": "xgboost", "files": files}],
            }
        },
    )
    req = InferenceExecutionRequest.model_validate(payload)
    core, sink, _ = inference._build_request(req)
    prepared = inference._prepare_model(core, sink, req)
    assert prepared.input_binding.tensors[0].columns == ("active_days", "spend")
    (folder / "feature_names.json").write_text('["changed","names"]')
    with pytest.raises(inference._InferenceConfigurationError) as error:
        inference._prepare_model(core, sink, req)
    assert error.value.code == "MODEL_ARTIFACT_CORRUPTED"
