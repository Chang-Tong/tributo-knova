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
                "origin": {"table_alias": "t0", "column_name": "user_id"},
                "result_column": "entity_id",
            },
            "query": {
                "sql": (
                    "SELECT t0.user_id, t0.spend, t0.active_days "
                    "FROM analytics.features t0 WHERE t0.stat_month = {month:String}"
                ),
                "params": {"month": "202601"},
            },
            "features": [
                {
                    "origin": {"table_alias": "t0", "column_name": "spend"},
                    "result_column": "t0__spend",
                },
                {
                    "origin": {
                        "table_alias": "t0",
                        "column_name": "active_days",
                    },
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
    assert core.model.kind == "bundle"
    assert core.input.engine == "tributo.ray_data"
    assert core.input.binding_id == "tributo.knova.ray.clickhouse"
    assert core.input.source.options["table"] == "analytics.features"
    assert core.input.source.options["columns"] == [
        "user_id",
        "spend",
        "active_days",
        "stat_month",
    ]
    assert core.input.source.options["partitioning"] == {
        "mode": "parallel",
        "column": "user_id",
        "num_partitions": 3,
    }
    assert core.input.transforms.steps[0].column == "stat_month"
    assert core.input.transforms.steps[0].value == "202601"
    assert core.input_binding.tensors[0].columns == ("spend", "active_days")
    assert core.input_binding.passthrough_columns == ("user_id",)
    assert core.execution.batch_size == 500
    assert core.execution.concurrency == 3
    assert core.result_sink.sink_id == "data-write-v1"
    assert sink.sink_id == "data-write-v1"
    assert credentials == ("reader", _SECRET)
    assert _SECRET not in core.model_dump_json()
    assert _SECRET not in repr(core)


def test_standalone_onnx_maps_to_public_artifact_importer() -> None:
    request = _request()
    payload = request.model_dump(mode="python")
    payload["model"].pop("bundle_uri")
    payload["model"]["storage"] = {
        "type": "nfs",
        "bucket": "/mnt/models",
        "prefix": "model-1",
        "properties": {},
    }
    payload["model"]["model_artifacts"] = {
        "model_weights": {
            "alternatives": [
                {
                    "format": "onnx",
                    "files": [
                        {
                            "path": "model.onnx",
                            "hash": "sha256:" + "a" * 64,
                            "metadata": {
                                "input_names": ["X"],
                                "output_names": ["label", "probabilities"],
                            },
                        }
                    ],
                }
            ]
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

    assert core.model.kind == "bundle"
    assert core.model.role == "inference"
    assert sink._model_reference["kind"] == "bundle"
    assert sink._model_reference["role"] == "native"
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
            "alternatives": [
                {
                    "format": "onnx",
                    "files": [
                        {
                            "path": "artifacts/onnx-model/model.onnx",
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
                            "hash": "sha256:" + "b" * 64,
                            "metadata": {"supports_tree_shap": True},
                        },
                        {
                            "path": "artifacts/native-model/feature_names.json",
                            "hash": "sha256:" + "c" * 64,
                            "metadata": {},
                        },
                    ],
                },
            ]
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
    assert sink._model_reference["uri"].endswith(
        "/artifacts/native-model/model.ubj"
    )


def test_complex_query_is_rejected_without_executing_raw_sql() -> None:
    payload = _request().model_dump(mode="python")
    payload["input"]["query"]["sql"] = (
        "SELECT * FROM analytics.features t0 "
        "JOIN analytics.other t1 ON t0.user_id=t1.user_id"
    )

    with pytest.raises(ValueError, match="single-table SELECT") as captured:
        inference._build_request(InferenceExecutionRequest.model_validate(payload))

    assert _SECRET not in str(captured.value)


def test_execute_inference_scopes_credentials_and_reports_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _Reporter()
    captured: list[tuple[Any, Any, str | None]] = []
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
            (core, bound_sink, os.environ.get("TRIBUTO_CLICKHOUSE_PASSWORD"))
        )
        return Result()

    monkeypatch.setattr(inference, "run_inference", run)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 7)

    result = inference.execute_inference(_request(), reporter)

    assert result.status == "succeeded"
    assert captured[0][2] == _SECRET
    assert os.environ["TRIBUTO_CLICKHOUSE_PASSWORD"] == "previous"
    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "PROGRESS",
        "COMPLETED",
    ]
    assert reporter.events[1][1] == {
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
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 1)

    with pytest.raises(RuntimeError) as captured:
        inference.execute_inference(_request(), _Reporter())

    assert _SECRET not in str(captured.value)


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


def test_memory_pressure_retries_with_smaller_batch(
    monkeypatch: pytest.MonkeyPatch,
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
        return Result(succeeded=len(batches) > 1)

    monkeypatch.setattr(inference, "run_inference", run)
    monkeypatch.setattr(inference, "_measure_input_rows", lambda *_args: 12)

    result = inference.execute_inference(_request(), reporter)

    assert result.status == "succeeded"
    assert batches == [12, 6]
    assert [event[0] for event in reporter.events] == [
        "PHASE",
        "PROGRESS",
        "LOG",
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
    assert json.loads(result["pred_extra"][0]) == {
        "inference_feature_values": [
            {"feature_name": "t0__spend", "value": 10.5},
            {"feature_name": "t0__active_days", "value": 30},
        ]
    }


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


def test_bound_sink_delegates_to_ray_native_clickhouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _core, sink, _credentials = inference._build_request(_request())
    calls: list[tuple[str, dict[str, Any]]] = []

    class Dataset:
        def map_batches(self, _fn: Any, **kwargs: Any) -> Dataset:
            calls.append(("map_batches", kwargs))
            return self

        def materialize(self) -> Dataset:
            calls.append(("materialize", {}))
            return self

        def count(self) -> int:
            calls.append(("count", {}))
            return 7

        def write_clickhouse(self, **kwargs: Any) -> None:
            calls.append(("write_clickhouse", kwargs))

    import ray.data

    monkeypatch.setattr(ray.data, "SinkMode", SimpleNamespace(APPEND="append"))
    receipt = sink.write(Dataset(), run_id="inference-1", plan_digest="a" * 64)

    assert [name for name, _kwargs in calls] == [
        "map_batches",
        "materialize",
        "count",
        "write_clickhouse",
    ]
    write = calls[3][1]
    assert write["table"] == "analytics.ml_inference_result"
    assert write["mode"] == "append"
    assert write["max_insert_block_rows"] == 500
    assert receipt.rows_written == 7
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

        def count(self) -> int:
            calls.append("count")
            return 1

        def write_clickhouse(self, **_kwargs: Any) -> None:
            calls.append("write")

    import ray.data

    monkeypatch.setattr(ray.data, "SinkMode", SimpleNamespace(APPEND="append"))
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
        "materialize",
        "count",
        "write",
    ]
