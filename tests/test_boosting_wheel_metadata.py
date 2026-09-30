"""The portable model must expose the same outputs as its checkpoint contract."""

import json

# Execute the authoritative wheel source, never a possibly stale editable install.
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import pytest
import xgboost
from tributo.exporting.models import (
    CheckpointField,
    ExportCheckpointV1,
    ExportContext,
    ExportSource,
)

_WHEEL = (
    Path(__file__).resolve().parents[1]
    / "vendor/wheels/tributo_algorithms_boosting-0.1.0-py3-none-any.whl"
)
with zipfile.ZipFile(_WHEEL) as _archive:
    _source = _archive.read("tributo_algorithms_boosting/native.py")
_vendor_module = types.ModuleType("_vendored_boosting_native_schema_test")
sys.modules[_vendor_module.__name__] = _vendor_module
exec(compile(_source, str(_WHEEL) + ":native.py", "exec"), _vendor_module.__dict__)
OfficialXGBoostONNXExporter = _vendor_module.OfficialXGBoostONNXExporter


@pytest.mark.parametrize(
    "objective,data_case",
    [
        ("binary:logistic", "balanced"),
        ("multi:softprob", "balanced"),
        ("reg:squarederror", "nonzero_mean"),
        ("binary:logistic", "constant_0.2"),
        ("binary:logistic", "constant_0.5"),
        ("binary:logistic", "constant_0.8"),
        ("binary:logistic", "variable_0.2"),
        ("binary:logistic", "variable_0.8"),
        ("multi:softprob", "constant_imbalanced"),
        ("multi:softprob", "variable_imbalanced"),
        ("reg:squarederror", "constant_nonzero_mean"),
    ],
)
def test_exported_graph_matches_checkpoint_names_and_native_values(
    tmp_path: Path, objective: str, data_case: str
) -> None:
    classification = objective != "reg:squarederror"
    multi = objective == "multi:softprob"
    features = np.asarray([[i / 7, i % 3] for i in range(30)], dtype=np.float32)
    labels = (
        np.arange(30) % (3 if multi else 2) if classification else np.arange(30) / 9 - 2
    )
    if data_case.startswith("constant_"):
        features = np.zeros((30, 2), dtype=np.float32)
    if objective == "binary:logistic" and data_case != "balanced":
        positive_fraction = float(data_case.split("_")[-1])
        labels = np.asarray(
            [1] * int(30 * positive_fraction) + [0] * (30 - int(30 * positive_fraction))
        )
    elif multi and data_case.endswith("imbalanced"):
        labels = np.asarray([0] * 3 + [1] * 9 + [2] * 18)
    matrix = xgboost.DMatrix(features, label=labels, feature_names=["f0", "f1"])
    parameters = {"objective": objective, "max_depth": 2, "nthread": 1}
    if multi:
        parameters["num_class"] = 3
    booster = xgboost.train(parameters, matrix, num_boost_round=3)
    output_schema = (
        (
            CheckpointField(name="label", dtype="int64", shape=("batch",)),
            CheckpointField(
                name="probabilities",
                dtype="float32",
                shape=("batch", 3 if multi else 2),
            ),
        )
        if classification
        else (CheckpointField(name="prediction", dtype="float32", shape=("batch", 1)),)
    )
    contract = ExportCheckpointV1(
        trainer_type="xgboost",
        architecture_id="xgboost",
        input_schema=(
            CheckpointField(name="float_input", dtype="float32", shape=("batch", 2)),
        ),
        output_schema=output_schema,
        preprocessing={"type": "none"},
        task_type="classification" if classification else "regression",
        framework="xgboost",
        framework_version=xgboost.__version__,
        checkpoint_format_version=1,
    )
    source = ExportSource(
        source_kind="xgboost_result",
        model_object=booster,
        feature_schema={"feature_names": ["opaque_spend", "opaque_age"]},
        checkpoint_contract=contract,
    )
    draft = OfficialXGBoostONNXExporter().export(
        ExportContext(execution_id="test", node_id="onnx", artifact_dir=tmp_path),
        source,
        {},
        SimpleNamespace(typed_options={}, target=SimpleNamespace(name="onnx")),
    )
    path = tmp_path / draft.entrypoint
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert json.loads(metadata["feature_names"]) == ["opaque_spend", "opaque_age"]
    assert metadata["task_type"] == (
        "MULTICLASS_CLASSIFICATION"
        if multi
        else "BINARY_CLASSIFICATION"
        if classification
        else "REGRESSION"
    )
    assert metadata["objective"] == objective
    assert metadata["framework"] == "xgboost"
    assert metadata["num_features"] == "2"
    if classification:
        assert metadata["num_classes"] == ("3" if multi else "2")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(path), sess_options=options)
    assert tuple(item.name for item in session.get_outputs()) == tuple(
        field.name for field in contract.output_schema
    )
    values = session.run(None, {"float_input": features})
    native = booster.predict(matrix)
    actual = (
        values[1] if multi else values[1][:, 1] if classification else values[0][:, 0]
    )
    np.testing.assert_allclose(actual, native, atol=1e-6, rtol=1e-6)


def test_boosting_wheel_record_matches_all_payload_bytes():
    import base64
    import csv
    import hashlib
    import io

    with zipfile.ZipFile(_WHEEL) as archive:
        record_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        records = csv.reader(io.StringIO(archive.read(record_name).decode()))
        for name, digest, size in records:
            if name == record_name:
                assert digest == size == ""
                continue
            data = archive.read(name)
            actual = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .decode()
                .rstrip("=")
            )
            assert digest == "sha256=" + actual, name
            assert int(size) == len(data), name
