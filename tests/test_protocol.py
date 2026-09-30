import json

import pytest

from tributo_knova.protocol import (
    InferenceExecutionRequest,
    KnovaProtocolFailure,
    TrainingExecutionRequest,
    parse_request,
)


def test_parse_training_request_keeps_job_id_on_the_wire() -> None:
    from test_training import _request

    payload = _request().model_dump(mode="json")

    request = parse_request(
        json.dumps(payload),
        outer_operation_id="job-1",
        expected_operation_type="training",
    )

    assert isinstance(request, TrainingExecutionRequest)
    assert request.execution_id == "job-1"
    assert request.model_dump(mode="json")["job_id"] == "job-1"
    assert request.model_dump(mode="json")["datasource"] == payload["datasource"]


def test_parse_inference_accepts_a_higher_v2_minor() -> None:
    from test_training import _request

    train = _request().model_dump(mode="json")
    payload = {
        "protocol_version": "2.1",
        "execution_id": "inference-1",
        "task_id": "task-1",
        "tenant_id": "tenant-1",
        "model": {
            "model_id": "model-1",
            "version_id": "v1",
            "algorithm_key": "xgboost",
            "algorithm_category": "CLASSIFICATION",
            "task_type": "BINARY_CLASSIFICATION",
            "storage": train["storage_context"],
            "model_artifacts": {"model_weights": {}},
        },
        "input": {
            "datasource": train["datasource"],
            "tables": train["tables"],
            "features": train["features"],
            "query": train["data_query"]["query"],
            "entity_key": {
                "origin": {
                    "table_alias": "t0",
                    "column_name": "id",
                    "column_type": "Int64",
                },
                "result_column": "opaque_entity",
            },
        },
        "output": {
            "datasource": train["datasource"],
            "table_name": "ml_inference_result",
        },
        "new_minor_field": {"kept": True},
    }

    request = parse_request(
        json.dumps(payload),
        outer_operation_id="inference-1",
        expected_operation_type="batch_inference",
    )

    assert isinstance(request, InferenceExecutionRequest)
    assert request.model_dump(mode="json")["new_minor_field"] == {"kept": True}


@pytest.mark.parametrize("version", ["1.9", "3.0", "2", "latest"])
def test_reject_unsupported_or_malformed_protocol_versions(version: str) -> None:
    payload = {
        "protocol_version": version,
        "execution_id": "inference-1",
    }

    with pytest.raises(KnovaProtocolFailure) as captured:
        parse_request(
            json.dumps(payload),
            outer_operation_id="inference-1",
            expected_operation_type="batch_inference",
        )

    assert captured.value.code == "UNSUPPORTED_PROTOCOL_VERSION"
    assert version not in captured.value.sanitized_message


def test_reject_identity_mismatch() -> None:
    payload = {
        "protocol_version": "2.0",
        "job_id": "job-inside",
        "model_id": "model-1",
        "version_id": "version-1",
        "tenant_id": "tenant-1",
        "algorithm": {"algorithm_key": "xgboost"},
    }

    with pytest.raises(KnovaProtocolFailure) as captured:
        parse_request(
            json.dumps(payload),
            outer_operation_id="job-outside",
            expected_operation_type="training",
        )

    assert captured.value.code == "INVALID_PAYLOAD"
    assert "job-inside" not in captured.value.sanitized_message


def test_reject_invalid_json_without_echoing_payload() -> None:
    with pytest.raises(KnovaProtocolFailure) as captured:
        parse_request(
            '{"secret":"not-json"',
            outer_operation_id="job-1",
            expected_operation_type="training",
        )

    assert captured.value.code == "INVALID_PAYLOAD"
    assert "secret" not in captured.value.sanitized_message


@pytest.mark.parametrize(
    "mutate,field",
    [
        (
            lambda p: p["features"][1].update(
                feature_id=p["features"][0]["feature_id"]
            ),
            "unique",
        ),
        (
            lambda p: p["features"][1].update(
                result_column=p["features"][0]["result_column"]
            ),
            "unique",
        ),
        (
            lambda p: p["features"][0]["origin"].update(table_alias="unregistered"),
            "table_alias",
        ),
        (lambda p: p["tables"][0].update(role="JOINED"), "PRIMARY"),
        (lambda p: p["algorithm"].update(category="CLUSTERING"), "algorithm.category"),
        (lambda p: p.update(data_sampling={"sample_ratio": 100}), "sample_ratio"),
        (lambda p: p.update(data_sampling={"sample_ratio": True}), "sample_ratio"),
        (lambda p: p.update(data_sampling={"sample_limit": 0}), "sample_limit"),
        (lambda p: p.update(tuning={"mode": "INVALID"}), "tuning.mode"),
        (
            lambda p: p.update(tuning={"mode": "MANUAL", "auto_config": {}}),
            "auto_config",
        ),
        (lambda p: p["data_query"].update(query={"sql": ""}), "sql"),
        (lambda p: p["data_query"]["query"].update(params={"p": 123}), "params"),
        (lambda p: p["datasource"].pop("port"), "port"),
    ],
)
def test_training_admission_rejects_invalid_nested_payload(mutate, field):
    from test_training import _request

    payload = _request().model_dump(mode="json")
    mutate(payload)
    with pytest.raises(KnovaProtocolFailure, match=field) as raised:
        parse_request(
            json.dumps(payload),
            outer_operation_id="job-1",
            expected_operation_type="training",
        )
    assert raised.value.code == "INVALID_PAYLOAD"


@pytest.mark.parametrize(
    "extensions",
    [
        None,
        "opaque-v3",
        [1, 2],
        {"tributo": "opaque-v3"},
        {"tributo": {"training_runtime": {"ray": {"num_workers": "future"}}}},
    ],
)
def test_extensions_never_invalidate_admission_or_training_mapping(extensions):
    from test_training import _request
    from tributo_knova.training import _build_execution

    payload = _request(extensions=extensions).model_dump(mode="json")
    request = parse_request(
        json.dumps(payload),
        outer_operation_id="job-1",
        expected_operation_type="training",
    )
    assert _build_execution(request)[0].worker_count == 2


def test_relation_graph_is_checked_before_execution():
    from test_training import _request
    from tributo_knova.validation import (
        ProtocolValidationError,
        validate_training_payload,
    )

    payload = _request().model_dump(mode="python")
    payload["tables"].append(
        {
            "table_alias": "t1",
            "database_name": "analytics",
            "table_name": "other",
            "role": "JOINED",
        }
    )
    with pytest.raises(ProtocolValidationError, match="relations"):
        validate_training_payload(payload)
    payload["relations"] = [
        {
            "left_alias": "t0",
            "right_alias": "t1",
            "join_type": "LEFT",
            "on": [{"left_column": "id", "right_column": "id"}],
        }
    ]
    with pytest.raises(ProtocolValidationError, match="entity_key"):
        validate_training_payload(payload)
    payload["data_query"]["entity_key"] = {
        "origin": {"table_alias": "t0", "column_name": "id", "column_type": "UInt64"},
        "result_column": "opaque_id",
    }
    validate_training_payload(payload)
    payload["relations"][0]["left_alias"] = "t1"
    with pytest.raises(ProtocolValidationError, match="relations"):
        validate_training_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["sampler"].update(type="UNKNOWN"),
        lambda a: a["budget"].update(max_trials=0),
        lambda a: a.update(search_space={}),
        lambda a: a["search_space"].update(unknown={"type": "INT", "min": 1, "max": 5}),
        lambda a: a["search_space"]["max_depth"].update(max=100),
        lambda a: a["search_space"]["max_depth"].update(type="FLOAT"),
        lambda a: a["sampler"].update(type="GRID"),
        lambda a: a.update(
            scheduler={
                "type": "ASHA",
                "resource_param": "max_depth",
                "min_resource": 1,
                "max_resource": 3,
            }
        ),
    ],
)
def test_auto_search_contract_validation(mutation):
    from test_training import _request
    from tributo_knova.validation import (
        ProtocolValidationError,
        validate_training_payload,
    )

    auto = {
        "sampler": {"type": "RANDOM"},
        "budget": {"max_trials": 2},
        "search_space": {"max_depth": {"type": "INT", "min": 2, "max": 5}},
    }
    payload = _request(tuning={"mode": "AUTO", "auto_config": auto}).model_dump(
        mode="python"
    )
    validate_training_payload(payload)
    mutation(payload["tuning"]["auto_config"])
    with pytest.raises(ProtocolValidationError):
        validate_training_payload(payload)


@pytest.mark.parametrize("operation", ["training", "inference"])
@pytest.mark.parametrize(
    "mapping",
    [
        {"a": 0, "b": 0, "c": 2},
        {"a": 0, "b": 2},
        {"a": -1, "b": 0},
        {"a": False, "b": True},
        {"a": 0, "b": 1.0},
        {"a": "0", "b": "1"},
    ],
)
def test_label_mapping_invalid_codes_are_rejected_for_both_operations(
    operation, mapping
):
    from test_training import _request
    from tributo_knova.validation import (
        ProtocolValidationError,
        validate_inference_payload,
        validate_training_payload,
    )

    payload = _request().model_dump(mode="python")
    if operation == "training":
        payload["target"]["label_mapping"] = mapping
        validate = validate_training_payload
    else:
        payload = _inference_mapping_payload(payload, mapping)
        validate = validate_inference_payload
    with pytest.raises(ProtocolValidationError, match="label_mapping"):
        validate(payload)


def _inference_mapping_payload(train, mapping):
    return {
        "model": {
            "model_id": "model",
            "version_id": "v1",
            "algorithm_key": "xgboost",
            "algorithm_category": "CLASSIFICATION",
            "task_type": "MULTICLASS_CLASSIFICATION",
            "label_mapping": mapping,
            "storage": train["storage_context"],
            "model_artifacts": {"model_weights": {}},
        },
        "input": {
            "datasource": train["datasource"],
            "tables": train["tables"],
            "features": train["features"],
            "query": train["data_query"]["query"],
            "entity_key": {
                "origin": {
                    "table_alias": "t0",
                    "column_name": "id",
                    "column_type": "UInt64",
                },
                "result_column": "id",
            },
        },
        "output": {"datasource": train["datasource"], "table_name": "result"},
    }


@pytest.mark.parametrize("mapping", [None, {}, {"a": 0, "b": 1, "c": 2}])
def test_numeric_label_defaults_and_contiguous_mapping_remain_valid(mapping):
    from test_training import _request
    from tributo_knova.validation import (
        validate_inference_payload,
        validate_training_payload,
    )

    training = _request().model_dump(mode="python")
    training["target"]["label_mapping"] = mapping
    validate_training_payload(training)
    inference = _inference_mapping_payload(training, mapping)
    validate_inference_payload(inference)
    training["target"].pop("label_mapping")
    inference["model"].pop("label_mapping")
    validate_training_payload(training)
    validate_inference_payload(inference)
