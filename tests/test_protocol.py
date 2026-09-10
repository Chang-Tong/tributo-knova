import json

import pytest

from tributo_knova.protocol import (
    InferenceExecutionRequest,
    KnovaProtocolFailure,
    TrainingExecutionRequest,
    parse_request,
)


def test_parse_training_request_keeps_job_id_on_the_wire() -> None:
    payload = {
        "protocol_version": "2.0",
        "job_id": "job-1",
        "model_id": "model-1",
        "version_id": "version-1",
        "tenant_id": "tenant-1",
        "algorithm": {"algorithm_key": "xgboost"},
        "datasource": {"type": "CLICKHOUSE", "host": "clickhouse"},
    }

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
    payload = {
        "protocol_version": "2.1",
        "execution_id": "inference-1",
        "task_id": "task-1",
        "tenant_id": "tenant-1",
        "model": {"algorithm_key": "xgboost"},
        "input": {"expected_total_rows": 10},
        "output": {"table_name": "ml_inference_result"},
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

    assert captured.value.code == "IDENTITY_MISMATCH"
    assert "job-inside" not in captured.value.sanitized_message


def test_reject_invalid_json_without_echoing_payload() -> None:
    with pytest.raises(KnovaProtocolFailure) as captured:
        parse_request(
            '{"secret":"not-json"',
            outer_operation_id="job-1",
            expected_operation_type="training",
        )

    assert captured.value.code == "INVALID_JSON"
    assert "secret" not in captured.value.sanitized_message
