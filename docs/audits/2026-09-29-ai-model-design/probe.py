"""Read-only protocol audit; isolated temporary Redis, no Ray jobs or business DB.

Run with the KnoVa environment and the desired source PYTHONPATH. Use --no-redis
inside the release image to exercise exactly its installed mapping code.
FAIL means a protocol expectation is unmet, not that the probe crashed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from tributo_knova import inference, training, execution_driver
from tributo_knova.broker import parse_broker_request, prepare_broker_operation
from tributo_knova.protocol import TrainingExecutionRequest, InferenceExecutionRequest, parse_request
from tributo_knova.reporter import KnovaRedisEventReporter

RESULTS = []


def check(name, expected, fn, satisfies, category="contract"):
    try:
        actual = fn()
        passed = satisfies(actual)
    except Exception as exc:
        actual = {"exception": type(exc).__name__, "message": str(exc)[:250]}
        passed = False
    RESULTS.append({"id": name, "category": category, "expected": expected,
                    "status": "PASS" if passed else "FAIL", "actual": actual})


def ds():
    return {"type": "CLICKHOUSE", "host": "localhost", "port": 8123,
            "database_name": "audit", "properties": {}}


def feature(column, feature_id):
    return {"feature_id": feature_id, "display_name": column,
            "feature_role": "REGULAR", "origin": {"table_alias": "t0",
            "column_name": column, "column_type": "Float64"},
            "result_column": "opaque_" + column}


def train():
    return {"protocol_version": "2.0", "job_id": "audit-train", "model_id": "m",
            "version_id": "v", "tenant_id": "t", "algorithm": {
                "category": "CLASSIFICATION", "algorithm_key": "xgboost",
                "hyper_params": {"n_estimators": 5}}, "datasource": ds(),
            "tables": [{"table_alias": "t0", "database_name": "audit",
                        "table_name": "input", "role": "PRIMARY"}],
            "data_query": {"mode": "DIRECT_QUERY", "query": {
                "sql": "SELECT t0.x AS opaque_x, t0.z AS opaque_z, t0.y AS opaque_y FROM audit.input t0",
                "params": {}}}, "features": [feature("x", "f1"), feature("z", "f2")],
            "target": {"display_name": "y", "origin": {"table_alias": "t0",
                "column_name": "y", "column_type": "String"}, "result_column": "opaque_y",
                "task_type": "BINARY_CLASSIFICATION", "label_mapping": {"no": 0, "yes": 1},
                "positive_label_value": "yes"},
            "data_split": {"strategy": "RANDOM", "train_ratio": .7, "validation_ratio": .1,
                "test_ratio": .2, "stratify": False, "cross_validation": {"enabled": False}},
            "tuning": {"mode": "MANUAL", "auto_config": None},
            "storage_context": {"type": "nas", "bucket": "/tmp/protocol-audit-models", "prefix": "t/m/v/"},
            "evaluation": {"primary_metric": "auc"}}


def infer():
    return {"protocol_version": "2.0", "execution_id": "audit-infer", "task_id": "task",
            "tenant_id": "t", "model": {"model_id": "m", "version_id": "v",
                "algorithm_category": "CLASSIFICATION", "algorithm_key": "xgboost",
                "task_type": "BINARY_CLASSIFICATION", "label_mapping": {"no": 0, "yes": 1},
                "positive_label_value": "yes", "storage": {"type": "s3", "bucket": "audit",
                "prefix": "t/m/v/", "properties": {}}, "model_artifacts": {"model_weights": {
                    "required_for_inference": True, "alternatives": [{"format": "onnx", "files": [
                    {"path": "model.onnx", "hash": "sha256:" + "a" * 64, "size": 100}]}]}}},
            "input": {"datasource": ds(), "tables": train()["tables"],
                "features": train()["features"], "entity_key": {"origin": {
                    "table_alias": "t0", "column_name": "id", "column_type": "Int64"},
                    "result_column": "opaque_id"}, "query": {
                        "sql": "SELECT t0.id AS opaque_id, t0.x AS opaque_x, t0.z AS opaque_z FROM audit.input t0",
                        "params": {}}},
            "output": {"datasource": ds(), "table_name": "ml_inference_result"},
            "execution": {"batch_size": 5}}


def train_plan(p):
    return training._build_execution(TrainingExecutionRequest.model_validate(p))[0]


def infer_plan(p):
    return inference._build_request(InferenceExecutionRequest.model_validate(p))[0]


def rejection(fn):
    try:
        fn()
    except Exception as exc:
        return {"rejected": True, "code": getattr(exc, "code", None),
                "exception": type(exc).__name__, "message": str(exc)[:200]}
    return {"rejected": False}


class MemoryRedis:
    def __init__(self):
        self.events = []

    def xadd(self, stream, fields, **kwargs):
        self.events.append((stream, fields))
        return "1-0"


def reporter(redis, op="batch_inference", identity="audit-infer", prefix="audit:events", **kw):
    return KnovaRedisEventReporter(redis, event_stream_prefix=prefix, operation_id=identity,
        operation_type=op, execution_profile="distributed", run_id=identity,
        attempt_id="attempt-1", outer_identity_field="job_id" if op == "training" else "execution_id", **kw)


def mapping_checks():
    check("compatible_minor", "accept 2.9 and unknown field", lambda: parse_request(
        json.dumps({**train(), "protocol_version": "2.9", "future_field": {"x": 1}}),
        outer_operation_id="audit-train", expected_operation_type="training").protocol_version,
        lambda x: x == "2.9")
    p = train(); p.pop("protocol_version")
    check("missing_version", "UNSUPPORTED_PROTOCOL_VERSION", lambda: rejection(lambda: parse_request(
        json.dumps(p), outer_operation_id="audit-train", expected_operation_type="training")),
        lambda x: x.get("code") == "UNSUPPORTED_PROTOCOL_VERSION")
    check("baseline_training_plan", "supported numeric single-table plan", lambda: train_plan(train()).algorithm_request.algorithm,
          lambda x: x == "xgboost")
    check("baseline_inference_plan", "protocol model_weights+ONNX accepted", lambda: infer_plan(infer()).model.kind,
          lambda x: x == "artifact")
    p = train(); p["data_split"]["stratify"] = True
    check("classification_stratify", "valid stratified classification accepted", lambda: rejection(lambda: train_plan(p)),
          lambda x: not x["rejected"], "capability")
    for name, mutate in [
        ("duplicate_feature_id", lambda p: p["features"][1].update(feature_id="f1")),
        ("duplicate_result_column", lambda p: p["features"][1].update(result_column="opaque_x")),
        ("undeclared_origin", lambda p: p["features"][0]["origin"].update(table_alias="missing")),
        ("no_primary_table", lambda p: p["tables"][0].update(role="JOINED")),
        ("category_mismatch", lambda p: p["algorithm"].update(category="CLUSTERING")),
        ("missing_entity_column", lambda p: p["data_query"].update(entity_key={"origin": {
            "table_alias": "t0", "column_name": "id", "column_type": "Int64"}, "result_column": "not_selected"})),
        ("invalid_sampling_ratio", lambda p: p.update(data_sampling={"sample_ratio": 100})),
        ("invalid_tuning_mode", lambda p: p.update(tuning={"mode": "INVALID"})),
    ]:
        p = train(); mutate(p)
        check(name, "reject invalid payload before execution", lambda: rejection(lambda: train_plan(p)), lambda x: x["rejected"])
    p = train(); p["resource_limits"] = {"max_training_time_seconds": 1, "early_stopping_patience": 3}
    check("training_controls", "timeout and protocol patience reach execution plan", lambda: dict(train_plan(p).algorithm_request.algorithm_config["training"]),
          lambda x: x.get("early_stopping_rounds") == 3 and x.get("max_training_time_seconds") == 1)
    p = train(); p["extensions"] = {"tributo": "future-opaque-content"}
    check("extension_tolerance", "arbitrary extension content does not reject task", lambda: rejection(lambda: train_plan(p)), lambda x: not x["rejected"])
    p = infer(); p["input"]["query"]["sql"] += " WHERE t0.id = {p:Int64}"; p["input"]["query"]["params"] = {"p": "42"}
    check("typed_sql_parameter", "Int64 parameter remains integer 42", lambda: {
        "value": infer_plan(p).input.transforms.steps[0].value,
        "type": type(infer_plan(p).input.transforms.steps[0].value).__name__}, lambda x: x == {"value": 42, "type": "int"})
    p = infer(); p["input"].pop("query")
    check("required_query", "reject missing authoritative SQL", lambda: rejection(lambda: infer_plan(p)), lambda x: x["rejected"])
    p = infer(); p["input"]["query"]["sql"] = "SELECT t0.id AS opaque_id, t0.x * 100 AS opaque_x, t0.z AS opaque_z FROM audit.input t0"
    def projection_observation():
        try:
            core = infer_plan(p)
        except inference._InferenceConfigurationError:
            return {"rejected": True}
        return {"columns": list(core.input_binding.tensors[0].columns),
                "transforms": core.input.transforms.model_dump(mode="json")}
    check("sql_projection_semantics", "preserve expression or reject unsupported SQL", projection_observation,
        lambda x: x.get("rejected", False) or any("100" in str(s) for s in x["transforms"].get("steps", [])))
    def feature_order():
        import numpy as np
        from tributo.inference.kernel import _build_input_tensor
        p = infer()
        first = infer_plan(p).input_binding.tensors[0]
        p["input"]["features"].reverse()
        second = infer_plan(p).input_binding.tensors[0]
        batch = {"x": np.array([1.0]), "z": np.array([10.0])}
        def matrix(binding):
            return _build_input_tensor(batch, columns=binding.columns, dtype=binding.dtype,
                single_column_mode=binding.single_column_mode, null_policy="error", nan_policy="error").tolist()
        return {"original": matrix(first), "permuted_feature_declarations": matrix(second)}
    check("artifact_feature_order", "reordering declarations must not silently permute the model tensor", feature_order,
          lambda x: x["original"] == x["permuted_feature_declarations"])
    for name, mutate in [
        ("missing_file_hash", lambda p: p["model"]["model_artifacts"]["model_weights"]["alternatives"][0]["files"][0].pop("hash")),
        ("missing_prefix_slash", lambda p: p["model"]["storage"].update(prefix="t/m/v")),
        ("unknown_artifact_key", lambda p: p["model"]["model_artifacts"].update(unregistered={"required_for_inference": True})),
        ("required_preprocessor", lambda p: p["model"]["model_artifacts"].update(preprocessor={
            "required_for_inference": True, "alternatives": [{"format": "sklearn", "files": [{
            "path": "pre.joblib", "size": 10, "hash": "sha256:" + "b" * 64}]}]})),
        ("duplicate_alternative", lambda p: p["model"]["model_artifacts"]["model_weights"]["alternatives"].append(
            copy.deepcopy(p["model"]["model_artifacts"]["model_weights"]["alternatives"][0]))),
    ]:
        p = infer(); mutate(p)
        check(name, "reject invalid/unsupported mandatory artifact", lambda: rejection(lambda: infer_plan(p)), lambda x: x["rejected"])
    with patch.dict(os.environ, {"AUDIT_FAKE_CREDENTIAL": "reference-fixture"}):
        check("credential_ref_precedence", "credential_ref wins over inline fixture", lambda: inference._credentials(
            {"password": "inline-fixture", "credential_ref": "env://AUDIT_FAKE_CREDENTIAL"}, inline_password_env="AUDIT_INLINE")[1],
            lambda x: x == "reference-fixture")
    check("decimal_json_safety", "Decimal feature serializes to native JSON number", lambda: json.dumps({"value": inference._json_value(Decimal("1.25"))}),
          lambda x: json.loads(x)["value"] == 1.25)
    check("training_invalid_error_code", "INVALID_PAYLOAD", lambda: execution_driver._failure_code(training._TrainingConfigurationError("safe fixture")),
          lambda x: x == "INVALID_PAYLOAD")
    check("inference_started", "STARTED with measured total_rows", lambda: reporter(MemoryRedis()).publish("STARTED", {"total_rows": 25}),
          lambda x: x["event_type"] == "STARTED" and x.get("total_rows") == 25)
    check("queued_cancel_phase", "inference phase uses valid INIT enum", lambda: reporter(MemoryRedis()).publish("CANCELLED", {}, phase="QUEUED"),
          lambda x: x["phase"] == "INIT")
    def fake_inference():
        p = infer(); p["execution"].update(max_execution_time_seconds=1, progress_report_interval_rows=5)
        r = MemoryRedis()
        result = SimpleNamespace(status="succeeded", failure=None, output_rows=20, sink_receipt=None, model_dump=lambda **k: {})
        with patch.object(inference, "_measure_input_rows", return_value=20), patch.object(inference, "run_inference", return_value=result):
            inference.execute_inference(InferenceExecutionRequest.model_validate(p), reporter(r))
        return [json.loads(fields["payload"]) for _, fields in r.events]
    check("inference_final_progress", "explicit final PROGRESS=100 and STARTED", fake_inference,
          lambda events: any(e["event_type"] == "STARTED" for e in events) and any(e["event_type"] == "PROGRESS" and e["percent"] == 100 for e in events))
    # Synthetic clocks, not a wall-clock wait: start at 100, return at 110.
    def elapsed_inference():
        p = infer(); p["execution"]["max_execution_time_seconds"] = 1
        r = MemoryRedis()
        result = SimpleNamespace(status="succeeded", failure=None, output_rows=20, sink_receipt=None, model_dump=lambda **k: {})
        with patch("tributo_knova.reporter.time.monotonic", side_effect=[100.0, 110.0]), patch.object(inference, "_measure_input_rows", return_value=20), patch.object(inference, "run_inference", return_value=result):
            inference.execute_inference(InferenceExecutionRequest.model_validate(p), reporter(r))
        return json.loads(r.events[-1][1]["payload"])
    check("inference_timeout", "elapsed 10s with limit 1s must fail EXECUTION_TIMEOUT", elapsed_inference,
          lambda x: x.get("event_type") == "FAILED" and x.get("error_code") == "EXECUTION_TIMEOUT")


def redis_checks():
    import redis
    import tributo
    from tributo_broker_redis.config import normalize_config
    from tributo_broker_redis.runtime import RedisBrokerRuntime
    executable = shutil.which("redis-server")
    if not executable:
        raise RuntimeError("redis-server is required unless --no-redis is specified")
    with tempfile.TemporaryDirectory(prefix="knova-protocol-audit-") as directory:
        sock = str(Path(directory) / "redis.sock")
        process = subprocess.Popen([executable, "--port", "0", "--unixsocket", sock,
            "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = redis.Redis(unix_socket_path=sock, decode_responses=True)
        try:
            for _ in range(100):
                try:
                    if client.ping():
                        break
                except redis.ConnectionError:
                    time.sleep(.02)
            def concurrent_terminals():
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(lambda i: reporter(client, identity="race").publish(
                        ["COMPLETED", "FAILED", "CANCELLED"][i % 3], {}), range(24)))
                reporter(client, identity="race").publish("LOG", {"message": "late"})
                return len(client.xrange("audit:events:race"))
            check("redis_terminal_exclusion", "24 concurrent terminal writes + late LOG produce one entry", concurrent_terminals, lambda x: x == 1)
            def bounded_stream():
                r = reporter(client, identity="maxlen", max_stream_length=5)
                for _ in range(12):
                    r.publish("LOG", {"message": "fixture"})
                return client.xlen("audit:events:maxlen")
            check("redis_event_maxlen", "MAXLEN=5 enforced", bounded_stream, lambda x: x == 5)
            source_root = str(Path(tributo.__file__).resolve().parent.parent)
            channel = lambda op, field: {"task_stream_key": f"audit:{op}:tasks", "event_stream_prefix": f"audit:{op}:events",
                "cancel_key_prefix": f"audit:{op}:cancel", "consumer_group": f"audit:{op}", "consumer_name": "auditor", "outer_identity_field": field}
            cfg = normalize_config({"broker_id": "tributo-redis", "api_version": 1,
                "transport": {"url": "redis://127.0.0.1:1/0", "claim_idle_ms": 1},
                "channels": {"training": channel("training", "job_id"), "batch_inference": channel("inference", "execution_id")},
                "operations": {"training": {"execution_profiles": ["distributed"]}, "batch_inference": {"execution_profiles": ["distributed"]}},
                "execution": {"project_root": source_root, "ray_dashboard_url": "http://127.0.0.1:1",
                    "extra_py_modules": [str(Path(training.__file__).resolve().parent)]}})
            runtime = RedisBrokerRuntime(cfg, redis_client=client, request_parser=parse_broker_request,
                operation_preparer=prepare_broker_operation, reporter_factory=KnovaRedisEventReporter,
                submitter=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected Ray submission")))
            consumer = runtime.consumers["training"]
            def deliver(payload, identity):
                client.xadd("audit:training:tasks", {"job_id": identity, "payload": payload})
                runtime.run_once(timeout_ms=0)
                events = client.xrange(f"audit:training:events:{identity}")
                return {"event": json.loads(events[-1][1]["payload"]) if events else None,
                        "pending": client.xpending("audit:training:tasks", "audit:training")["pending"]}
            check("redis_incompatible_version_ack", "FAILED UNSUPPORTED_PROTOCOL_VERSION and ACK", lambda: deliver(
                json.dumps({**train(), "job_id": "badversion", "protocol_version": "3.0"}), "badversion"),
                lambda x: x["pending"] == 0 and x["event"]["error_code"] == "UNSUPPORTED_PROTOCOL_VERSION")
            client.set("audit:training:cancel:precancel", "")
            check("redis_empty_cancel_value", "existing empty cancel marker gives CANCELLED+ACK", lambda: deliver(
                json.dumps({**train(), "job_id": "precancel"}), "precancel"),
                lambda x: x["pending"] == 0 and x["event"]["event_type"] == "CANCELLED")
            client.set("audit:training:cancel:cancelbad", "")
            check("cancel_before_schema_validation", "queued cancellation wins before malformed payload validation", lambda: deliver("{", "cancelbad"),
                lambda x: x["event"]["event_type"] == "CANCELLED")
            # Recovery is transport-only: claim a known pending entry and ACK it.
            def recover():
                client.xadd("audit:training:tasks", {"job_id": "recover", "payload": json.dumps({**train(), "job_id": "recover"})})
                consumer.poll(0)
                consumer.transport = consumer.transport.model_copy(update={"claim_idle_ms": 1})
                time.sleep(.01)
                count = consumer.recover_pending()
                if count:
                    consumer.ack(consumer.poll(0))
                return count
            check("redis_pending_recovery", "XAUTOCLAIM recovers staged pending delivery", recover, lambda x: x >= 1)
            def cancellation_snapshot():
                from tributo.ray_jobs import RayJobSubmission
                from tributo_broker_redis.cancellation import ActiveSubmission, ActiveSubmissionMap, CancelWatcher
                active = ActiveSubmissionMap()
                active.put(ActiveSubmission(operation_id="cancel-snapshot", operation_type="batch_inference",
                    execution_profile="distributed", run_id="cancel-snapshot", channel=cfg.channels.batch_inference,
                    submission=RayJobSubmission(run_id="cancel-snapshot", attempt_id="attempt-1", submission_id="fixture-submission")))
                reporter(client, identity="cancel-snapshot", prefix="audit:inference:events").publish("PROGRESS", {
                    "processed_rows": 8, "result_rows": 4, "total_rows": 10, "percent": 80.0}, phase="WRITE_RESULT")
                client.set("audit:inference:cancel:cancel-snapshot", "1")
                statuses = iter(["RUNNING", "STOPPED"])
                watcher = CancelWatcher(client, active, dashboard_url="http://127.0.0.1:1", interval_seconds=1,
                    max_event_bytes=1048576, max_stream_length=1000, status_getter=lambda *a, **k: next(statuses),
                    stopper=lambda *a, **k: True, reporter_factory=KnovaRedisEventReporter)
                watcher.check_once(); watcher.check_once()
                return json.loads(client.xrange("audit:inference:events:cancel-snapshot")[-1][1]["payload"])
            check("cancel_progress_snapshot", "CANCELLED preserves 8 processed / 4 written / 10 total and WRITE_RESULT", cancellation_snapshot,
                lambda x: (x["processed_rows"], x["result_rows"], x["total_rows"], x["phase"]) == (8, 4, 10, "WRITE_RESULT"))
            check("invalid_utf8_contained", "invalid UTF-8 must not escape the consume loop", lambda: deliver(b"\xff", "invalid-utf8"),
                lambda x: x["event"] is not None and x["pending"] == 0)
        finally:
            client.close()
            process.terminate()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-redis", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mapping_checks()
    if not args.no_redis:
        redis_checks()
    data = {"scope": "mapping, synthetic execution, isolated Redis; no business database or Ray jobs",
            "summary": {status: sum(r["status"] == status for r in RESULTS) for status in ("PASS", "FAIL")},
            "results": RESULTS}
    encoded = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
