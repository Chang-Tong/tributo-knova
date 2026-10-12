"""Read-only source/native audit of the retained matrix ClickHouse fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import clickhouse_connect
import numpy as np
import xgboost


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit_training_rejections(root: Path, client) -> list[dict]:
    """Require the actual native failure site, not merely any FAILED event."""
    records = []
    for case in read(root / "matrix-results.json"):
        if not case["name"].startswith("reject_train_"):
            continue
        directory = root / "cases" / case["name"]
        request = read(directory / "request.redacted.json")
        terminal = read(directory / "terminal.json")
        log = (directory / "ray.log").read_text()
        table = request["datasource"]["properties"]["native_table"]
        assert re.fullmatch(r"knova\.matrix_[A-Za-z0-9_]+", table)
        rows = client.query(f"SELECT count() FROM {table}").first_row[0]
        assert rows == case["rows"] in (0, 1)
        assert terminal["event_type"] == "FAILED"
        assert terminal["error_code"] == "EXECUTION_FAILED"
        expected_message = case.get("expected_error_message") or (
            "XGBoost training execution failed (ValueError)"
        )
        assert terminal["error_message"] == expected_message
        assert "phase=EXECUTING error_type=_TrainingExecutionError" in log
        reason = (
            "training dataset is empty"
            if rows == 0
            else "Couldn't create non-empty splits with the given proportions."
        )
        target_function = "_split_dataset" if rows == 0 else "split_proportionately"
        evidence = []
        for path, line, function in re.findall(
            r'File "([^"]+)", line (\d+), in ([^\n]+)', log
        ):
            if function != target_function:
                continue
            source = Path(path).read_text().splitlines()
            number = int(line)
            if reason in "\n".join(source[number - 1 : number + 3]):
                evidence.append({"path": path, "line": number, "function": function})
        assert evidence, (
            "Training rejection has no data-insufficiency failure-site proof"
        )
        records.append(
            {
                "name": case["name"],
                "source_rows": rows,
                "error_code": terminal["error_code"],
                "phase": "EXECUTING",
                "native_exception_type": "ValueError",
                "confirmed_reason": reason,
                "native_failure_sites": evidence,
                "passed": True,
                "message_limitation": (
                    "FAILED message exposes type, not this safe reason"
                ),
            }
        )
    assert len(records) == 6
    return records


def native_model(model: dict) -> xgboost.Booster:
    item = next(
        item
        for alternative in model["model_artifacts"]["model_weights"]["alternatives"]
        if alternative["format"] == "xgboost"
        for item in alternative["files"]
        if item["path"].endswith(".ubj")
    )
    storage = model["storage"]
    key = storage["prefix"] + item["path"]
    if storage["type"] == "nas":
        content = (Path(storage["bucket"]) / key).read_bytes()
    else:
        content = (
            boto3.client("s3")
            .get_object(Bucket=storage["bucket"], Key=key)["Body"]
            .read()
        )
    assert "sha256:" + hashlib.sha256(content).hexdigest() == item["hash"]
    booster = xgboost.Booster(params={"nthread": 1})
    booster.load_model(bytearray(content))
    return booster


def audit_chunk(rows: list, case: dict, names: list[str], booster) -> dict:
    source = np.asarray([row[6:] for row in rows], dtype=np.float32)
    extras = [json.loads(row[5]) for row in rows]
    for extra in extras:
        assert [x["feature_name"] for x in extra["inference_feature_values"]] == names
    returned = np.asarray(
        [[x["value"] for x in e["inference_feature_values"]] for e in extras],
        dtype=np.float64,
    )
    assert np.array_equal(returned, source.astype(np.float64)), (
        "Returned feature values differ from entity-matched ClickHouse source"
    )
    matrix = xgboost.DMatrix(source, feature_names=names)
    prediction = booster.predict(matrix, strict_shape=True)
    shap = case["shap"] != "none"
    margins = (
        booster.predict(matrix, output_margin=True, strict_shape=True) if shap else None
    )
    contributions = (
        booster.predict(
            matrix,
            pred_contribs=True,
            approx_contribs=case["shap"] == "approximate",
            strict_shape=True,
        )
        if shap
        else None
    )
    maximum_prediction_error = 0.0
    maximum_margin_error = 0.0
    for i, (row, extra) in enumerate(zip(rows, extras, strict=True)):
        if case["kind"] == "binary":
            native = float(prediction[i, 0])
            if case["positive_index"] == 0:
                native = 1 - native
            delta = abs(row[2] - native)
            assert delta < 1e-5
        elif case["kind"] == "multi":
            delta = float(
                np.max(np.abs(np.asarray(json.loads(row[3])) - prediction[i]))
            )
            assert delta < 1e-5
        else:
            delta = abs(row[4] - float(prediction[i, 0]))
            assert delta < 1e-4
        maximum_prediction_error = max(maximum_prediction_error, delta)
        if not shap:
            continue
        explanation = extra["explanation"]
        assert [
            x["feature_name"] for x in explanation["feature_contributions"]
        ] == names
        group = explanation["explained_class_index"] if case["kind"] == "multi" else 0
        sign = -1 if case["kind"] == "binary" and case["positive_index"] == 0 else 1
        values = np.asarray(
            [
                *[x["shap_value"] for x in explanation["feature_contributions"]],
                explanation["base_value"],
            ]
        )
        assert np.allclose(values, sign * contributions[i, group], atol=1e-5)
        margin_error = abs(float(values.sum()) - sign * float(margins[i, group]))
        assert margin_error < 1e-4
        maximum_margin_error = max(maximum_margin_error, margin_error)
    return {
        "rows": len(rows),
        "maximum_prediction_error": maximum_prediction_error,
        "maximum_margin_error": maximum_margin_error,
    }


def main(root: Path, output: Path) -> int:
    started = time.monotonic()
    report = {"started_utc": datetime.now(UTC).isoformat(), "cases": []}
    client = clickhouse_connect.get_client(
        host="clickhouse",
        username="knova",
        password=os.environ["KNOVA_OUTPUT_CLICKHOUSE_PASSWORD"],
    )
    results = read(root / "matrix-results.json")
    assert len(results) == 54, "Post-audit requires all original cases completed"
    report["training_rejections"] = audit_training_rejections(root, client)
    write(output, report)
    for case in results:
        if case["operation"] != "inference" or case["expected"] != "COMPLETED":
            continue
        report["active_case"] = case["name"]
        write(output, report)
        directory = root / "cases" / case["name"]
        request = read(directory / "request.redacted.json")
        terminal = read(directory / "terminal.json")
        source_table = request["input"]["tables"][0]["table_name"]
        output_table = request["output"]["table_name"]
        identity = request["execution_id"]
        names = [feature["result_column"] for feature in request["input"]["features"]]
        assert names == [f"f{i}" for i in range(1, 117)]
        booster = native_model(request["model"])
        # Ray's numpy batches can produce a native Booster without embedded names;
        # the published sidecar supplies the checkpoint's ordered feature schema.
        sidecar = next(
            f
            for a in request["model"]["model_artifacts"]["model_weights"][
                "alternatives"
            ]
            if a["format"] == "xgboost"
            for f in a["files"]
            if f["path"].endswith("feature_names.json")
        )
        storage = request["model"]["storage"]
        key = storage["prefix"] + sidecar["path"]
        content = (
            (Path(storage["bucket"]) / key).read_bytes()
            if storage["type"] == "nas"
            else boto3.client("s3")
            .get_object(Bucket=storage["bucket"], Key=key)["Body"]
            .read()
        )
        assert "sha256:" + hashlib.sha256(content).hexdigest() == sidecar["hash"]
        assert json.loads(content) == names
        assert booster.feature_names is None or booster.feature_names == names
        count = client.query(
            f"SELECT count() FROM knova.{output_table} "
            "WHERE execution_id={id:String}",
            parameters={"id": identity},
        ).first_row[0]
        record = {
            "name": case["name"],
            "source_table": source_table,
            "source_feature_count": len(names),
            "output_rows": count,
            "source_matched_rows": 0,
            "native_prediction_checked_rows": 0,
            "native_shap_checked_rows": 0,
            "maximum_prediction_error": 0.0,
            "maximum_margin_error": 0.0,
        }
        query = (
            "SELECT r.entity_id,r.pred_label,r.pred_probability,r.pred_probabilities,"
            "r.pred_value,r.pred_extra,"
            + ",".join(f"s.{name}" for name in names)
            + f" FROM knova.{output_table} r INNER JOIN knova.{source_table} s "
            "ON toUInt64(r.entity_id)=s.entity_id "
            "WHERE r.execution_id={id:String} ORDER BY toUInt64(r.entity_id)"
        )
        with client.query_rows_stream(query, parameters={"id": identity}) as stream:
            batch = []
            for row in stream:
                batch.append(row)
                if len(batch) < 512:
                    continue
                update = audit_chunk(batch, case, names, booster)
                record["source_matched_rows"] += update["rows"]
                record["maximum_prediction_error"] = max(
                    record["maximum_prediction_error"],
                    update["maximum_prediction_error"],
                )
                record["maximum_margin_error"] = max(
                    record["maximum_margin_error"], update["maximum_margin_error"]
                )
                batch = []
            if batch:
                update = audit_chunk(batch, case, names, booster)
                record["source_matched_rows"] += update["rows"]
                record["maximum_prediction_error"] = max(
                    record["maximum_prediction_error"],
                    update["maximum_prediction_error"],
                )
                record["maximum_margin_error"] = max(
                    record["maximum_margin_error"], update["maximum_margin_error"]
                )
        assert record["source_matched_rows"] == count == terminal["result_rows"]
        record["native_prediction_checked_rows"] = count
        record["native_shap_checked_rows"] = count if case["shap"] != "none" else 0
        record["passed"] = True
        report["cases"].append(record)
        write(output, report)
        print(f"PASS {case['name']} source/native rows={count}", flush=True)
    client.close()
    report.update(
        passed=True,
        finished_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=round(time.monotonic() - started, 2),
        source_matched_rows=sum(x["source_matched_rows"] for x in report["cases"]),
        native_prediction_checked_rows=sum(
            x["native_prediction_checked_rows"] for x in report["cases"]
        ),
        native_shap_checked_rows=sum(
            x["native_shap_checked_rows"] for x in report["cases"]
        ),
        commands_are_read_only=True,
    )
    write(output, report)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.root, args.output))
