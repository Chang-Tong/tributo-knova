"""Summarize one completed isolated business matrix without changing its evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def inventory(path: Path) -> dict[str, tuple[str, str]]:
    return {
        fields[0]: (fields[1], fields[2])
        for line in path.read_text().splitlines()
        if len(fields := line.split()) == 3
    }


def summarize(root: Path) -> dict[str, Any]:
    launch = read(root / "launch.json")
    if "finished_utc" not in launch:
        raise RuntimeError("Matrix is still running; refusing a final summary")
    plan = read(root / "plan.json")
    results = read(root / "matrix-results.json")
    expected_names = {case["name"] for case in plan}
    actual_names = {case["name"] for case in results}
    case_counts_match = len(results) == len(plan) == len(actual_names)
    before = inventory(root / "containers-before.txt")
    after = inventory(root / "containers-after.txt")
    retained = all(after.get(key) == value for key, value in before.items())
    own_remaining = [
        name for name, _state in after.values() if name.startswith(launch["project"])
    ]
    images = read(root / "running-image-provenance.json")
    runtime_images = [
        item
        for item in images
        if item["service"] in {"ray-head", "ray-worker", "workloads"}
    ]
    pinned_images = len(runtime_images) == 3 and all(
        item["image_id"] == launch["image_id"] for item in runtime_images
    )
    unexpected = [case["name"] for case in results if not case["passed"]]
    shap_rows = 0
    processed_rows = 0
    written_rows = 0
    trained_rows = 0
    source_verified_rows = 0
    native_prediction_rows = 0
    max_margin_error = 0.0
    rows = []
    ray_statuses: dict[str, int] = {}
    missing_driver_evidence = []
    silent_expected_failures = []
    terminal_count_mismatches = []
    ray_status_mismatches = []
    driver_log_audit = []
    driver_input_bytes = []
    unknown_inference_status = []
    unknown_ray_job_id = []
    phase_events: set[str] = set()
    credential_marker_hits = []
    for case in results:
        directory = root / "cases" / case["name"]
        terminal = (
            read(directory / "terminal.json")
            if (directory / "terminal.json").exists()
            else {}
        )
        ray = (
            read(directory / "ray-status.json")
            if (directory / "ray-status.json").exists()
            else {}
        )
        status = ray.get("status", "not submitted")
        log = (
            (directory / "ray.log").read_text()
            if (directory / "ray.log").exists()
            else ""
        )
        events = (
            read(directory / "events.json")
            if (directory / "events.json").exists()
            else []
        )
        if "KnoVa inference execution finished:" in log and "status=unknown" in log:
            unknown_inference_status.append(case["name"])
        if "ray_job_id=<unknown>" in log:
            unknown_ray_job_id.append(case["name"])
        terminals = [
            event
            for event in events
            if event["event_type"] in {"FAILED", "COMPLETED", "CANCELLED"}
        ]
        if len(terminals) != 1:
            terminal_count_mismatches.append(case["name"])
        phase_events.update(
            str(event["phase"]) for event in events if event.get("phase")
        )
        markers = {
            "module_path": "KnoVa execution driver module loaded: path=",
            "operation_and_submission": "Driver input loaded: operation_id=",
            "runtime": "Ray driver runtime initialized:",
            "phase_and_error": "Ray driver failed:",
            "failed_event_publication": "FAILED event published:",
            "successful_completion": "Ray driver completed successfully:",
        }
        encoded_sizes = [
            int(size) for size in re.findall(r"\bencoded_input_bytes=(\d+)\b", log)
        ]
        driver_input_bytes.extend(encoded_sizes)
        driver_log_audit.append(
            {
                "case": case["name"],
                "log": f"cases/{case['name']}/ray.log",
                "line_numbers": {
                    key: [
                        number
                        for number, line in enumerate(log.splitlines(), 1)
                        if marker in line
                    ]
                    for key, marker in markers.items()
                },
                "event_phases": sorted(
                    {str(event["phase"]) for event in events if event.get("phase")}
                ),
                "terminal_count": len(terminals),
                "terminal": terminal.get("event_type"),
                "error_code": terminal.get("error_code"),
                "ray_status": status,
                "driver_input_bytes": encoded_sizes,
            }
        )
        expected_ray_status = {"COMPLETED": "SUCCEEDED", "FAILED": "FAILED"}.get(
            terminal.get("event_type")
        )
        if expected_ray_status is not None and status != expected_ray_status:
            ray_status_mismatches.append(case["name"])
        if ray and not all(
            marker in log
            for marker in (
                "KnoVa execution driver module loaded: path=",
                "Driver input loaded: operation_id=",
                "submission_id=",
                "Ray driver runtime initialized:",
            )
        ):
            missing_driver_evidence.append(case["name"])
        if case["expected"] == "FAILED" and not (
            "ERROR tributo_knova." in log
            and "FAILED event published:" in log
            and len(terminals) == 1
            and terminals[0]["event_type"] == "FAILED"
        ):
            silent_expected_failures.append(case["name"])
        for marker in (
            "local-repro-only",
            "local-test-only",
            "local-test-secret",
            "local-sentinel-only",
        ):
            if marker in log or marker in json.dumps(events):
                credential_marker_hits.append(case["name"])
        ray_statuses[status] = ray_statuses.get(status, 0) + 1
        detail = case.get("detail", {})
        if case["passed"] and case["expected"] == "COMPLETED":
            if case["operation"] == "training":
                trained_rows += detail["sample_rows"]["total"]
            else:
                processed_rows += terminal["processed_rows"]
                written_rows += terminal["result_rows"]
                oracle = detail.get("native_comparison", {})
                native_prediction_rows += oracle.get(
                    "checked_rows", oracle.get("sample_rows", 0)
                )
                if oracle.get("source_features_match_returned_values"):
                    source_verified_rows += oracle.get("checked_rows", 0)
                if case["shap"] != "none":
                    shap_rows += oracle.get("checked_rows", 0)
                    max_margin_error = max(
                        max_margin_error, oracle.get("maximum_margin_error", 0)
                    )
        outcome = "PASS" if case["passed"] else "FAIL"
        error = terminal.get("error_code", "")
        counts = (
            str(detail.get("sample_rows", ""))
            if case["operation"] == "training"
            else (
                f"{terminal.get('processed_rows', 0)} / "
                f"{terminal.get('result_rows', 0)}"
            )
        )
        rows.append(
            f"| {case['name']} | {case['kind']} | "
            f"{case.get('query_input_rows', case['rows']):,} "
            f"× 116 | {outcome} | {terminal.get('event_type', '')} {error} | "
            f"{counts} | {case['seconds']:.2f} | {status} |"
        )
    summary = {
        "image_id": launch["image_id"],
        "started_utc": launch["started_utc"],
        "finished_utc": launch["finished_utc"],
        "elapsed_seconds": launch["elapsed_seconds"],
        "planned_cases": len(plan),
        "executed_cases": len(results),
        "passed_cases": len(results) - len(unexpected),
        "unexpected_failures": unexpected,
        "case_names_match": expected_names == actual_names,
        "case_counts_and_unique_names_match": case_counts_match,
        "successful_training_rows": trained_rows,
        "successful_inference_processed_rows": processed_rows,
        "successful_inference_written_rows": written_rows,
        "native_shap_verified_rows": shap_rows,
        "native_prediction_verified_rows": native_prediction_rows,
        "clickhouse_source_feature_verified_rows": source_verified_rows,
        "all_successful_output_rows_source_verified": source_verified_rows
        == written_rows,
        "native_shap_verified_feature_contributions": shap_rows * 116,
        "maximum_native_margin_error": max_margin_error,
        "ray_job_statuses": ray_statuses,
        "runtime_images_pinned": pinned_images,
        "preexisting_containers_unchanged": retained,
        "preexisting_container_count": len(before),
        "owned_containers_remaining": own_remaining,
        "cleanup_returncode": launch["cleanup_returncode"],
        "consumer_invocation": (
            "real Redis Streams + embedded KnovaBrokerPlugin "
            "ConsumerRuntime.start/run_once"
        ),
        "driver_invocation": "native Ray Jobs with image-packaged modules",
        "standalone_consumer_process_exercised_by_this_matrix": False,
        "missing_driver_identity_or_module_evidence": missing_driver_evidence,
        "silent_expected_failures": silent_expected_failures,
        "terminal_count_mismatches": terminal_count_mismatches,
        "ray_status_mismatches": ray_status_mismatches,
        "driver_input_bytes": {
            "observations": len(driver_input_bytes),
            "minimum": min(driver_input_bytes, default=None),
            "maximum": max(driver_input_bytes, default=None),
        },
        "unknown_inference_success_status_cases": unknown_inference_status,
        "unknown_ray_job_id_cases": unknown_ray_job_id,
        "observed_event_phases": sorted(phase_events),
        "fixture_credential_marker_hits_in_job_logs_or_events": credential_marker_hits,
    }
    summary["passed"] = (
        not unexpected
        and expected_names == actual_names
        and case_counts_match
        and retained
        and not own_remaining
        and pinned_images
        and launch["cleanup_returncode"] == 0
        and launch["returncode"] == 0
        and not missing_driver_evidence
        and not silent_expected_failures
        and not terminal_count_mismatches
        and not ray_status_mismatches
        and not credential_marker_hits
        and source_verified_rows == written_rows
    )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "driver-log-audit.json").write_text(
        json.dumps(driver_log_audit, ensure_ascii=False, indent=2) + "\n"
    )
    text = [
        "# Final XGBoost / SHAP matrix",
        "",
        f"Image: `{summary['image_id']}`",
        "",
        f"UTC: {summary['started_utc']} → {summary['finished_utc']}",
        "",
        f"Cases: {summary['passed_cases']} / {summary['planned_cases']}; "
        f"elapsed {summary['elapsed_seconds']:.2f} seconds.",
        "",
        "Training parameters: XGBoost, 3 boosting rounds, max_depth=3, "
        "two Ray workers, 116 numeric Float32 features.",
        "",
        f"Native SHAP verified {shap_rows:,} output rows × 116 features; "
        f"maximum selected-margin error {max_margin_error:.3g}.",
        "",
        (
            f"All {source_verified_rows:,} successful output rows were joined by "
            "entity_id to their ClickHouse source: every one of 116 returned feature "
            "values matched exactly, and native predictions were recalculated from "
            "those source values. Every SHAP output row received the same independent "
            "source-based contribution comparison. Result-filter oracles also read "
            "all input features and verify the exact expected output ID set."
            if source_verified_rows == written_rows
            else "Historical harness boundary: SHAP uses returned feature values; "
            "non-SHAP native checks sample up to seven rows. These checks alone do "
            "not prove source-value preservation. See post-audit.json, when present, "
            "for the subsequent independent ClickHouse source/native comparison."
        ),
        "",
        "Invocation: real Redis task/event streams and the actual ConsumerRuntime "
        "embedded in the harness process; independent native Ray Job drivers. "
        "Standalone deployed Consumer process coverage is supplied separately "
        "by Kind acceptance.",
        "",
        f"Actual encoded Driver inputs: {min(driver_input_bytes, default=0):,}–"
        f"{max(driver_input_bytes, default=0):,} bytes across "
        f"{len(driver_input_bytes)} submissions. Table rows are read from ClickHouse; "
        "300,000 rows are not embedded into the task payload. Payload size depends "
        "on feature metadata and name lengths; the separate 11 transport boundary "
        "checks, not these row counts, cover the earlier 64 KiB limit.",
        "",
        "The workloads container inherits a Consumer CLI healthcheck against the baked "
        "default config; the harness supplies an isolated runtime config in memory. "
        "Its unhealthy marker is recorded in runner-healthcheck-explanation.json; "
        "business outcomes use actual Ray jobs, Redis events and stored results.",
        "",
        (
            "Diagnostic limitation in the v2.6 image: "
            "src/tributo_knova/execution_driver.py:268 reads result.execution.status "
            "for inference, whose status lives on the result itself; the success INFO "
            "line says status=unknown. Actual COMPLETED events and Ray SUCCEEDED "
            "states are the success evidence."
            if unknown_inference_status
            else "No inference success line reported status=unknown. "
            "Actual COMPLETED events and Ray SUCCEEDED states remain the success "
            "evidence; driver-log-audit.json retains per-case log references."
        ),
        f"Driver logs with unknown Ray job ID: {len(unknown_ray_job_id)}.",
        "",
        "Expected FAILED rows verify explicit protocol rejection requirements.",
        f"Driver identity/module evidence missing: {len(missing_driver_evidence)}; "
        f"silent expected failures: {len(silent_expected_failures)}; "
        f"Ray/event terminal mismatches: {len(ray_status_mismatches)}. "
        "driver-log-audit.json records exact log line references and event phases.",
        "",
        "| Case | Type | Query shape | Assertion | Terminal | "
        "Train split or processed / written | Seconds | Ray |",
        "|---|---|---:|---|---|---|---:|---|",
        *rows,
        "",
        f"Preexisting containers unchanged: {retained}; cleanup exit: "
        f"{launch['cleanup_returncode']}; owned remaining: {len(own_remaining)}.",
        "",
        "Per-case requests, terminals, events, output samples, results, "
        "and Ray logs are in cases/. "
        "The launch record pins image bytes and includes the script snapshot digest.",
    ]
    (root / "final-summary.md").write_text("\n".join(text) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = summarize(parser.parse_args().directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
