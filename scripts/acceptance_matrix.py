"""Isolated, evidence-producing KnoVa task/shape acceptance matrix.

--plan is read-only and imports no service clients. --launch owns one unique
Compose project and only removes that project. --run is the container entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TASKS = {
    "binary": "BINARY_CLASSIFICATION",
    "multi": "MULTICLASS_CLASSIFICATION",
    "reg": "REGRESSION",
}
LABELS = {
    "binary": {"原始负类": 0, "原始正类": 1},
    "multi": {"原始甲类": 0, "原始乙类": 1, "原始丙类": 2},
    "reg": {},
}
WIDTH = 116


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    operation: str
    kind: str
    rows: int
    expected: str = "COMPLETED"
    storage: str = "nfs"
    shap: str = "none"
    threshold: float | None = None
    positive_index: int = 1
    algorithm: str = "xgboost"
    filter_mode: str | None = None
    filter_label_index: int = 0
    string_target: bool = False
    empty_filter: bool = False
    expected_error_code: str | None = None
    expected_error_message: str | None = None
    expected_output_rows: int | None = None
    model_training_case: str | None = None


def build_plan() -> list[Case]:
    cases = [
        Case(
            f"train_{kind}_1000",
            "training",
            kind,
            1000,
            storage="s3" if kind == "reg" else "nfs",
        )
        for kind in TASKS
    ]
    cases += [Case(f"infer_{kind}_7", "inference", kind, 7) for kind in TASKS]
    for rows in (24, 300000):
        cases += [
            Case(f"train_{kind}_{rows}", "training", kind, rows) for kind in TASKS
        ]
        if rows == 24:
            cases.append(
                Case(
                    "infer_reg_nas_7",
                    "inference",
                    "reg",
                    7,
                    model_training_case="train_reg_24",
                    shap="exact",
                )
            )
    for rows in (0, 1000, 100000):
        cases += [
            Case(
                f"infer_{kind}_{rows}",
                "inference",
                kind,
                rows,
                expected="FAILED" if rows == 0 else "COMPLETED",
                expected_error_code="EMPTY_INPUT" if rows == 0 else None,
            )
            for kind in TASKS
        ]
    cases += [
        Case(f"shap_{kind}_exact", "inference", kind, 7, shap="exact") for kind in TASKS
    ]
    cases += [
        Case(f"shap_{kind}_1000", "inference", kind, 1000, shap="exact")
        for kind in TASKS
    ]
    cases.append(Case("shap_binary_10000", "inference", "binary", 10000, shap="exact"))
    cases += [
        Case(
            f"infer_{kind}_filtered_empty",
            "inference",
            kind,
            7,
            expected="FAILED",
            empty_filter=True,
            expected_error_code="EMPTY_INPUT",
        )
        for kind in TASKS
    ]
    cases += [
        Case("shap_binary_approx", "inference", "binary", 7, shap="approximate"),
        Case(
            "shap_binary_positive_zero",
            "inference",
            "binary",
            7,
            shap="exact",
            positive_index=0,
            threshold=0.5,
        ),
        Case("threshold_binary_zero", "inference", "binary", 1000, threshold=0.0),
        Case("threshold_binary_one", "inference", "binary", 1000, threshold=1.0),
        Case(
            "infer_binary_all_output_filtered",
            "inference",
            "binary",
            1000,
            threshold=1.0,
            filter_mode="INCLUDE",
            filter_label_index=1,
            expected_output_rows=0,
        ),
        Case(
            "filter_binary_include", "inference", "binary", 1000, filter_mode="INCLUDE"
        ),
        Case("filter_multi_exclude", "inference", "multi", 1000, filter_mode="EXCLUDE"),
        Case(
            "reject_multi_threshold",
            "inference",
            "multi",
            7,
            expected="FAILED",
            threshold=0.5,
            expected_error_code="INVALID_PAYLOAD",
        ),
        Case(
            "reject_reg_threshold",
            "inference",
            "reg",
            7,
            expected="FAILED",
            threshold=0.5,
            expected_error_code="INVALID_PAYLOAD",
        ),
        Case(
            "reject_reg_filter",
            "inference",
            "reg",
            7,
            expected="FAILED",
            filter_mode="INCLUDE",
            expected_error_code="INVALID_PAYLOAD",
        ),
        Case(
            "train_binary_string_labels", "training", "binary", 1000, string_target=True
        ),
        Case(
            "infer_binary_string_labels",
            "inference",
            "binary",
            7,
            shap="exact",
            model_training_case="train_binary_string_labels",
        ),
        Case(
            "train_multi_string_labels", "training", "multi", 1000, string_target=True
        ),
        Case(
            "infer_multi_string_labels",
            "inference",
            "multi",
            7,
            shap="exact",
            model_training_case="train_multi_string_labels",
        ),
        Case(
            "train_binary_positive_zero",
            "training",
            "binary",
            1000,
            positive_index=0,
        ),
        Case(
            "infer_trained_binary_positive_zero",
            "inference",
            "binary",
            7,
            positive_index=0,
            shap="exact",
            threshold=0.5,
            model_training_case="train_binary_positive_zero",
        ),
    ]
    for rows in (0, 1):
        cases += [
            Case(
                f"reject_train_{kind}_{rows}",
                "training",
                kind,
                rows,
                expected="FAILED",
                expected_error_code="EXECUTION_FAILED",
                expected_error_message=(
                    "Training input is empty after filtering"
                    if rows == 0
                    else "Training input is too small for the configured "
                    "train/validation/test split"
                ),
            )
            for kind in TASKS
        ]
    assert len({case.name for case in cases}) == len(cases)
    return cases


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def launch(image: str, output: Path, selected: list[str]) -> int:
    """Use the existing disposable fixture definitions with a unique project."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "launch.json").exists():
        raise RuntimeError("Refusing to overwrite an existing matrix launch")
    (output / "shared").mkdir(exist_ok=True)
    script = Path(__file__).read_bytes()
    (output / "acceptance_matrix.snapshot.py").write_bytes(script)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    project = "knova-matrix-" + uuid.uuid4().hex[:8]
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(Path(__file__).with_name("acceptance.compose.yaml")),
    ]
    env = dict(os.environ, ACCEPT_IMAGE=image_id, ACCEPT_ARTIFACTS=str(output))
    fmt = "{{.ID}} {{.Names}} {{.State}}"
    (output / "containers-before.txt").write_text(
        subprocess.check_output(["docker", "ps", "-a", "--format", fmt], text=True)
    )
    record = {
        "image": image,
        "image_id": image_id,
        "project": project,
        "started_utc": datetime.now(UTC).isoformat(),
        "selected": selected,
        "script_sha256": hashlib.sha256(script).hexdigest(),
    }
    write_json(output / "launch.json", record)
    write_json(
        output / "plan.json",
        [
            dataclasses.asdict(case)
            for case in build_plan()
            if not selected or case.name in selected
        ],
    )
    status = 1
    start = time.monotonic()
    try:
        with (output / "startup.log").open("w") as log:
            subprocess.run(
                compose + ["up", "-d"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        command = compose + [
            "run",
            "--rm",
            "--no-deps",
            "-e",
            "ACCEPT_MATRIX_ISOLATED=1",
            "-e",
            f"ACCEPT_MATRIX_IMAGE_ID={image_id}",
            "--entrypoint",
            "python",
            "workloads",
            "/scripts/acceptance_matrix.py",
            "--run",
        ]
        for name in selected:
            command += ["--case", name]
        record["workload_command"] = command
        write_json(output / "launch.json", record)
        with (output / "workloads.log").open("w") as log:
            status = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
    finally:
        with (output / "services.log").open("w") as log:
            subprocess.run(
                compose + ["logs", "--no-color"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        with (output / "cleanup.log").open("w") as log:
            cleanup = subprocess.run(
                compose + ["down", "--volumes"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record.update(
            returncode=status,
            cleanup_returncode=cleanup.returncode,
            elapsed_seconds=round(time.monotonic() - start, 2),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        write_json(output / "launch.json", record)
        (output / "containers-after.txt").write_text(
            subprocess.check_output(["docker", "ps", "-a", "--format", fmt], text=True)
        )
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return status


class MatrixRunner:
    def __init__(self, output: Path) -> None:
        if os.environ.get("ACCEPT_MATRIX_ISOLATED") != "1":
            raise RuntimeError(
                "Use --launch; direct runs require an isolated acceptance network"
            )
        import clickhouse_connect
        import redis
        from ray.job_submission import JobSubmissionClient

        from tributo_knova.broker import KnovaBrokerPlugin

        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]
        self.results: list[dict[str, Any]] = []
        self.models: dict[str, dict[str, Any]] = {}
        self.boosters: dict[str, Any] = {}
        self.tables: dict[tuple[str, int, bool, int], str] = {}
        self.case_timeout = 600
        for attempt in range(120):
            try:
                self.ch = clickhouse_connect.get_client(
                    host="clickhouse",
                    username="knova",
                    password="local-test-only",
                    connect_timeout=3,
                )
                self.ch.command("SELECT 1")
                break
            except Exception:
                if attempt == 119:
                    raise
                time.sleep(1)
        self.redis = redis.Redis(
            host="repro-redis", password="local-repro-only", decode_responses=True
        )
        self.jobs = JobSubmissionClient("http://repro-ray:8265")
        self.config = json.loads(Path("/etc/tributo-knova/config.json").read_text())
        self.config["transport"] = {
            "url": "redis://repro-redis:6379/0",
            "password_ref": "env:KNOVA_REDIS_PASSWORD",
            "block_ms": 100,
            "claim_idle_ms": 500,
            "max_driver_input_bytes": 262144,
        }
        self.config["execution"]["ray_dashboard_url"] = "http://repro-ray:8265"
        for name, channel in self.config["channels"].items():
            for key in (
                "task_stream_key",
                "event_stream_prefix",
                "cancel_key_prefix",
                "consumer_group",
            ):
                channel[key] = f"matrix:{self.run_id}:{name}:{key}"
            channel["group_start_id"] = "0-0"
        self.runtime = KnovaBrokerPlugin().create_runtime(self.config)
        self.runtime.start()
        self.args = SimpleNamespace(
            storage_mode="nfs",
            s3_bucket="knova-matrix",
            clickhouse_host="clickhouse",
            clickhouse_port=8123,
            clickhouse_user="knova",
        )
        self.ch.command("CREATE DATABASE IF NOT EXISTS knova")
        self.result_table = f"matrix_{self.run_id}_results"
        self.ch.command(
            f"CREATE TABLE knova.{self.result_table} (entity_id String, "
            "pred_label Nullable(String), pred_probability Nullable(Float64), "
            "pred_probabilities Nullable(String), pred_value Nullable(Float64), "
            "pred_extra String, execution_id String, model_id String, "
            "version_id String, "
            "tenant_id String, inferred_at DateTime64(6)) ENGINE=MergeTree "
            "ORDER BY (execution_id,entity_id)"
        )
        import boto3

        self.s3 = boto3.client(
            "s3",
            endpoint_url="http://rustfs:9000",
            aws_access_key_id="localtestkey",
            aws_secret_access_key="local-test-secret",
            region_name="us-east-1",
        )
        if self.args.s3_bucket not in [
            item["Name"] for item in self.s3.list_buckets()["Buckets"]
        ]:
            self.s3.create_bucket(Bucket=self.args.s3_bucket)

    def table(self, case: Case) -> str:
        key = (case.kind, case.rows, case.string_target, case.positive_index)
        if key in self.tables:
            return self.tables[key]
        name = (
            f"matrix_{self.run_id}_{case.kind}_{case.rows}_"
            f"{int(case.string_target)}_{case.positive_index}"
        )
        columns = ", ".join(f"f{i} Float32" for i in range(1, WIDTH + 1))
        values = ", ".join(
            f"toFloat32((number*{i * 2 + 1})%{31 + i})/{31 + i}"
            for i in range(1, WIDTH + 1)
        )
        label_type = "Float64" if case.kind == "reg" else "UInt8"
        label = (
            "toFloat64(number%97)/11.0-3.0"
            if case.kind == "reg"
            else f"toUInt8(number%{3 if case.kind == 'multi' else 2})"
        )
        if case.name == "train_binary_positive_zero":
            # Constant features force a constant class-0 prediction. An 80/20
            # target makes a reversed confusion matrix observable, not symmetric.
            values = ", ".join("toFloat32(0)" for _ in range(WIDTH))
            label = "toUInt8(number%5=0)"
        if case.string_target:
            label_type = "String"
            label = (
                "multiIf(number%3=0,'原始甲类',number%3=1,'原始乙类','原始丙类')"
                if case.kind == "multi"
                else "if(number%2=0,'原始负类','原始正类')"
            )
        self.ch.command(
            f"CREATE TABLE knova.{name} (entity_id UInt64, {columns}, "
            f"label {label_type}, stat_month String) "
            "ENGINE=MergeTree ORDER BY entity_id"
        )
        if case.rows:
            self.ch.command(
                f"INSERT INTO knova.{name} SELECT number, {values}, {label}, "
                f"'202601' FROM numbers({case.rows})"
            )
        shape = self.ch.query(
            f"SELECT count(),uniqExact(entity_id) FROM knova.{name}"
        ).first_row
        assert tuple(shape) == (case.rows, case.rows), shape
        self.tables[key] = name
        return name

    def submit(
        self, request: dict[str, Any], case: Case, case_dir: Path
    ) -> dict[str, Any]:
        channel = self.config["channels"][
            "training" if case.operation == "training" else "batch_inference"
        ]
        identity = request[channel["outer_identity_field"]]
        stream = channel["event_stream_prefix"] + ":" + identity
        # Persist request evidence with password values removed.
        redacted = copy.deepcopy(request)

        def redact(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {
                        "password",
                        "secret",
                        "secret_key",
                        "access_key",
                    }:
                        value[key] = "<redacted>"
                    else:
                        redact(item)
            elif isinstance(value, list):
                for item in value:
                    redact(item)

        redact(redacted)
        write_json(case_dir / "request.redacted.json", redacted)
        self.redis.xadd(
            channel["task_stream_key"],
            {channel["outer_identity_field"]: identity, "payload": json.dumps(request)},
        )
        start = time.monotonic()
        assert self.runtime.run_once(timeout_ms=1000), (
            "Consumer did not receive matrix task"
        )
        active = self.runtime.active_submissions.get(identity)
        try:
            while time.monotonic() - start < self.case_timeout:
                events = [
                    json.loads(item["payload"]) for _, item in self.redis.xrange(stream)
                ]
                terminals = [
                    event
                    for event in events
                    if event["event_type"] in {"COMPLETED", "FAILED", "CANCELLED"}
                ]
                if terminals:
                    assert len(terminals) == 1, "Duplicate terminal events"
                    write_json(case_dir / "terminal.json", terminals[0])
                    assert terminals[0]["event_type"] == case.expected, terminals[
                        0
                    ].get("error_message", terminals[0]["event_type"])
                    if case.expected_error_code is not None:
                        assert (
                            terminals[0].get("error_code") == case.expected_error_code
                        )
                    if case.expected_error_message is not None:
                        assert (
                            terminals[0].get("error_message")
                            == case.expected_error_message
                        ), "Training rejection reason does not match insufficient input"
                    return terminals[0]
                time.sleep(0.5)
            if active:
                self.jobs.stop_job(active.submission.submission_id)
            raise TimeoutError(
                "Matrix case exceeded its deadline; only its own Ray job was stopped"
            )
        finally:
            if active:
                job = active.submission.submission_id
                info = self.jobs.get_job_info(job)
                settled = time.monotonic() + 30
                while str(info.status) not in {"SUCCEEDED", "FAILED", "STOPPED"}:
                    if time.monotonic() >= settled:
                        break
                    time.sleep(0.5)
                    info = self.jobs.get_job_info(job)
                write_json(
                    case_dir / "ray-status.json",
                    {
                        "submission_id": job,
                        "status": str(info.status),
                        "error_type": info.error_type,
                    },
                )
                (case_dir / "ray.log").write_text(self.jobs.get_job_logs(job))
            write_json(
                case_dir / "events.json",
                [json.loads(item["payload"]) for _, item in self.redis.xrange(stream)],
            )
            if active:
                expected_ray = "SUCCEEDED" if case.expected == "COMPLETED" else "FAILED"
                assert str(info.status) == expected_ray, (
                    "Ray terminal status disagrees with the expected business terminal"
                )

    def train(self, case: Case, directory: Path) -> dict[str, Any]:
        from local_smoke import _training_request

        args = copy.copy(self.args)
        args.storage_mode = case.storage
        identity = f"matrix-{self.run_id}-{case.name}"
        request = _training_request(
            args,
            job_id=identity,
            table=self.table(case),
            storage_root="/accept-shared",
            storage_prefix=identity,
            password="local-test-only",
        )
        request["features"] = [
            {"origin": {"column_name": f"f{i}"}} for i in range(1, WIDTH + 1)
        ]
        request["algorithm"].update(
            algorithm_key=case.algorithm,
            category="REGRESSION" if case.kind == "reg" else "CLASSIFICATION",
            hyper_params={"n_estimators": 3, "max_depth": 3},
        )
        request["target"].update(
            task_type=TASKS[case.kind], label_mapping=LABELS[case.kind]
        )
        if case.kind == "binary":
            request["target"]["positive_label_value"] = next(
                label
                for label, index in LABELS[case.kind].items()
                if index == case.positive_index
            )
            if case.positive_index == 0:
                request["evaluation"] = {
                    "artifacts": {"roc_curve": True, "threshold_analysis": True}
                }
        request["data_split"] = {
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "strategy": "RANDOM",
            "stratify": False,
            "random_seed": 42,
            "cross_validation": {"enabled": False},
        }
        terminal = self.submit(request, case, directory)
        if case.expected == "FAILED":
            return {
                "terminal_status": "FAILED",
                "error_code": terminal.get("error_code"),
                "error_message": terminal.get("error_message"),
            }
        rows = terminal["result_summary"]["sample_rows"]
        assert (
            rows["total"] == case.rows
            and sum(rows[key] for key in ("train", "validation", "test")) == case.rows
        ), rows
        assert rows["train"] >= 2 and rows["validation"] >= 2, rows
        evaluation = terminal["training_result"]["evaluation"]
        assert evaluation["eval_type"] == TASKS[case.kind]
        assert evaluation["sample_rows"] == rows["test"]
        metrics = {item["metric_name"]: item["value"] for item in evaluation["metrics"]}
        required = {"mae", "rmse", "r2"} if case.kind == "reg" else {"accuracy", "f1"}
        assert required <= metrics.keys()
        assert all(math.isfinite(value) for value in metrics.values())
        confusion = evaluation["details"].get("confusion_matrix")
        if case.kind == "binary":
            assert (
                sum(confusion[key] for key in ("tp", "fp", "tn", "fn")) == rows["test"]
            )
            if case.positive_index == 0:
                assert confusion["fn"] == confusion["tn"] == 0
                assert confusion["tp"] > confusion["fp"] > 0
                assert evaluation["details"]["roc_curve"] is not None
                assert (
                    len(evaluation["details"]["threshold_analysis"]["thresholds"]) == 19
                )
        elif case.kind == "multi":
            assert confusion["labels"] == list(LABELS[case.kind])
            assert len(confusion["matrix"]) == 3
            assert all(len(row) == 3 for row in confusion["matrix"])
            assert sum(map(sum, confusion["matrix"])) == rows["test"]
        manifest = terminal["artifact_manifest"]
        artifacts = [
            item
            for spec in manifest["model_artifacts"].values()
            for alt in spec.get("alternatives", [])
            for item in alt.get("files", [])
        ]
        assert artifacts, "Missing model artifacts"
        for artifact in artifacts:
            storage = manifest["storage"]
            key = storage["prefix"].rstrip("/") + "/" + artifact["path"].lstrip("/")
            if storage["type"] == "s3":
                body = self.s3.get_object(Bucket=storage["bucket"], Key=key)["Body"]
                try:
                    content = body.read()
                finally:
                    body.close()
            else:
                content = (Path(storage["bucket"]) / key).read_bytes()
            assert len(content) == artifact["size"], artifact["path"]
            assert (
                "sha256:" + hashlib.sha256(content).hexdigest() == artifact["hash"]
            ), artifact["path"]
            if artifact["path"].endswith(".ubj"):
                import xgboost

                booster = xgboost.Booster()
                booster.load_model(bytearray(content))
                booster.set_param({"nthread": 1})
                self.boosters[case.name] = booster
        self.models[case.name] = manifest
        return {
            "terminal_status": "COMPLETED",
            "sample_rows": rows,
            "evaluation": evaluation,
            "storage": manifest["storage"]["type"],
            "verified_artifact_files": len(artifacts),
            "artifact_manifest": manifest,
        }

    def infer(self, case: Case, directory: Path) -> dict[str, Any]:
        from local_smoke import _inference_request

        model_key = case.model_training_case or f"train_{case.kind}_1000"
        if model_key not in self.models:
            raise RuntimeError(
                "Required reference model training did not pass; inference not executed"
            )
        identity = f"matrix-{self.run_id}-{case.name}"
        request = _inference_request(
            self.args,
            execution_id=identity,
            input_table=self.table(case),
            output_table=self.result_table,
            artifact_manifest=self.models[model_key],
            password="local-test-only",
            shap_mode=case.shap,
        )
        request["model"].update(
            task_type=TASKS[case.kind], label_mapping=LABELS[case.kind]
        )
        request["model"].pop("positive_label_value", None)
        if case.kind == "binary":
            request["model"]["positive_label_value"] = next(
                label
                for label, index in LABELS[case.kind].items()
                if index == case.positive_index
            )
        request["input"]["features"] = [
            {
                "origin": {"table_alias": "t0", "column_name": f"f{i}"},
                "result_column": f"f{i}",
            }
            for i in range(1, WIDTH + 1)
        ]
        request["execution"].update(batch_size=4096, concurrency=2)
        if case.empty_filter:
            request["input"]["query"]["params"]["month"] = "empty-matrix-month"
        if case.threshold is not None:
            request.setdefault("extensions", {})["decision_threshold"] = case.threshold
        if case.filter_mode:
            request["output"]["result_filter"] = {
                "mode": case.filter_mode,
                "labels": [
                    next(
                        (
                            label
                            for label, index in LABELS[case.kind].items()
                            if index == case.filter_label_index
                        ),
                        "invalid-regression-label",
                    )
                ],
            }
        terminal = self.submit(request, case, directory)
        if case.expected == "FAILED":
            count = self.ch.query(
                f"SELECT count() FROM knova.{self.result_table} "
                "WHERE execution_id={id:String}",
                parameters={"id": identity},
            ).first_row[0]
            assert count == 0, "Rejected inference wrote output rows"
            return {
                "terminal_status": "FAILED",
                "error_code": terminal.get("error_code"),
                "error_message": terminal.get("error_message"),
                "written_rows": count,
            }
        names = [f"f{i}" for i in range(1, WIDTH + 1)]
        source_table = request["input"]["tables"][0]["table_name"]
        query = (
            "SELECT r.entity_id,r.pred_label,r.pred_probability,r.pred_probabilities,"
            "r.pred_value,r.pred_extra,"
            + ",".join(f"s.{name}" for name in names)
            + f" FROM knova.{self.result_table} r "
            + f"INNER JOIN knova.{source_table} s ON toUInt64(r.entity_id)=s.entity_id "
            + "WHERE r.execution_id={id:String} ORDER BY toUInt64(r.entity_id)"
        )
        seen: set[int] = set()
        sampled = []
        oracle_buffer: list[dict[str, Any]] = []
        oracle_results: list[dict[str, Any]] = []
        positives = 0
        reverse = {index: label for label, index in LABELS[case.kind].items()}
        with self.ch.query_rows_stream(query, parameters={"id": identity}) as stream:
            for row in stream:
                entity, label, probability, vector_json, value, extra_json = row[:6]
                number = int(entity)
                assert 0 <= number < case.rows and number not in seen, (
                    "Invalid or duplicate entity ID"
                )
                seen.add(number)
                extra = json.loads(extra_json)
                assert len(extra["inference_feature_values"]) == WIDTH
                if case.kind == "binary":
                    assert (
                        label in LABELS[case.kind]
                        and value is None
                        and vector_json is None
                    )
                    assert (
                        probability is not None
                        and math.isfinite(probability)
                        and 0 <= probability <= 1
                    )
                    threshold = 0.5 if case.threshold is None else case.threshold
                    expected_index = (
                        case.positive_index
                        if probability >= threshold
                        else 1 - case.positive_index
                    )
                    assert label == reverse[expected_index], (
                        "Threshold/positive-class mapping mismatch"
                    )
                    positives += int(label == reverse[case.positive_index])
                elif case.kind == "multi":
                    vector = json.loads(vector_json)
                    assert (
                        label in LABELS[case.kind]
                        and probability is None
                        and value is None
                    )
                    assert len(vector) == 3 and all(
                        math.isfinite(x) and 0 <= x <= 1 for x in vector
                    )
                    assert abs(sum(vector) - 1) < 1e-5, (
                        "Multiclass probabilities are not normalized"
                    )
                    assert (
                        label == reverse[max(range(3), key=lambda index: vector[index])]
                    )
                else:
                    assert label is None and probability is None and vector_json is None
                    assert value is not None and math.isfinite(value)
                if case.filter_mode:
                    wanted = reverse[case.filter_label_index]
                    assert (label == wanted) == (case.filter_mode == "INCLUDE")
                explanation = extra.get("explanation")
                if case.shap != "none":
                    assert explanation and explanation["method"] == "TREE_SHAP"
                    assert explanation["approximate"] == (case.shap == "approximate")
                    contributions = explanation["feature_contributions"]
                    assert len(contributions) == WIDTH
                    margin = explanation["base_value"] + sum(
                        item["shap_value"] for item in contributions
                    )
                    assert math.isfinite(margin)
                    if case.kind == "binary":
                        assert (
                            explanation["explained_class_index"] == case.positive_index
                        )
                        assert (
                            explanation["explained_class_label"]
                            == reverse[case.positive_index]
                        )
                        reconstructed = 1 / (1 + math.exp(-margin))
                        assert abs(reconstructed - probability) < 1e-5, (
                            "Binary SHAP margin does not reconstruct "
                            "positive-class probability"
                        )
                    elif case.kind == "multi":
                        assert explanation["explained_class_label"] == label
                        assert (
                            explanation["explained_class_index"]
                            == LABELS[case.kind][label]
                        )
                    else:
                        assert (
                            "explained_class_index" not in explanation
                            and "explained_class_label" not in explanation
                        )
                        assert abs(margin - value) < 1e-4, (
                            "Regression SHAP sum does not reconstruct prediction"
                        )
                else:
                    assert explanation is None
                record = dict(
                    zip(
                        (
                            "entity_id",
                            "pred_label",
                            "pred_probability",
                            "pred_probabilities",
                            "pred_value",
                            "pred_extra",
                        ),
                        row[:6],
                        strict=True,
                    )
                )
                record["source_features"] = list(row[6:])
                if len(sampled) < 7:
                    sampled.append(record)
                oracle_buffer.append(record)
                if len(oracle_buffer) >= 256:
                    oracle_results.append(self.compare_native(case, oracle_buffer))
                    oracle_buffer.clear()
        if oracle_buffer:
            oracle_results.append(self.compare_native(case, oracle_buffer))
        assert (
            terminal["processed_rows"] == case.rows
            and terminal["total_rows"] == case.rows
        ), terminal
        assert terminal["result_rows"] == len(seen)
        if case.expected_output_rows is not None:
            assert len(seen) == case.expected_output_rows
        if not case.filter_mode:
            assert seen == set(range(case.rows)), "Missing inference rows"
        else:
            import numpy as np
            import xgboost

            names = [f"f{i}" for i in range(1, WIDTH + 1)]
            table = self.tables[
                (case.kind, case.rows, case.string_target, case.positive_index)
            ]
            source = self.ch.query(
                f"SELECT entity_id,{','.join(names)} FROM knova.{table} "
                "ORDER BY entity_id"
            ).result_rows
            matrix = xgboost.DMatrix(
                np.asarray([row[1:] for row in source], dtype=np.float32),
                feature_names=names,
            )
            predictions = self.boosters[model_key].predict(matrix)
            if case.kind == "binary":
                probability = (
                    predictions if case.positive_index == 1 else 1 - predictions
                )
                threshold = 0.5 if case.threshold is None else case.threshold
                indices = np.where(
                    probability >= threshold,
                    case.positive_index,
                    1 - case.positive_index,
                )
            else:
                indices = predictions.argmax(axis=1)
            selected_index = case.filter_label_index
            expected_ids = {
                int(row[0])
                for row, index in zip(source, indices, strict=True)
                if (index == selected_index) == (case.filter_mode == "INCLUDE")
            }
            assert seen == expected_ids, "Result filtering lost or added entity rows"
        write_json(directory / "output-sample.json", sampled)
        native_comparison = {
            "checked_rows": sum(item["sample_rows"] for item in oracle_results),
            "scope": "all_output_rows",
            "source": "entity_id JOIN to actual ClickHouse source table",
            "source_features_match_returned_values": True,
            "prediction_matches_native": True,
            "feature_order_verified": True,
            "shap_matches_native": case.shap != "none",
            "maximum_margin_error": max(
                (item["maximum_margin_error"] for item in oracle_results), default=0
            ),
        }
        assert native_comparison["checked_rows"] == len(seen)
        return {
            "terminal_status": "COMPLETED",
            "input_rows": case.rows,
            "written_rows": len(seen),
            "unique_entity_rows": len(seen),
            "positive_predictions": positives if case.kind == "binary" else None,
            "task_type": TASKS[case.kind],
            "output_types_checked": True,
            "shap": case.shap,
            "native_comparison": native_comparison,
        }

    def compare_native(
        self, case: Case, sampled: list[dict[str, Any]]
    ) -> dict[str, Any]:
        import numpy as np
        import xgboost

        if not sampled:
            return {"sample_rows": 0}
        names = [f"f{i}" for i in range(1, WIDTH + 1)]
        extras = [json.loads(row["pred_extra"]) for row in sampled]
        for extra in extras:
            assert [
                item["feature_name"] for item in extra["inference_feature_values"]
            ] == names
        returned_values = np.asarray(
            [
                [item["value"] for item in extra["inference_feature_values"]]
                for extra in extras
            ],
            dtype=np.float64,
        )
        values = np.asarray(
            [row["source_features"] for row in sampled], dtype=np.float32
        )
        assert np.array_equal(returned_values, values.astype(np.float64)), (
            "Returned feature values differ from entity-matched ClickHouse source"
        )
        matrix = xgboost.DMatrix(values, feature_names=names)
        booster = self.boosters[case.model_training_case or f"train_{case.kind}_1000"]
        predicted = booster.predict(matrix, strict_shape=True)
        margins = booster.predict(matrix, output_margin=True, strict_shape=True)
        contributions = (
            booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=case.shap == "approximate",
                strict_shape=True,
            )
            if case.shap != "none"
            else None
        )
        largest_margin_error = 0.0
        for index, (row, extra) in enumerate(zip(sampled, extras, strict=True)):
            if case.kind == "binary":
                native_value = float(predicted[index, 0])
                if case.positive_index == 0:
                    native_value = 1 - native_value
                assert abs(row["pred_probability"] - native_value) < 1e-5
            elif case.kind == "multi":
                assert np.allclose(
                    json.loads(row["pred_probabilities"]), predicted[index], atol=1e-5
                )
            else:
                assert abs(row["pred_value"] - float(predicted[index, 0])) < 1e-4
            if contributions is None:
                continue
            explanation = extra["explanation"]
            assert [
                item["feature_name"] for item in explanation["feature_contributions"]
            ] == names
            group = explanation["explained_class_index"] if case.kind == "multi" else 0
            sign = -1 if case.kind == "binary" and case.positive_index == 0 else 1
            actual = np.asarray(
                [
                    *[
                        item["shap_value"]
                        for item in explanation["feature_contributions"]
                    ],
                    explanation["base_value"],
                ]
            )
            assert np.allclose(actual, sign * contributions[index, group], atol=1e-5)
            error = abs(float(actual.sum()) - sign * float(margins[index, group]))
            assert error < 1e-4, (
                "SHAP does not reconstruct native selected-class margin"
            )
            largest_margin_error = max(largest_margin_error, error)
        return {
            "sample_rows": len(sampled),
            "feature_order_verified": True,
            "prediction_matches_native": True,
            "shap_matches_native": contributions is not None,
            "maximum_margin_error": largest_margin_error,
        }

    def run(self, cases: list[Case]) -> int:
        try:
            for case in cases:
                directory = self.output / "cases" / case.name
                directory.mkdir(parents=True, exist_ok=True)
                record = {
                    **dataclasses.asdict(case),
                    "task_type": TASKS[case.kind],
                    "source_table_rows": case.rows,
                    "query_input_rows": 0 if case.empty_filter else case.rows,
                    "input_features": WIDTH,
                    "image_id": os.environ.get("ACCEPT_MATRIX_IMAGE_ID"),
                    "started_utc": datetime.now(UTC).isoformat(),
                }
                start = time.monotonic()
                try:
                    record["detail"] = (
                        self.train(case, directory)
                        if case.operation == "training"
                        else self.infer(case, directory)
                    )
                    record["passed"] = True
                except Exception as error:
                    record.update(
                        passed=False, error_type=type(error).__name__, error=str(error)
                    )
                    (directory / "assertion.log").write_text(traceback.format_exc())
                record.update(
                    seconds=round(time.monotonic() - start, 2),
                    finished_utc=datetime.now(UTC).isoformat(),
                )
                self.results.append(record)
                write_json(directory / "result.json", record)
                write_json(self.output / "matrix-results.json", self.results)
                print(
                    f"{'PASS' if record['passed'] else 'FAIL'} {case.name} "
                    f"{record['seconds']}s",
                    flush=True,
                )
        finally:
            self.runtime.close()
            self.ch.close()
            self.redis.close()
        return int(any(not item["passed"] for item in self.results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--image", default="knova-tributo-devbox:v2.5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "dist/extended-acceptance-20260925/workloads",
    )
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    plan = build_plan()
    if args.case:
        known = {case.name for case in plan}
        if set(args.case) - known:
            parser.error("Unknown case name")
        selected = set(args.case)
        # Inference needs a model; add only the required canonical training case.
        selected.update(
            case.model_training_case or f"train_{case.kind}_1000"
            for case in plan
            if case.name in selected and case.operation == "inference"
        )
        plan = [case for case in plan if case.name in selected]
    if args.plan:
        print(
            json.dumps(
                [dataclasses.asdict(case) for case in plan],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.launch:
        return launch(
            args.image,
            args.output.resolve(),
            [case.name for case in plan] if args.case else [],
        )
    return MatrixRunner(Path("/artifacts")).run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
