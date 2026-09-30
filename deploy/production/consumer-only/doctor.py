#!/usr/bin/env python3
"""Run in a non-consuming Pod; use Ray Jobs and native node-affinity tasks."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

PREFIX = "KNOVA_DOCTOR "
HINTS = {
    "E_CONFIG": (
        "检查 config.json 和 probe.j"
        "son；凭据只能使用 env:/mount: 引"
        "用，禁止 PATH/PYTHONPATH 覆盖。"
    ),
    "E_IMPORT": (
        "核对所列模块的 Consumer/Ray 镜像依赖、原生动态库、算法用途和 Python 版本。"
    ),
    "E_SOURCE": (
        "核对实际导入路径与 py_modules；更新分发源码，避免旧代码覆盖新 site-packages。"
    ),
    "E_SECRET": (
        "在报错所在 Consumer 或 Ray Pod"
        " 注入所列变量；Consumer Secret "
        "不会自动传给 Ray。"
    ),
    "E_SENTINEL": (
        "核对 Sentinel 地址/26379 网络、sentinel_password_ref 和 master 名称。"
    ),
    "E_REDIS": (
        "核对 Sentinel 发现的数据节点网络、Re"
        "dis 数据密码、DB 与 ACL；临时 pro"
        "be 前缀与业务 stream 的 ACL 需分"
        "别核对。"
    ),
    "E_CK": (
        "核对 ClickHouse HTTP 地址/端口/TLS、用户与凭据引用、库表及 SELECT 权限。"
    ),
    "E_S3": (
        "核对 bucket、endpoint/TLS、凭"
        "据、Put/Get/Delete 权限；检查临时"
        " probe 对象是否需清理。"
    ),
    "E_RAY": (
        "核对 Dashboard/Jobs API、Py"
        "thon/Ray 版本、runtime_env "
        "安装、Head/Worker 网络和资源。"
    ),
    "E_BUSY": ("暂停上游，等待列出的 Ray 作业结束；不要并发启动新旧 Consumer。"),
    "E_TOPOLOGY": ("检查 Ray 存活节点数、CPU 总量及训练配置所需资源。"),
}


class Fault(Exception):
    def __init__(self, code, stage, **details):
        super().__init__(code)
        self.code, self.stage, self.details = code, stage, details


def emit(status, stage, **details):
    print(
        PREFIX
        + json.dumps({"status": status, "stage": stage, **details}, ensure_ascii=False),
        flush=True,
    )


def failure(exc, stage="unexpected", code="E_CONFIG"):
    if isinstance(exc, Fault):
        code, stage, details = exc.code, exc.stage, exc.details
    else:
        details = {"exception_type": type(exc).__name__}
    return {
        "status": "FAIL",
        "stage": stage,
        "code": code,
        "hint": HINTS[code],
        **details,
    }


def secret(ref, stage):
    if not ref:
        return None
    try:
        if ref.startswith("env:"):
            value = os.environ.get(ref[4:])
        elif ref.startswith("mount:"):
            value = Path(ref[6:]).read_text().rstrip("\r\n")
        else:
            raise ValueError()
        if not value:
            raise ValueError()
        return value
    except Exception:
        raise Fault("E_SECRET", stage, reference=ref) from None


def validate_probe(probe):
    allowed = {
        "min_alive_nodes",
        "min_total_cpus",
        "require_idle",
        "algorithms",
        "inference_formats",
        "required_env",
        "s3",
        "clickhouse",
        "redis_stream_permissions",
    }
    for flag in ("require_idle", "redis_stream_permissions"):
        if flag in probe and not isinstance(probe[flag], bool):
            raise Fault("E_CONFIG", "probe." + flag)
    for collection in ("algorithms", "inference_formats", "required_env"):
        if collection in probe and (
            not isinstance(probe[collection], list)
            or not all(isinstance(item, str) for item in probe[collection])
        ):
            raise Fault("E_CONFIG", "probe." + collection)
    for name in probe.get("required_env", []):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise Fault("E_CONFIG", "probe.required_env")
    if set(probe) - allowed:
        raise Fault(
            "E_CONFIG", "probe.schema", unknown_fields=sorted(set(probe) - allowed)
        )
    for group, keys in (
        (
            "s3",
            {
                "bucket",
                "endpoint_url",
                "access_key_ref",
                "secret_key_ref",
                "session_token_ref",
                "region_name",
            },
        ),
        (
            "clickhouse",
            {
                "host",
                "port",
                "database",
                "username",
                "password_ref",
                "secure",
                "input_table",
                "output_table",
            },
        ),
    ):
        entry = probe.get(group)
        if entry is not None:
            if not isinstance(entry, dict) or set(entry) - keys:
                raise Fault("E_CONFIG", "probe." + group)
            for key, value in entry.items():
                if key.endswith("_ref") and (
                    not isinstance(value, str)
                    or not value.startswith(("env:", "mount:"))
                ):
                    raise Fault("E_CONFIG", "probe." + group, field=key)
            endpoint = entry.get("endpoint_url", "")
            if endpoint and (urlsplit(endpoint).username or urlsplit(endpoint).query):
                raise Fault("E_CONFIG", "probe." + group, field="endpoint_url")
    if set(probe.get("algorithms", ["xgboost"])) - {"xgboost"}:
        raise Fault("E_CONFIG", "probe.algorithms")
    if set(probe.get("inference_formats", ["onnx"])) - {"onnx", "native"}:
        raise Fault("E_CONFIG", "probe.inference_formats")
    if (
        int(probe.get("min_alive_nodes", 1)) < 1
        or float(probe.get("min_total_cpus", 0)) < 0
    ):
        raise Fault("E_CONFIG", "probe.topology")


def inventory(probe):
    modules = [
        "tributo",
        "tributo_knova",
        "tributo_broker_redis",
        "tributo_algorithms_boosting",
        "ray_clickhouse",
    ]
    result = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "packages": {},
        "sources": {},
    }
    dependencies = {
        "ray": "ray",
        "redis": "redis",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "pyarrow": "pyarrow",
        "clickhouse-connect": "clickhouse_connect",
    }
    if probe.get("s3"):
        dependencies["boto3"] = "boto3"
    for algorithm in probe.get("algorithms", ["xgboost"]):
        dependencies[algorithm] = algorithm
    if "onnx" in probe.get("inference_formats", ["onnx"]):
        dependencies.update(
            {
                "onnx": "onnx",
                "onnxruntime": "onnxruntime",
                "onnxmltools": "onnxmltools",
                "skl2onnx": "skl2onnx",
            }
        )
    for distribution, module in dependencies.items():
        try:
            importlib.import_module(module)
            result["packages"][distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            raise Fault(
                "E_IMPORT", "imports", module=module, exception_type=type(exc).__name__
            ) from None
    for module in modules:
        try:
            imported = importlib.import_module(module)
            path = Path(next(iter(imported.__path__)))
            files = sorted(path.rglob("*.py"))
            if not files:
                raise FileNotFoundError("package has no Python source")
            digest = hashlib.sha256()
            for file in files:
                digest.update(str(file.relative_to(path)).encode())
                digest.update(b"\0")
                digest.update(file.read_bytes())
            result["sources"][module] = {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "files": len(files),
            }
        except Exception as exc:
            raise Fault(
                "E_IMPORT", "imports", module=module, exception_type=type(exc).__name__
            ) from None
    for module in (
        "tributo_knova.execution_driver",
        "tributo_algorithms_boosting.native",
    ):
        try:
            result.setdefault("entrypoints", {})[module] = importlib.import_module(
                module
            ).__file__
        except Exception as exc:
            raise Fault(
                "E_IMPORT", "imports", module=module, exception_type=type(exc).__name__
            ) from None
    return result


def compare_inventory(actual, expected):
    if actual["python"].split(".")[:2] != expected["python"].split(".")[:2]:
        raise Fault(
            "E_IMPORT",
            "python.version",
            expected=expected["python"],
            actual=actual["python"],
        )
    for name, version in expected["packages"].items():
        if actual["packages"].get(name) != version:
            raise Fault(
                "E_IMPORT",
                "dependency.version",
                module=name,
                expected=version,
                actual=actual["packages"].get(name),
            )
    for name, source in expected["sources"].items():
        got = actual["sources"].get(name, {})
        if got.get("sha256") != source["sha256"]:
            raise Fault(
                "E_SOURCE", "source.hash", module=name, expected=source, actual=got
            )


def check_redis(config, probe=None, *, driver=False):
    import redis
    from redis.sentinel import Sentinel

    transport = config["transport"]
    stage = "redis.driver" if driver else "redis.consumer"
    password = secret(transport.get("password_ref"), stage)
    sentinel_password = secret(transport.get("sentinel_password_ref"), stage)
    nodes = transport.get("driver_sentinel_nodes") if driver else None
    nodes = nodes or transport["sentinel_nodes"]
    sentinels, failures = [], []
    for address in nodes:
        node = urlsplit("redis://" + address)
        conn = redis.Redis(
            host=node.hostname,
            port=node.port,
            password=sentinel_password,
            socket_timeout=4,
            socket_connect_timeout=4,
            decode_responses=True,
        )
        try:
            conn.ping()
            master = conn.sentinel_get_master_addr_by_name(transport["sentinel_master"])
            if not master:
                raise ValueError()
            sentinels.append((node.hostname, node.port))
        except Exception as exc:
            failures.append({"node": address, "exception_type": type(exc).__name__})
        finally:
            conn.close()
    if not sentinels:
        raise Fault("E_SENTINEL", stage, nodes=failures)
    if failures:
        emit("WARN", stage, code="SENTINEL_PARTIAL", unavailable=failures)
    client = None
    try:
        sentinel = Sentinel(
            sentinels,
            sentinel_kwargs={
                "password": sentinel_password,
                "socket_timeout": 4,
                "socket_connect_timeout": 4,
            },
            password=password,
            db=transport.get("database", 0),
            socket_timeout=4,
            socket_connect_timeout=4,
        )
        client = sentinel.master_for(transport["sentinel_master"])
        client.ping()
        # Read-only check; does not create groups or consume business messages.
        for channel in config["channels"].values():
            client.exists(channel["task_stream_key"])
        if (probe or {}).get("redis_stream_permissions", True):
            stream = "tributo-doctor:" + uuid.uuid4().hex
            created = False
            try:
                entry = client.xadd(stream, {"probe": "1"})
                created = True
                client.xgroup_create(stream, "doctor", id="0-0")
                if not client.xreadgroup("doctor", "doctor", {stream: ">"}, count=1):
                    raise ValueError()
                if client.xack(stream, "doctor", entry) != 1:
                    raise ValueError()
            finally:
                if created:
                    client.delete(stream)
        else:
            emit("WARN", stage, code="REDIS_STREAM_ACL_NOT_CHECKED")
    except Exception as exc:
        raise Fault("E_REDIS", stage, exception_type=type(exc).__name__) from None
    finally:
        if client is not None:
            client.close()
    return {
        "reachable_sentinels": len(sentinels),
        "database": transport.get("database", 0),
        "stream_permissions": "temporary tributo-doctor prefix only"
        if (probe or {}).get("redis_stream_permissions", True)
        else "NOT_CHECKED",
    }


def quote_table(identifier):
    parts = identifier.split(".")
    if len(parts) not in {1, 2} or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts
    ):
        raise Fault(
            "E_CONFIG",
            "clickhouse.table",
            hint_detail="自查表名只接受 database.table / table 常规标识符",
        )
    return ".".join("`" + part + "`" for part in parts)


def check_native_s3_configuration(
    boto3, probe_client, explicit_credentials, client_config
):
    """Match the native S3 path used by model/UBJ/SHAP reads without logging keys."""
    native = None
    try:
        native = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"), config=client_config
        )
        if native.meta.endpoint_url.rstrip(
            "/"
        ) != probe_client.meta.endpoint_url.rstrip("/"):
            raise Fault(
                "E_S3",
                "s3.native_endpoint",
                hint_detail=(
                    "probe endpoint 必须匹配真实作业的 "
                    "AWS_ENDPOINT_URL / AWS 服务 endpoint 配置"
                ),
            )
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise Fault(
                "E_SECRET",
                "s3.native_credentials",
                hint_detail=(
                    "真实模型读取依赖 boto3 默认凭据链；仅在 probe 显式填写凭据引用不够"
                ),
            )
        frozen = credentials.get_frozen_credentials()
        for key, actual in (
            ("aws_access_key_id", frozen.access_key),
            ("aws_secret_access_key", frozen.secret_key),
            ("aws_session_token", frozen.token),
        ):
            if key in explicit_credentials and not hmac.compare_digest(
                str(explicit_credentials[key] or ""), str(actual or "")
            ):
                raise Fault(
                    "E_S3",
                    "s3.native_credentials",
                    credential_field=key,
                    hint_detail=(
                        "probe 引用与真实作业默认凭据链不同；"
                        "核对 Ray Pod 的 AWS_* / IAM"
                    ),
                )
    except Fault:
        raise
    except Exception as exc:
        raise Fault(
            "E_S3", "s3.native_configuration", exception_type=type(exc).__name__
        ) from None
    finally:
        if native is not None:
            native.close()


def check_storage(probe):
    result = {}
    cfg = probe.get("clickhouse")
    if cfg:
        import clickhouse_connect

        password = secret(cfg.get("password_ref"), "clickhouse.credentials")
        client = None
        try:
            client = clickhouse_connect.get_client(
                host=cfg["host"],
                port=cfg.get("port", 8123),
                database=cfg.get("database", "default"),
                username=cfg.get("username", "default"),
                password=password or "",
                secure=cfg.get("secure", False),
                connect_timeout=5,
                send_receive_timeout=10,
            )
            client.command("SELECT 1")
            for key in ("input_table", "output_table"):
                if cfg.get(key):
                    # A qualified table must be quoted by segment; a single
                    # Identifier parameter quotes database.table as one name.
                    client.query("SELECT * FROM " + quote_table(cfg[key]) + " LIMIT 0")
            result["clickhouse"] = "SELECT passed"
            if cfg.get("output_table"):
                version = str(client.command("SELECT version()"))
                numbers = tuple(int(part) for part in version.split(".")[:2])
                if numbers >= (24, 11):
                    # CHECK GRANT does not support Identifier query parameters.
                    # Quote only strictly validated identifier segments instead.
                    granted = client.command(
                        "CHECK GRANT INSERT ON " + quote_table(cfg["output_table"])
                    )
                    if int(granted) != 1:
                        raise Fault(
                            "E_CK",
                            "clickhouse.insert_grant",
                            operation="INSERT",
                            granted=False,
                        )
                    result["clickhouse_insert_grant"] = "PASS (non-writing CHECK GRANT)"
                else:
                    result["clickhouse_insert_grant"] = (
                        "NOT_CHECKED: CHECK GRANT requires ClickHouse >=24.11"
                    )
            else:
                result["clickhouse_insert_grant"] = "NOT_CHECKED: no output_table"

        except Fault:
            raise
        except Exception as exc:
            raise Fault(
                "E_CK", "clickhouse", exception_type=type(exc).__name__
            ) from None
        finally:
            if client is not None:
                client.close()
    else:
        result["clickhouse"] = "NOT_CHECKED: no explicit task datasource probe"
    cfg = probe.get("s3")
    if cfg:
        import boto3
        from botocore.config import Config

        kwargs = {
            key: cfg[key] for key in ("endpoint_url", "region_name") if cfg.get(key)
        }
        for source, target in (
            ("access_key_ref", "aws_access_key_id"),
            ("secret_key_ref", "aws_secret_access_key"),
            ("session_token_ref", "aws_session_token"),
        ):
            if cfg.get(source):
                kwargs[target] = secret(cfg[source], "s3.credentials")
        client_config = Config(
            connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}
        )
        try:
            client = boto3.client("s3", config=client_config, **kwargs)
        except Exception as exc:
            raise Fault(
                "E_S3", "s3.client", exception_type=type(exc).__name__
            ) from None
        key = "tributo-doctor/" + uuid.uuid4().hex
        created = False
        try:
            check_native_s3_configuration(boto3, client, kwargs, client_config)
            client.head_bucket(Bucket=cfg["bucket"])
            client.put_object(Bucket=cfg["bucket"], Key=key, Body=b"tributo-doctor")
            created = True
            if (
                client.get_object(Bucket=cfg["bucket"], Key=key)["Body"].read()
                != b"tributo-doctor"
            ):
                raise ValueError()
        except Fault:
            raise
        except Exception as exc:
            raise Fault(
                "E_S3", "s3", exception_type=type(exc).__name__, probe_key=key
            ) from None
        finally:
            if created:
                try:
                    client.delete_object(Bucket=cfg["bucket"], Key=key)
                except Exception as exc:
                    raise Fault(
                        "E_S3",
                        "s3.cleanup",
                        exception_type=type(exc).__name__,
                        probe_key=key,
                    ) from None
            client.close()
        result["s3"] = (
            "native boto3 endpoint/credentials match; HEAD/PUT/GET/DELETE passed"
        )
    else:
        result["s3"] = "NOT_CHECKED: no S3 probe (check shared filesystem separately)"
    return result


def check_environment(probe):
    for name in probe.get("required_env", []):
        secret("env:" + name, "environment")


def node_probe(config, probe, expected):
    try:
        actual = inventory(probe)
        compare_inventory(actual, expected)
        check_environment(probe)
        return {
            "status": "PASS",
            "inventory": actual,
            "redis": check_redis(config, probe, driver=True),
            "storage": check_storage(probe),
        }
    except Exception as exc:
        return failure(exc, "node", "E_RAY")


def driver(directory):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    config, probe, expected = [
        json.loads((directory / name).read_text())
        for name in ("config.json", "probe.json", "expected.json")
    ]
    ray.init(address="auto", log_to_driver=False)
    try:
        own = node_probe(config, probe, expected)
        own_status = own.pop("status")
        if "stage" in own:
            own["check_stage"] = own.pop("stage")
        emit(own_status, "ray.driver", **own)
        if own.get("code"):
            return 1
        nodes = [node for node in ray.nodes() if node["Alive"]]
        cpus = sum(node.get("Resources", {}).get("CPU", 0) for node in nodes)
        if len(nodes) < int(probe.get("min_alive_nodes", 1)) or cpus < float(
            probe.get("min_total_cpus", 0)
        ):
            raise Fault(
                "E_TOPOLOGY", "ray.topology", alive_nodes=len(nodes), total_cpus=cpus
            )
        refs = [
            ray.remote(num_cpus=0)(node_probe)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node["NodeID"], soft=False
                )
            )
            .remote(config, probe, expected)
            for node in nodes
        ]
        results = ray.get(refs, timeout=120)
        for node, result in zip(nodes, results, strict=True):
            result_status = result.pop("status")
            if "stage" in result:
                result["check_stage"] = result.pop("stage")
            emit(result_status, "ray.node", node_id=node["NodeID"], **result)
        if any(result.get("code") for result in results):
            return 1
        emit("PASS", "ray.complete", alive_nodes=len(nodes), total_cpus=cpus)
        return 0
    finally:
        ray.shutdown()


def active_jobs(client):
    busy = [
        {"submission_id": job.submission_id, "status": str(job.status)}
        for job in client.list_jobs()
        if (job.submission_id or "").startswith("tributo-")
        and not job.status.is_terminal()
    ]
    if busy:
        raise Fault("E_BUSY", "ray.active_jobs", jobs=busy)
    emit("PASS", "ray.idle")


def run(config_path, probe_path, timeout, idle_only=False):
    from ray.job_submission import JobSubmissionClient

    config, probe = (
        json.loads(config_path.read_text()),
        json.loads(probe_path.read_text()),
    )
    validate_probe(probe)
    client = JobSubmissionClient(config["execution"]["ray_dashboard_url"])
    if idle_only:
        active_jobs(client)
        return 0
    from tributo._common import build_runtime_env

    from tributo_knova.broker import KnovaBrokerPlugin

    KnovaBrokerPlugin().validate_config(config, check_connectivity=False)
    if config["transport"]["mode"] != "sentinel":
        raise Fault("E_CONFIG", "redis.mode")
    execution = config["execution"]
    if {"PATH", "PYTHONPATH"} & set(execution.get("env_vars", {})):
        raise Fault("E_CONFIG", "execution.env_vars")
    emit("PASS", "config")
    expected = inventory(probe)
    emit("PASS", "consumer.imports", inventory=expected)
    check_environment(probe)
    emit("PASS", "consumer.redis", **check_redis(config, probe))
    if probe.get("require_idle", True):
        active_jobs(client)
    submission_id = "knova-doctor-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="knova-doctor-") as temp:
        package = Path(temp) / "_knova_doctor"
        package.mkdir()
        (package / "__init__.py").write_text("")
        shutil.copyfile(__file__, package / "doctor.py")
        for name, value in (
            ("config.json", config),
            ("probe.json", probe),
            ("expected.json", expected),
        ):
            (package / name).write_text(json.dumps(value))
        runtime_env = build_runtime_env(
            project_root=Path(execution["project_root"]).resolve(),
            extra_py_modules=execution.get("extra_py_modules", []),
            runtime_pip_packages=execution.get("runtime_pip_packages", []),
            env_vars=execution.get("env_vars", {}),
        )
        runtime_env["py_modules"].append(str(package))
        client.submit_job(
            entrypoint="python -m _knova_doctor.doctor --driver",
            submission_id=submission_id,
            entrypoint_num_cpus=0,
            runtime_env=runtime_env,
        )
    emit("PASS", "ray.submitted", submission_id=submission_id)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            status = client.get_job_status(submission_id)
            if status.is_terminal():
                # Only structured lines; never print SDK exceptions or raw logs.
                records = []
                for line in client.get_job_logs(submission_id).splitlines():
                    if line.startswith(PREFIX):
                        record = json.loads(line[len(PREFIX) :])
                        records.append(record)
                        print(
                            PREFIX + json.dumps(record, ensure_ascii=False), flush=True
                        )
                if str(status) != "SUCCEEDED" or not any(
                    row.get("stage") == "ray.complete" for row in records
                ):
                    raise Fault(
                        "E_RAY",
                        "ray.job",
                        submission_id=submission_id,
                        job_status=str(status),
                    )
                emit("PASS", "complete", submission_id=submission_id)
                return 0
            time.sleep(1)
        raise Fault("E_RAY", "ray.timeout", submission_id=submission_id)
    except BaseException:
        client.stop_job(submission_id)
        raise


def main():
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("/etc/tributo-knova/config.json")
    )
    parser.add_argument(
        "--probe", type=Path, default=Path("/etc/tributo-knova/probe.json")
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    try:
        return (
            driver(Path(__file__).parent)
            if args.driver
            else run(args.config, args.probe, args.timeout, args.idle_only)
        )
    except Exception as exc:
        record = failure(exc, "doctor", "E_RAY")
        print(PREFIX + json.dumps(record, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
