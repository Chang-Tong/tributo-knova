"""Run in a preflight Job; check each Ray node without exposing credentials."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
def check_node(config, bucket, shared_marker):
    import importlib
    import platform
    import socket

    from tributo_broker_redis.config import RedisConnectionConfig
    from tributo_broker_redis.driver_transport import read_driver_input

    assert callable(read_driver_input)
    from tributo_broker_redis.redis_client import create_redis_client

    subprocess.run(["python", "/opt/tributo-runtime/verify_image.py"], check=True)
    boosting_path = importlib.import_module(
        "tributo_algorithms_boosting.native"
    ).__file__
    transport = config["transport"].copy()
    if transport.get("driver_url"):
        transport["url"] = transport["driver_url"]
    if transport.get("driver_sentinel_nodes"):
        transport["sentinel_nodes"] = transport["driver_sentinel_nodes"]
    allowed = RedisConnectionConfig.model_fields
    connection = RedisConnectionConfig.model_validate(
        {key: value for key, value in transport.items() if key in allowed}
    )
    client = create_redis_client(connection)
    try:
        assert client.ping()
    finally:
        client.close()
    if shared_marker:
        assert Path(shared_marker).read_text() == "tributo-preflight"
        node_file = Path(shared_marker + "." + socket.gethostname())
        node_file.write_text("worker-can-write")
    if bucket:
        import boto3

        client = boto3.client("s3")
        client.head_bucket(Bucket=bucket)
        # A private temporary object verifies Put/Get/Delete permissions too.
        import uuid

        key = "tributo-preflight/" + uuid.uuid4().hex
        created = False
        try:
            client.put_object(Bucket=bucket, Key=key, Body=b"tributo-preflight")
            created = True
            assert (
                client.get_object(Bucket=bucket, Key=key)["Body"].read()
                == b"tributo-preflight"
            )
        finally:
            if created:
                client.delete_object(Bucket=bucket, Key=key)
    return {
        "node": socket.gethostname(),
        "architecture": platform.machine(),
        "redis": "ok",
        "s3": "ok" if bucket else "not-configured",
        "driver_path": importlib.import_module(
            "tributo_knova.execution_driver"
        ).__file__,
        "boosting_path": boosting_path,
        "shared_marker": str(node_file) if shared_marker else None,
    }


def check_cluster():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    ray.init(address="auto")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    expected = int(os.environ["EXPECTED_NODES"])
    if len(nodes) != expected:
        raise RuntimeError(f"expected {expected} live Ray nodes, found {len(nodes)}")
    shared = os.environ.get("SHARED_CHECK_PATH", "")
    marker = ""
    if shared:
        import uuid

        marker = str(Path(shared) / (".tributo-check-" + uuid.uuid4().hex))
        Path(marker).write_text("tributo-preflight")
    checks = [
        check_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        ).remote(config, os.environ.get("S3_CHECK_BUCKET", ""), marker)
        for node in nodes
    ]
    try:
        results = ray.get(checks, timeout=120)
        for result in results:
            if result["shared_marker"]:
                assert Path(result["shared_marker"]).read_text() == "worker-can-write"
        print(json.dumps({"nodes": results}, indent=2), flush=True)
    finally:
        if marker:
            for path in Path(shared).glob(Path(marker).name + "*"):
                path.unlink(missing_ok=True)
        ray.shutdown()


def submit_check(config_path=Path("/deploy/config.json")):
    import uuid

    from ray.job_submission import JobSubmissionClient
    from tributo._common import build_runtime_env

    config = json.loads(config_path.read_text())
    subprocess.run(
        ["tributo-knova", "health", "--config", str(config_path)], check=True
    )
    execution = config["execution"]
    client = JobSubmissionClient(execution["ray_dashboard_url"])
    submission_id = "knova-deploy-check-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="knova-check-") as directory:
        package = Path(directory) / "_tributo_knova_preflight"
        package.mkdir()
        (package / "__init__.py").write_text("")
        shutil.copyfile(__file__, package / "check_cluster.py")
        shutil.copyfile(config_path, package / "config.json")
        runtime_env = build_runtime_env(
            project_root=Path(execution["project_root"]).expanduser().resolve(),
            extra_py_modules=[
                module if "://" in module else str(Path(module).expanduser().resolve())
                for module in execution.get("extra_py_modules", [])
            ],
            runtime_pip_packages=list(execution.get("runtime_pip_packages", [])),
            env_vars=dict(execution.get("env_vars", {})),
        )
        runtime_env["py_modules"].append(str(package))
        runtime_env.setdefault("env_vars", {}).update(
            {
                key: os.environ.get(key, "")
                for key in ("EXPECTED_NODES", "S3_CHECK_BUCKET", "SHARED_CHECK_PATH")
            }
        )
        client.submit_job(
            entrypoint="python -m _tributo_knova_preflight.check_cluster --driver",
            submission_id=submission_id,
            entrypoint_num_cpus=0,
            runtime_env=runtime_env,
        )
    print("preflight submission_id=" + submission_id, flush=True)
    try:
        deadline = time.monotonic() + 160
        while time.monotonic() < deadline:
            status = client.get_job_status(submission_id)
            if status.is_terminal():
                print(client.get_job_logs(submission_id), flush=True)
                if str(status) != "SUCCEEDED":
                    raise RuntimeError("Ray preflight failed; inspect " + submission_id)
                return
            time.sleep(1)
        raise TimeoutError("Ray preflight timed out: " + submission_id)
    except Exception:
        client.stop_job(submission_id)
        raise


if __name__ == "__main__":
    if "--driver" in sys.argv:
        check_cluster()
    else:
        submit_check()
