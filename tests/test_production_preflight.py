"""Local checks for the Ray preflight contract; never start Ray or contact services."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ray.job_submission import JobStatus
from tributo._common import build_runtime_env

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "knova_preflight", ROOT / "deploy/production/check_cluster.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize("failure", [None, "missing-boosting", "image-mismatch"])
def test_node_verifies_image_and_imports_distributed_boosting(monkeypatch, failure):
    calls = []

    class Connection:
        model_fields = {"url": None}

        @classmethod
        def model_validate(cls, values):
            return values

    class Client:
        def ping(self):
            calls.append("ping")
            return True

        def close(self):
            calls.append("close")

    def verify(command, *, check):
        assert command == ["python", "/opt/tributo-runtime/verify_image.py"]
        assert check is True
        calls.append("verify-image")
        if failure == "image-mismatch":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(preflight.subprocess, "run", verify)
    modules = {
        "tributo_broker_redis.config": SimpleNamespace(
            RedisConnectionConfig=Connection
        ),
        "tributo_broker_redis.driver_transport": SimpleNamespace(
            read_driver_input=lambda: None
        ),
        "tributo_broker_redis.redis_client": SimpleNamespace(
            create_redis_client=lambda connection: Client()
        ),
        "tributo_knova.execution_driver": SimpleNamespace(
            __file__="/ray/py_modules/tributo_knova/execution_driver.py"
        ),
        "tributo_algorithms_boosting.native": (
            None
            if failure == "missing-boosting"
            else SimpleNamespace(
                __file__="/ray/py_modules/tributo_algorithms_boosting/native.py"
            )
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    config = {"transport": {"url": "redis://unused-local-test:6379"}}
    if failure:
        error = (
            ModuleNotFoundError
            if failure == "missing-boosting"
            else subprocess.CalledProcessError
        )
        with pytest.raises(error):
            preflight.check_node._function(config, "", "")
        assert calls == ["verify-image"]
    else:
        result = preflight.check_node._function(config, "", "")
        assert result["boosting_path"] == (
            "/ray/py_modules/tributo_algorithms_boosting/native.py"
        )
        assert result["redis"] == "ok"
        assert calls == ["verify-image", "ping", "close"]


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "business"
    (root / "tributo").mkdir(parents=True)
    (root / "tributo" / "__init__.py").write_text("")
    extension = tmp_path / "extensions" / "sample"
    extension.mkdir(parents=True)
    (extension / "__init__.py").write_text("")
    execution = {
        "ray_dashboard_url": "http://unused-local-test:8265",
        "project_root": "business",
        "extra_py_modules": ["extensions/sample", "s3://unused-bucket/extension.zip"],
        "runtime_pip_packages": ["example-runtime==1.0"],
        "env_vars": {
            "PATH": "/business/bin:/usr/bin",
            "AWS_ENDPOINT_URL": "http://business-s3:9000",
            "PYTHONPATH": "/business/custom-modules",
            "EXPECTED_NODES": "999",
        },
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"execution": execution}))
    monkeypatch.setenv("EXPECTED_NODES", "2")
    monkeypatch.setenv("S3_CHECK_BUCKET", "check-bucket")
    monkeypatch.setenv("SHARED_CHECK_PATH", "/shared/check")
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY", "local-test-secret-must-not-be-uploaded"
    )
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: None)
    return config, execution


@pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED])
def test_submit_check_preserves_business_environment(
    configuration, monkeypatch, status
):
    config, execution = configuration
    submitted = {}
    stopped = []

    class Client:
        def __init__(self, address):
            assert address == execution["ray_dashboard_url"]

        def submit_job(self, **kwargs):
            submitted.update(kwargs)
            package = Path(kwargs["runtime_env"]["py_modules"][-1])
            assert package.name == "_tributo_knova_preflight"
            assert (package / "__init__.py").is_file()
            assert (package / "config.json").read_bytes() == config.read_bytes()
            assert (package / "check_cluster.py").read_bytes() == Path(
                preflight.__file__
            ).read_bytes()

        def get_job_status(self, submission_id):
            assert submission_id == submitted["submission_id"]
            return status

        def get_job_logs(self, submission_id):
            return "local preflight result"

        def stop_job(self, submission_id):
            stopped.append(submission_id)

    monkeypatch.setattr("ray.job_submission.JobSubmissionClient", Client)
    if status == JobStatus.FAILED:
        with pytest.raises(RuntimeError, match="Ray preflight failed"):
            preflight.submit_check(config)
        assert stopped == [submitted["submission_id"]]
    else:
        preflight.submit_check(config)
        assert not stopped

    business = build_runtime_env(
        project_root=Path(execution["project_root"]).resolve(),
        extra_py_modules=[
            str(Path("extensions/sample").resolve()),
            "s3://unused-bucket/extension.zip",
        ],
        runtime_pip_packages=execution["runtime_pip_packages"],
        env_vars=execution["env_vars"],
    )
    runtime = submitted["runtime_env"]
    assert runtime["working_dir"] == business["working_dir"]
    assert runtime["py_modules"][:-1] == business["py_modules"]
    assert runtime["pip"] == business["pip"]
    assert runtime["excludes"] == business["excludes"]
    assert runtime["env_vars"] == {
        **business["env_vars"],
        "EXPECTED_NODES": "2",
        "S3_CHECK_BUCKET": "check-bucket",
        "SHARED_CHECK_PATH": "/shared/check",
    }
    assert submitted["entrypoint"] == (
        "python -m _tributo_knova_preflight.check_cluster --driver"
    )
    assert not Path(runtime["py_modules"][-1]).exists()
    assert "local-test-secret-must-not-be-uploaded" not in json.dumps(submitted)


def test_driver_reads_packaged_config_with_business_working_directory(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    package.mkdir()
    expected = {"marker": "packaged configuration"}
    (package / "config.json").write_text(json.dumps(expected))
    business = tmp_path / "business"
    business.mkdir()
    (business / "config.json").write_text(json.dumps({"marker": "wrong configuration"}))
    monkeypatch.chdir(business)
    monkeypatch.setattr(preflight, "__file__", str(package / "check_cluster.py"))
    monkeypatch.setenv("EXPECTED_NODES", "2")
    monkeypatch.setenv("SHARED_CHECK_PATH", "")
    monkeypatch.setenv("S3_CHECK_BUCKET", "check-bucket")
    checked = []
    shutdown = []

    class CheckNode:
        def options(self, **kwargs):
            return self

        def remote(self, config, bucket, marker):
            checked.append((config, bucket, marker))
            return {"shared_marker": None}

    monkeypatch.setattr(preflight, "check_node", CheckNode())
    monkeypatch.setattr(preflight.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        preflight.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": "01" * 28},
            {"Alive": True, "NodeID": "02" * 28},
        ],
    )
    monkeypatch.setattr(preflight.ray, "get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(preflight.ray, "shutdown", lambda: shutdown.append(True))
    preflight.check_cluster()
    assert checked == [(expected, "check-bucket", ""), (expected, "check-bucket", "")]
    assert shutdown == [True]
