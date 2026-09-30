"""Consumer-only deployment tests. Fake Kubernetes state; no network/containers."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "consumer_only_deploy", ROOT / "deploy/production/consumer-only/deploy.py"
)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


@pytest.fixture
def settings(tmp_path):
    config = json.loads((deploy.HERE / "config.example.json").read_text())
    config["execution"]["ray_dashboard_url"] = "http://existing-ray:8265"
    value = {
        "context": "kind-test",
        "namespace": "existing",
        "name": "new-consumer",
        "image": "local:v2.5",
        "secret": "credentials",
        "output": str(tmp_path),
        "timeout_seconds": 30,
    }
    return (
        value,
        config,
        {
            "algorithms": ["xgboost"],
            "inference_formats": ["onnx"],
            "require_idle": True,
        },
    )


def old_deployment(name="old-consumer", managed=False):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": "existing",
            "uid": "uid-" + name,
            "labels": {deploy.LABEL: deploy.OWNER} if managed else {},
        },
        "spec": {
            "replicas": 2,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "spec": {"containers": [{"name": "old", "image": "old:image"}]}
            },
        },
        "status": {"readyReplicas": 2},
    }


class FakeKube:
    def __init__(self, old=None, failure=None):
        self.objects = (
            {("deployment", old["metadata"]["name"]): copy.deepcopy(old)} if old else {}
        )
        self.calls = []
        self.failure = failure

    def get(self, kind, name):
        return copy.deepcopy(self.objects.get((kind, name)))

    def scale_down(self, name):
        self.calls.append(("stop", name))
        item = self.objects.get(("deployment", name))
        if item:
            item["spec"]["replicas"] = 0

    def apply(self, item):
        self.calls.append(("apply", item["metadata"]["name"]))
        key = (item["kind"].lower(), item["metadata"]["name"])
        previous = self.objects.get(key, {})
        item = copy.deepcopy(item)
        item["metadata"]["uid"] = previous.get("metadata", {}).get(
            "uid", "uid-" + item["metadata"]["name"]
        )
        self.objects[key] = item

    def rollout(self, name):
        self.calls.append(("rollout", name))
        if self.failure == name:
            raise deploy.Error("E_ROLLOUT", "rollout failed")
        self.objects[("deployment", name)]["status"] = {
            "readyReplicas": self.objects[("deployment", name)]["spec"].get(
                "replicas", 1
            )
        }

    def delete(self, kind, name):
        self.calls.append(("delete", name))
        self.objects.pop((kind, name), None)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["transport"].update(mode="standalone"),
        lambda c: c["transport"].update(sentinel_nodes=[]),
        lambda c: c["execution"]["env_vars"].update(PATH="/unexpected"),
        lambda c: c["execution"]["env_vars"].update(
            KNOVA_REDIS_PASSWORD="do-not-print"
        ),
        lambda c: c["transport"].update(password_ref="do-not-print"),
        lambda c: c["transport"].update(
            sentinel_nodes=["user:do-not-print@host:26379"]
        ),
        lambda c: c["execution"].update(
            ray_dashboard_url="http://user:do-not-print@ray:8265"
        ),
        lambda c: c.update(password="do-not-print"),
        lambda c: c.update(channels={}),
    ],
)
def test_config_rejects_invalid_without_exposing_values(settings, change):
    _, config, _ = settings
    change(config)
    with pytest.raises(deploy.Error) as err:
        deploy.validate_config(config)
    assert "do-not-print" not in str(err.value)


def test_render_only_cm_deployment_and_no_path(settings):
    values, config, probe = settings
    cm, deployment = deploy.manifests(values, config, probe)
    assert [cm["kind"], deployment["kind"]] == ["ConfigMap", "Deployment"]
    assert cm["immutable"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["tributo-knova"]
    assert container["envFrom"] == [{"secretRef": {"name": "credentials"}}]
    assert all(item["name"] != "PATH" for item in container["env"])
    first = cm["metadata"]["name"]
    config["transport"]["database"] = 4
    assert deploy.manifests(values, config, probe)[0]["metadata"]["name"] != first


def test_apply_needs_explicit_quiescence(settings, monkeypatch):
    values, config, probe = settings
    kube = FakeKube()
    monkeypatch.setattr(deploy, "check", lambda *a, **k: pytest.fail("must not check"))
    with pytest.raises(deploy.Error, match="upstream-paused"):
        deploy.deploy(kube, values, *deploy.manifests(values, config, probe), False)
    assert kube.calls == []


def test_preflight_failure_does_not_stop_old(settings, monkeypatch):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment())

    def fail(*a, **k):
        raise deploy.Error("E_PREFLIGHT", "bad sentinel password")

    monkeypatch.setattr(deploy, "check", fail)
    with pytest.raises(deploy.Error):
        deploy.deploy(kube, values, *deploy.manifests(values, config, probe), True)
    assert kube.calls == []
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 2


def test_successful_migration_and_manual_rollback(settings, monkeypatch):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment())
    checks = []
    monkeypatch.setattr(
        deploy, "check", lambda *a, **k: checks.append(k.get("idle_only", False))
    )
    cm, deployment = deploy.manifests(values, config, probe)
    deploy.deploy(kube, values, cm, deployment, True)
    state = json.loads((Path(values["output"]) / "state.json").read_text())
    assert state["phase"] == "complete"
    assert checks == [False, True]
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 0
    assert kube.get("deployment", "new-consumer")["spec"]["replicas"] == 1
    assert (Path(values["output"]) / "state.json").stat().st_mode & 0o777 == 0o600
    deploy.restore(kube, values, state, cm, deployment)
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 2
    assert kube.get("deployment", "new-consumer") is None
    assert (
        json.loads((Path(values["output"]) / "state.json").read_text())["phase"]
        == "rolled_back"
    )


def test_rollout_failure_automatically_restores_old(settings, monkeypatch):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment(), failure="new-consumer")
    monkeypatch.setattr(deploy, "check", lambda *a, **k: [])
    with pytest.raises(deploy.Error):
        deploy.deploy(kube, values, *deploy.manifests(values, config, probe), True)
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 2
    assert kube.get("deployment", "new-consumer") is None
    assert kube.calls.index(("stop", "new-consumer")) < kube.calls.index(
        ("apply", "old-consumer")
    )


def test_rollback_busy_does_not_restart_old(settings, monkeypatch):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment(), failure="new-consumer")

    def check(*args, **kwargs):
        if ("rollout", "new-consumer") in kube.calls:
            raise deploy.Error("E_BUSY", "job still running")

    monkeypatch.setattr(deploy, "check", check)
    with pytest.raises(deploy.Error):
        deploy.deploy(kube, values, *deploy.manifests(values, config, probe), True)
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 0
    assert kube.get("deployment", "new-consumer")["spec"]["replicas"] == 0
    assert (
        json.loads((Path(values["output"]) / "state.json").read_text())["phase"]
        == "rollback_stopped_new"
    )


def test_incomplete_journal_not_overwritten(settings, monkeypatch):
    values, config, probe = settings
    deploy.write_json(
        Path(values["output"]) / "state.json",
        {
            "phase": "old_stopped",
            "context": "kind-test",
            "namespace": "existing",
            "name": "new-consumer",
        },
    )
    monkeypatch.setattr(deploy, "check", lambda *a, **k: pytest.fail("must not check"))
    with pytest.raises(deploy.Error, match="上次切换未完成"):
        deploy.deploy(
            FakeKube(), values, *deploy.manifests(values, config, probe), True
        )


def test_refuses_unowned_target(settings, monkeypatch):
    values, config, probe = settings
    monkeypatch.setattr(deploy, "check", lambda *a, **k: pytest.fail("must not check"))
    with pytest.raises(deploy.Error, match="不属于此部署器"):
        deploy.deploy(
            FakeKube(old_deployment("new-consumer")),
            values,
            *deploy.manifests(values, config, probe),
            True,
        )


def test_original_deployment_updated_in_place_and_restored(settings, monkeypatch):
    values, config, probe = settings
    kube = FakeKube(old_deployment("new-consumer", managed=True))
    monkeypatch.setattr(deploy, "check", lambda *a, **k: [])
    cm, deployment = deploy.manifests(values, config, probe)
    deploy.deploy(kube, values, cm, deployment, True)
    state = json.loads((Path(values["output"]) / "state.json").read_text())
    deploy.restore(kube, values, state, cm, deployment)
    restored = kube.get("deployment", "new-consumer")
    assert restored["spec"]["replicas"] == 2
    assert restored["spec"]["template"]["spec"]["containers"][0]["image"] == "old:image"


def test_diagnostics_do_not_print_exception_messages():
    pod = {
        "status": {
            "phase": "Failed",
            "containerStatuses": [
                {
                    "name": "consumer",
                    "state": {
                        "terminated": {
                            "exitCode": 1,
                            "reason": "Error",
                            "message": "SECRET",
                        }
                    },
                }
            ],
        }
    }
    assert "SECRET" not in json.dumps(deploy.diagnose_pod(pod))


def test_identical_apply_preserves_pods_and_original_rollback(settings, monkeypatch):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment())
    checks = []
    monkeypatch.setattr(deploy, "check", lambda *a, **k: checks.append(True))
    cm, deployment = deploy.manifests(values, config, probe)
    deploy.deploy(kube, values, cm, deployment, True)
    original = (Path(values["output"]) / "state.json").read_text()
    calls = list(kube.calls)
    deploy.deploy(kube, values, cm, deployment, True)
    assert kube.calls == calls
    assert len(checks) == 2
    assert (Path(values["output"]) / "state.json").read_text() == original
    deploy.restore(kube, values, json.loads(original), cm, deployment)
    assert kube.get("deployment", "old-consumer")["spec"]["replicas"] == 2


@pytest.mark.parametrize("changed", ["old-consumer", "new-consumer"])
def test_rollback_refuses_same_name_replacement(settings, monkeypatch, changed):
    values, config, probe = settings
    values["old_deployment"] = "old-consumer"
    kube = FakeKube(old_deployment())
    monkeypatch.setattr(deploy, "check", lambda *a, **k: [])
    cm, deployment = deploy.manifests(values, config, probe)
    deploy.deploy(kube, values, cm, deployment, True)
    state = json.loads((Path(values["output"]) / "state.json").read_text())
    kube.objects[("deployment", changed)]["metadata"]["uid"] = "external-replacement"
    calls = list(kube.calls)
    with pytest.raises(deploy.Error, match="身份已变更"):
        deploy.restore(kube, values, state, cm, deployment)
    assert kube.calls == calls


class PreflightKube:
    def __init__(self, *, phase="Succeeded", records=None, unowned_cm=False):
        self.applied = []
        self.deleted = []
        self.phase = phase
        self.records = (
            records
            if records is not None
            else [{"status": "PASS", "stage": "complete"}]
        )
        self.unowned_cm = unowned_cm

    def get(self, kind, name):
        if kind in {"namespace", "secret"}:
            return {"metadata": {"name": name}}
        if kind == "configmap":
            return {"metadata": {"labels": {}}} if self.unowned_cm else None
        if kind == "pod":
            return {"metadata": {"name": name}, "status": {"phase": self.phase}}
        return None

    def apply(self, document):
        self.applied.append(copy.deepcopy(document))

    def delete(self, kind, name):
        self.deleted.append((kind, name))

    def run(self, *args):
        assert args[0] == "logs"
        return "unsafe raw SDK line SECRET\n" + "\n".join(
            "KNOVA_DOCTOR " + json.dumps(row) for row in self.records
        )


def test_preflight_never_starts_consumer_and_filters_logs(settings, capsys):
    values, config, probe = settings
    kube = PreflightKube()
    records = deploy.check(kube, values, *deploy.manifests(values, config, probe))
    assert records == [{"status": "PASS", "stage": "complete"}]
    assert [obj["kind"] for obj in kube.applied] == ["ConfigMap", "Pod"]
    container = kube.applied[1]["spec"]["containers"][0]
    assert container["command"] == ["python", "/etc/tributo-knova/doctor.py"]
    assert "readinessProbe" not in container
    assert kube.deleted[0][0] == "pod"
    report_file = next(Path(values["output"]).glob("*-check-*.json"))
    assert "SECRET" not in report_file.read_text()
    assert "SECRET" not in capsys.readouterr().out


def test_preflight_failure_preserves_safe_report_and_cleans_pod(settings):
    values, config, probe = settings
    kube = PreflightKube(
        phase="Failed",
        records=[{"status": "FAIL", "stage": "redis.driver", "code": "E_SECRET"}],
    )
    with pytest.raises(deploy.Error) as caught:
        deploy.check(kube, values, *deploy.manifests(values, config, probe))
    assert caught.value.code == "E_PREFLIGHT"
    assert len(kube.deleted) == 1
    report_file = next(Path(values["output"]).glob("*-check-*.json"))
    assert json.loads(report_file.read_text())["records"][0]["code"] == "E_SECRET"


def test_preflight_refuses_unowned_configmap(settings):
    values, config, probe = settings
    kube = PreflightKube(unowned_cm=True)
    with pytest.raises(deploy.Error) as caught:
        deploy.check(kube, values, *deploy.manifests(values, config, probe))
    assert caught.value.code == "E_OWNERSHIP"
    assert kube.applied == []


def test_kubectl_context_is_explicit_and_forbidden_does_not_leak(settings, monkeypatch):
    from types import SimpleNamespace

    values, _, _ = settings
    invoked = []

    def fake_run(command, **kwargs):
        invoked.append(command)
        return SimpleNamespace(
            returncode=1, stdout="", stderr="Error from server (Forbidden): SECRET"
        )

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    with pytest.raises(deploy.Error) as caught:
        deploy.Kube(values).run("get", "pods")
    assert invoked[0][:5] == [
        "kubectl",
        "--context",
        "kind-test",
        "--namespace",
        "existing",
    ]
    assert caught.value.code == "E_KUBE_FORBIDDEN"
    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize(
    "exception, code",
    [
        ("redis.exceptions.NoPermissionError", "E_REDIS_ACL"),
        ("redis.exceptions.AuthenticationError", "E_REDIS_AUTH"),
        ("RedisCredentialUnavailable", "E_SECRET"),
        ("ModuleNotFoundError", "E_IMPORT"),
        ("ImportError", "E_IMPORT"),
        ("ValidationError", "E_CONFIG"),
        ("MemoryError", "E_MEMORY"),
    ],
)
def test_safe_log_classifier_drops_all_raw_message_values(exception, code):
    classified = deploy.classify_consumer_log(
        exception + ": password=SECRET-NOT-LOGGED"
    )
    assert classified["code"] == code
    assert "SECRET-NOT-LOGGED" not in json.dumps(classified)
    assert "password=" not in json.dumps(classified)


def test_unknown_log_does_not_leak_or_claim_known_cause():
    assert (
        deploy.classify_consumer_log("unrecognized failure SECRET")["code"]
        == "E_LOG_UNCLASSIFIED"
    )
    assert "SECRET" not in json.dumps(
        deploy.classify_consumer_log("unrecognized failure SECRET")
    )


def test_rollout_diagnostics_classify_logs_without_persisting_raw_text(
    settings, capsys
):
    values, _, _ = settings

    class Kube:
        def get(self, kind, name):
            return {"spec": {"selector": {"matchLabels": {"app": "consumer"}}}}

        def run(self, *args):
            if args[0] == "get":
                return json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "pod"},
                                "status": {"phase": "Running"},
                            }
                        ]
                    }
                )
            return "redis.exceptions.NoPermissionError: raw-password=SECRET"

    deploy.save_diagnostics(Kube(), values)
    record = (Path(values["output"]) / "rollout-failure.json").read_text()
    assert json.loads(record)["pods"][0]["diagnostic"]["code"] == "E_REDIS_ACL"
    assert "SECRET" not in record + capsys.readouterr().out
