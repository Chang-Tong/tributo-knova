"""Deployment guardrails. Real Kubernetes acceptance is a separate local gate."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "knova_deploy", ROOT / "deploy/production/deploy.py"
)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    config = json.loads((ROOT / "deploy/config/knova-sentinel.json").read_text())
    config["transport"]["sentinel_password_ref"] = "env:KNOVA_REDIS_SENTINEL_PASSWORD"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    argv = [
        "deploy.py",
        "apply",
        "--context",
        "test-only",
        "--namespace",
        "test",
        "--config",
        str(path),
        "--output",
        str(tmp_path / "release"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return argv, path


def test_all_roles_get_same_image_secret_and_mount(arguments):
    a = deploy.parse_args()
    a.shared_pvc = "nfs"
    items, job, required, _ = deploy.build(a)
    cluster = items[1]["spec"]
    templates = [
        cluster["headGroupSpec"]["template"],
        cluster["workerGroupSpecs"][0]["template"],
        items[2]["spec"]["template"],
        job["spec"]["template"],
    ]
    for template in templates:
        containers = template["spec"]["containers"]
        assert containers[0]["image"] == a.image
        assert containers[0]["envFrom"] == [{"secretRef": {"name": a.secret}}]
        assert (
            len([v for v in template["spec"]["volumes"] if v["name"] == "shared"]) == 1
        )
        assert (
            len([v for v in containers[0]["volumeMounts"] if v["name"] == "shared"])
            == 1
        )
    assert "KNOVA_REDIS_PASSWORD" in required
    assert "KNOVA_REDIS_SENTINEL_PASSWORD" in required
    assert items[2]["spec"]["replicas"] == 0


def test_head_uses_explicit_runtime_env_agent_port(arguments):
    items, _, _, _ = deploy.build(deploy.parse_args())
    head = items[1]["spec"]["headGroupSpec"]
    assert head["rayStartParams"]["runtime-env-agent-port"] == "52366"


def test_preflight_bucket_changes_release_digest(arguments):
    args = deploy.parse_args()
    original_digest = deploy.build(args)[3]
    args.s3_bucket = "new-preflight-bucket"
    items, job, _, changed_digest = deploy.build(args)
    assert changed_digest != original_digest
    for item in [*items, job]:
        assert (
            item["metadata"]["annotations"]["knova-deploy/spec-sha256"]
            == changed_digest
        )


@pytest.mark.parametrize(
    "key",
    ["KNOVA_REDIS_PASSWORD", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "TOKEN"],
)
def test_reject_secrets_in_runtime_env(arguments, key):
    _, path = arguments
    config = json.loads(path.read_text())
    config["execution"]["env_vars"][key] = "do-not-print-this"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError) as exc:
        deploy.build(deploy.parse_args())
    assert "do-not-print-this" not in str(exc.value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("image", "repo:latest"),
        ("image", "repo"),
        ("workers", 0),
        ("cpus", 0),
        ("name", "../bad"),
        ("shared_path", "/"),
    ],
)
def test_invalid_settings(arguments, name, value):
    a = deploy.parse_args()
    setattr(a, name, value)
    with pytest.raises(ValueError):
        deploy.build(a)


class FakeKubectl:
    def __init__(self):
        self.calls = []
        self.objects = {}
        self.fail_job = False
        self.fail_consumer = False
        self.missing_secret = False

    def __call__(self, argv, *, data=None, timeout=180):
        assert argv[:3] == ["kubectl", "--context", "test-only"]
        args = argv[6:]
        self.calls.append(args)
        command = args[0]
        if command == "get":
            kind = args[1].lower()
            if kind == "job" and (kind, args[2]) in self.objects:
                return json.dumps(
                    {
                        "metadata": self.objects[(kind, args[2])]["metadata"],
                        "status": {
                            "conditions": [
                                {
                                    "type": "Failed" if self.fail_job else "Complete",
                                    "status": "True",
                                }
                            ]
                        },
                    }
                )
            if kind == "nodes":
                return json.dumps(
                    {
                        "items": [
                            {
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}]
                                }
                            }
                        ]
                    }
                )
            if kind == "secret":
                return json.dumps(
                    {
                        "data": {}
                        if self.missing_secret
                        else {
                            "KNOVA_REDIS_PASSWORD": "eA==",
                            "KNOVA_REDIS_SENTINEL_PASSWORD": "eA==",
                        }
                    }
                )
            if kind in {"namespace", "crd"}:
                return "{}"
            return (
                json.dumps(self.objects[(kind, args[2])])
                if (kind, args[2]) in self.objects
                else ""
            )
        if command == "apply" and "--dry-run=server" not in args:
            for item in json.loads(data)["items"]:
                item["metadata"]["uid"] = item["metadata"]["name"] + "-uid"
                self.objects[(item["kind"].lower(), item["metadata"]["name"])] = item
        if command == "create" and data:
            item = json.loads(data)
            self.objects[(item["kind"].lower(), item["metadata"]["name"])] = item
        if command == "wait" and args[1].startswith("job/") and self.fail_job:
            raise RuntimeError("preflight failed")
        if (
            command == "rollout"
            and "deployment/tributo-knova-v25-consumer" in args
            and self.fail_consumer
            and self.objects[("deployment", "tributo-knova-v25-consumer")]["spec"][
                "replicas"
            ]
            > 0
        ):
            raise RuntimeError("consumer failed readiness")
        if command == "scale":
            self.objects[("deployment", args[2])]["spec"]["replicas"] = int(
                args[3].split("=")[1]
            )
        if command == "exec":
            return "[]"
        return "ok"


def test_missing_secret_blocks_mutation(arguments, monkeypatch):
    fake = FakeKubectl()
    fake.missing_secret = True
    monkeypatch.setattr(deploy, "run", fake)
    with pytest.raises(ValueError, match="missing required keys"):
        deploy.main()
    assert not any(call[0] in {"apply", "create", "scale"} for call in fake.calls)


def test_preflight_failure_never_starts_consumer(arguments, monkeypatch):
    fake = FakeKubectl()
    fake.fail_job = True
    monkeypatch.setattr(deploy, "run", fake)
    with pytest.raises(RuntimeError, match="preflight Job failed"):
        deploy.main()
    assert not any(call[0] == "scale" for call in fake.calls)


def test_success_idempotence_and_rollback(arguments, monkeypatch):
    fake = FakeKubectl()
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    count = len(fake.calls)
    deploy.main()
    assert not any(
        call[0] in {"apply", "scale", "delete", "create"}
        and "--dry-run=server" not in call
        for call in fake.calls[count:]
    )
    sys.argv[1] = "rollback"
    deploy.main()
    assert (
        fake.objects[("deployment", "tributo-knova-v25-consumer")]["spec"]["replicas"]
        == 0
    )


def test_active_release_cannot_silently_add_preflight_bucket(arguments, monkeypatch):
    fake = FakeKubectl()
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    count = len(fake.calls)
    sys.argv += ["--s3-bucket", "new-preflight-bucket"]
    with pytest.raises(ValueError, match="release configuration differs"):
        deploy.main()
    assert len(fake.calls) == count


def test_consumer_failure_restores_previous_replicas(arguments, monkeypatch):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.fail_consumer = True
    fake.objects[("deployment", "old")] = {"spec": {"replicas": 2}}
    monkeypatch.setattr(deploy, "run", fake)
    with pytest.raises(RuntimeError, match="consumer failed"):
        deploy.main()
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 2
    assert (
        fake.objects[("deployment", "tributo-knova-v25-consumer")]["spec"]["replicas"]
        == 0
    )


def test_failed_switch_can_retry_after_automatic_recovery(arguments, monkeypatch):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.fail_consumer = True
    fake.objects[("deployment", "old")] = {
        "metadata": {"uid": "old-uid"},
        "spec": {"replicas": 2},
    }
    monkeypatch.setattr(deploy, "run", fake)
    journal_path = deploy.parse_args().output / "switch.json"
    with pytest.raises(RuntimeError, match="consumer failed"):
        deploy.main()
    assert json.loads(journal_path.read_text())["state"] == "rolled_back"
    fake.fail_consumer = False
    deploy.main()
    assert json.loads(journal_path.read_text())["state"] == "active"
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 0
    assert (
        fake.objects[("deployment", "tributo-knova-v25-consumer")]["spec"]["replicas"]
        == 1
    )
    sys.argv[1] = "rollback"
    deploy.main()
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 2


def test_explicit_rollback_can_reapply_same_release(arguments, monkeypatch):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.objects[("deployment", "old")] = {
        "metadata": {"uid": "old-uid"},
        "spec": {"replicas": 2},
    }
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    sys.argv[1] = "rollback"
    deploy.main()
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 2
    sys.argv[1] = "apply"
    deploy.main()
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 0
    assert (
        fake.objects[("deployment", "tributo-knova-v25-consumer")]["spec"]["replicas"]
        == 1
    )


@pytest.mark.parametrize("config_state", ["deleted", "invalid", "omitted"])
def test_rollback_does_not_read_original_config(arguments, monkeypatch, config_state):
    fake = FakeKubectl()
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    sys.argv[1] = "rollback"
    _, path = arguments
    if config_state == "invalid":
        path.write_text("not JSON")
    else:
        path.unlink()
    if config_state == "omitted":
        index = sys.argv.index("--config")
        del sys.argv[index : index + 2]
    deploy.main()
    assert (
        fake.objects[("deployment", "tributo-knova-v25-consumer")]["spec"]["replicas"]
        == 0
    )


@pytest.mark.parametrize("changed", ["owner", "digest", "consumer_uid", "previous_uid"])
def test_rollback_rejects_replaced_or_changed_targets(arguments, monkeypatch, changed):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.objects[("deployment", "old")] = {
        "metadata": {"uid": "old-uid"},
        "spec": {"replicas": 2},
    }
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    metadata = fake.objects[("deployment", "tributo-knova-v25-consumer")]["metadata"]
    if changed == "owner":
        metadata["labels"]["app.kubernetes.io/managed-by"] = "another-tool"
    elif changed == "digest":
        metadata["annotations"]["knova-deploy/spec-sha256"] = "different-release"
    elif changed == "consumer_uid":
        metadata["uid"] = "replacement-uid"
    else:
        fake.objects[("deployment", "old")]["metadata"]["uid"] = "replacement-uid"
    count = len(fake.calls)
    sys.argv[1] = "rollback"
    with pytest.raises(ValueError, match="differs from saved switch"):
        deploy.main()
    assert not any(call[0] == "scale" for call in fake.calls[count:])


def test_unfinished_switch_requires_explicit_rollback(arguments, monkeypatch):
    fake = FakeKubectl()
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    journal_path = deploy.parse_args().output / "switch.json"
    journal = json.loads(journal_path.read_text())
    journal["state"] = "switching"
    journal_path.write_text(json.dumps(journal))
    count = len(fake.calls)
    with pytest.raises(ValueError, match="unfinished switch"):
        deploy.main()
    assert not any(
        call[0] in {"apply", "create", "scale", "delete"}
        and "--dry-run=server" not in call
        for call in fake.calls[count:]
    )
    sys.argv[1] = "rollback"
    deploy.main()
    assert json.loads(journal_path.read_text())["state"] == "rolled_back"


def test_failed_recovery_requires_completed_rollback_before_retry(
    arguments, monkeypatch
):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.fail_consumer = True
    fake.objects[("deployment", "old")] = {
        "metadata": {"uid": "old-uid"},
        "spec": {"replicas": 2},
    }
    fail_restore = True

    def kubectl(argv, **kwargs):
        if fail_restore and argv[6:] == ["scale", "deployment", "old", "--replicas=2"]:
            raise RuntimeError("old Consumer restore failed")
        return fake(argv, **kwargs)

    monkeypatch.setattr(deploy, "run", kubectl)
    with pytest.raises(RuntimeError, match="old Consumer restore failed"):
        deploy.main()
    journal_path = deploy.parse_args().output / "switch.json"
    assert json.loads(journal_path.read_text())["state"] == "switching"
    with pytest.raises(ValueError, match="unfinished switch"):
        deploy.main()
    fail_restore = False
    sys.argv[1] = "rollback"
    deploy.main()
    assert fake.objects[("deployment", "old")]["spec"]["replicas"] == 2
    assert json.loads(journal_path.read_text())["state"] == "rolled_back"
    fake.fail_consumer = False
    sys.argv[1] = "apply"
    deploy.main()
    assert json.loads(journal_path.read_text())["state"] == "active"


def test_legacy_journal_rollback_uses_saved_manifest(arguments, monkeypatch):
    fake = FakeKubectl()
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    journal_path = deploy.parse_args().output / "switch.json"
    journal = json.loads(journal_path.read_text())
    for key in ("state", "spec_sha256", "consumer_uid", "previous_uid"):
        journal.pop(key, None)
    journal_path.write_text(json.dumps(journal))
    arguments[1].unlink()
    sys.argv[1] = "rollback"
    deploy.main()
    updated = json.loads(journal_path.read_text())
    assert updated["state"] == "rolled_back"
    assert updated["spec_sha256"]


def test_legacy_active_release_keeps_idempotence(arguments, monkeypatch):
    sys.argv += ["--previous-consumer", "old"]
    fake = FakeKubectl()
    fake.objects[("deployment", "old")] = {
        "metadata": {"uid": "old-uid"},
        "spec": {"replicas": 2},
    }
    monkeypatch.setattr(deploy, "run", fake)
    deploy.main()
    journal_path = deploy.parse_args().output / "switch.json"
    journal = json.loads(journal_path.read_text())
    for key in ("state", "spec_sha256", "consumer_uid", "previous_uid"):
        journal.pop(key, None)
    journal_path.write_text(json.dumps(journal))
    count = len(fake.calls)
    deploy.main()
    assert not any(
        call[0] in {"apply", "create", "scale", "delete"}
        and "--dry-run=server" not in call
        for call in fake.calls[count:]
    )
    saved = json.loads(journal_path.read_text())
    assert saved["state"] == "active"
    assert saved["previous_replicas"] == 2
    assert saved["previous_uid"] == "old-uid"


def test_settings_resolves_paths_and_allows_cli_override(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "context": "ctx",
                "namespace": "ns",
                "config": "config.json",
                "output": "out",
                "workers": 2,
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["deploy.py", "render", "--settings", str(path), "--workers", "3"]
    )
    args = deploy.parse_args()
    assert args.context == "ctx"
    assert args.config == tmp_path / "config.json"
    assert args.output == tmp_path / "out"
    assert args.workers == 3


def test_unknown_settings_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"unknown_option": "x"}))
    monkeypatch.setattr(sys, "argv", ["deploy.py", "render", "--settings", str(path)])
    with pytest.raises(ValueError, match="unknown deployment setting"):
        deploy.parse_args()
