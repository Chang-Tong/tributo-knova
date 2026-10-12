#!/usr/bin/env python3
"""Deploy an immutable KnoVa release using kubectl and standard-library Python.

Creates a dedicated RayCluster; never patches an existing shared Ray cluster.
Secrets must already exist. No secret contents are written to the release folder.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

OWNER = "tributo-knova-deploy"


def run(argv, *, data=None, timeout=180):
    result = subprocess.run(
        argv, input=data, text=True, capture_output=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()}")
    return result.stdout


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", type=Path)
    initial, _ = pre.parse_known_args()
    settings = {}
    if initial.settings:
        settings = json.loads(initial.settings.read_text())
        if not isinstance(settings, dict):
            raise ValueError("settings must be a JSON object")
        for field in ("config", "output"):
            if field in settings:
                path = Path(settings[field])
                settings[field] = (
                    path if path.is_absolute() else initial.settings.parent / path
                )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["render", "check", "apply", "status", "rollback"])
    p.add_argument("--settings", type=Path)
    p.add_argument("--context", required="context" not in settings)
    p.add_argument("--namespace", required="namespace" not in settings)
    p.add_argument("--name", default="tributo-knova-v25")
    p.add_argument("--config", type=Path)
    p.add_argument("--secret", default="tributo-knova-env")
    p.add_argument("--image", default="knova-tributo-devbox:v2.5")
    p.add_argument("--architecture", choices=["arm64", "amd64"], default="arm64")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cpus", type=int, default=2, help="CPUs per worker")
    p.add_argument("--memory", default="8Gi", help="Memory per Ray pod")
    p.add_argument("--image-pull-secret")
    p.add_argument(
        "--shared-pvc",
        help="Existing ReadWriteMany PVC for shared Ray checkpoints/NFS artifacts",
    )
    p.add_argument("--shared-path", default="/shared/tributo")
    p.add_argument("--previous-config", default="/etc/tributo-knova/config.json")
    p.add_argument(
        "--s3-bucket", default="", help="Enable Head and Worker S3 read/write check"
    )
    p.add_argument(
        "--previous-consumer", help="Pause this Deployment only after preflight passes"
    )
    p.add_argument("--output", type=Path, default=Path("knova-release"))
    allowed = {action.dest for action in p._actions} - {"action", "settings", "help"}
    if set(settings) - allowed:
        raise ValueError(
            "unknown deployment setting: " + ", ".join(set(settings) - allowed)
        )
    p.set_defaults(**settings)
    args = p.parse_args()
    if args.action != "rollback" and args.config is None:
        p.error("--config is required for this action")
    return args


def build(a):
    for value in (a.name, a.namespace, a.secret):
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", value) or len(value) > 50:
            raise ValueError(
                "name, namespace and secret must be DNS labels of at most 50 characters"
            )
    if a.workers < 1 or a.cpus < 1:
        raise ValueError("at least one worker and one CPU per worker are required")
    if a.image.endswith(":latest") or (
        ":" not in a.image.rsplit("/", 1)[-1] and "@sha256:" not in a.image
    ):
        raise ValueError("use a versioned image tag or digest")
    if not a.shared_path.startswith("/") or a.shared_path in {
        "/",
        "/opt",
        "/deploy",
        "/dev/shm",
    }:
        raise ValueError("shared-path must be an isolated absolute directory")
    config = json.loads(a.config.read_text())
    env = config["execution"].get("env_vars", {})
    for key in env:
        if any(
            word in key.upper()
            for word in ("PASSWORD", "SECRET", "ACCESS_KEY", "TOKEN")
        ):
            raise ValueError(f"execution.env_vars must not contain credential: {key}")
    transport = config["transport"]
    for key in ("url", "driver_url"):
        if "@" in str(transport.get(key, "")):
            raise ValueError(
                "Redis URLs must use password_ref instead of embedded credentials"
            )
    for key in ("password", "sentinel_password"):
        if key in transport:
            raise ValueError(f"transport.{key} must be a Secret reference")
    required = []
    for key in ("password_ref", "sentinel_password_ref"):
        if transport.get(key):
            if not transport[key].startswith("env:"):
                raise ValueError(
                    "this deployment helper supports env: Secret references"
                )
            required.append(transport[key][4:])
    if a.s3_bucket:
        required += ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    config["execution"]["ray_dashboard_url"] = f"http://{a.name}-ray-head-svc:8265"
    labels = {"app.kubernetes.io/managed-by": OWNER, "knova-release": a.name}

    def obj(kind, name, spec, api="v1"):
        return {
            "apiVersion": api,
            "kind": kind,
            "metadata": {
                "name": name,
                "namespace": a.namespace,
                "labels": labels.copy(),
            },
            **spec,
        }

    def pod(container, *, ray_node=False):
        value = {
            "nodeSelector": {"kubernetes.io/arch": a.architecture},
            "securityContext": {
                "fsGroup": 10001,
                "fsGroupChangePolicy": "OnRootMismatch",
            },
            "containers": [container],
        }
        if a.image_pull_secret:
            value["imagePullSecrets"] = [{"name": a.image_pull_secret}]
        if ray_node:
            value["volumes"] = [
                {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}
            ]
            container["volumeMounts"] = [{"name": "dshm", "mountPath": "/dev/shm"}]
        if a.shared_pvc:
            value.setdefault("volumes", []).append(
                {"name": "shared", "persistentVolumeClaim": {"claimName": a.shared_pvc}}
            )
            container.setdefault("volumeMounts", []).append(
                {"name": "shared", "mountPath": a.shared_path}
            )
        return {"metadata": {"labels": labels.copy()}, "spec": value}

    def container(name):
        return {
            "name": name,
            "image": a.image,
            "imagePullPolicy": "IfNotPresent",
            "envFrom": [{"secretRef": {"name": a.secret}}],
            "env": [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "RAY_USAGE_STATS_ENABLED", "value": "0"},
            ],
        }

    head = container("ray-head")
    head["ports"] = [
        {"containerPort": n, "name": name}
        for n, name in [(6379, "gcs"), (8265, "dashboard"), (10001, "client")]
    ]
    head["resources"] = {
        "requests": {"cpu": "1", "memory": a.memory},
        "limits": {"cpu": "2", "memory": a.memory},
    }
    worker = container("ray-worker")
    worker["resources"] = {
        "requests": {"cpu": str(a.cpus), "memory": a.memory},
        "limits": {"cpu": str(a.cpus), "memory": a.memory},
    }
    cluster = obj(
        "RayCluster",
        a.name + "-ray",
        {
            "spec": {
                "rayVersion": "2.55.1",
                "headGroupSpec": {
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "runtime-env-agent-port": "52366",
                        "num-cpus": "0",
                        "object-store-memory": "536870912",
                    },
                    "template": pod(head, ray_node=True),
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "replicas": a.workers,
                        "minReplicas": a.workers,
                        "maxReplicas": a.workers,
                        "rayStartParams": {
                            "num-cpus": str(a.cpus),
                            "object-store-memory": "536870912",
                        },
                        "template": pod(worker, ray_node=True),
                    }
                ],
            }
        },
        "ray.io/v1",
    )
    configmap = obj(
        "ConfigMap",
        a.name + "-config",
        {
            "data": {
                "config.json": json.dumps(config, indent=2),
                "check_cluster.py": Path(__file__)
                .with_name("check_cluster.py")
                .read_text(),
            }
        },
    )
    consumer = container("consumer")
    consumer["command"] = ["tributo-knova"]
    consumer["args"] = ["consume", "--config", "/deploy/config.json"]
    consumer["resources"] = {
        "requests": {"cpu": "250m", "memory": "512Mi"},
        "limits": {"cpu": "2", "memory": "2Gi"},
    }
    consumer["readinessProbe"] = {
        "exec": {
            "command": ["tributo-knova", "health", "--config", "/deploy/config.json"]
        },
        "periodSeconds": 20,
        "timeoutSeconds": 15,
        "failureThreshold": 3,
    }
    consumer["volumeMounts"] = [
        {"name": "config", "mountPath": "/deploy", "readOnly": True}
    ]
    template = pod(consumer)
    template["spec"].setdefault("volumes", []).extend(
        [{"name": "config", "configMap": {"name": a.name + "-config"}}]
    )
    deployment = obj(
        "Deployment",
        a.name + "-consumer",
        {
            "spec": {
                "replicas": 0,
                "selector": {
                    "matchLabels": {"knova-release": a.name, "component": "consumer"}
                },
                "template": template,
            }
        },
        "apps/v1",
    )
    template["metadata"]["labels"]["component"] = "consumer"
    check = container("check")
    check["command"] = ["python", "/deploy/check_cluster.py"]
    check["env"] += [
        {"name": "RAY_ADDRESS", "value": f"{a.name}-ray-head-svc:6379"},
        {"name": "EXPECTED_NODES", "value": str(a.workers + 1)},
        {"name": "S3_CHECK_BUCKET", "value": a.s3_bucket},
        {"name": "SHARED_CHECK_PATH", "value": a.shared_path if a.shared_pvc else ""},
    ]
    check["volumeMounts"] = [
        mount.copy() for mount in consumer["volumeMounts"] if mount["name"] != "shared"
    ]
    check_template = pod(check)
    check_template["spec"].update(
        restartPolicy="Never", volumes=template["spec"]["volumes"]
    )
    job = obj(
        "Job",
        a.name + "-check",
        {
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 240,
                "template": check_template,
            }
        },
        "batch/v1",
    )
    items = [configmap, cluster, deployment]
    digest = hashlib.sha256(
        json.dumps([*items, job], sort_keys=True).encode()
    ).hexdigest()
    for value in [*items, job]:
        value["metadata"]["annotations"] = {"knova-deploy/spec-sha256": digest}
    return items, job, required, digest


def main():
    a = parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    base = [
        "kubectl",
        "--context",
        a.context,
        "--namespace",
        a.namespace,
        "--request-timeout=30s",
    ]

    def k(*args, data=None, timeout=180):
        return run(base + list(args), data=data, timeout=timeout)

    def get(kind, name):
        raw = k("get", kind, name, "--ignore-not-found", "-o", "json")
        return json.loads(raw) if raw.strip() else None

    journal_path = a.output / "switch.json"

    def read_journal():
        journal = json.loads(journal_path.read_text())
        if (journal["context"], journal["namespace"], journal["name"]) != (
            a.context,
            a.namespace,
            a.name,
        ):
            raise ValueError("rollback context does not match saved switch")
        if not journal.get("spec_sha256"):
            # Releases created before switch states were introduced saved the
            # same immutable digest in their local manifest.
            saved = json.loads((a.output / "manifest.json").read_text())
            matching = [
                item
                for item in saved["items"]
                if item["kind"] == "Deployment"
                and item["metadata"]["name"] == a.name + "-consumer"
                and item["metadata"]["namespace"] == a.namespace
            ]
            if len(matching) != 1:
                raise ValueError("saved manifest does not identify this Consumer")
            journal["spec_sha256"] = matching[0]["metadata"]["annotations"][
                "knova-deploy/spec-sha256"
            ]
        return journal

    def save_journal(journal):
        temporary = journal_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(journal, indent=2))
        temporary.replace(journal_path)

    def validate_switch_targets(journal):
        current = get("deployment", a.name + "-consumer")
        metadata = current.get("metadata", {}) if current else {}
        if (
            metadata.get("labels", {}).get("app.kubernetes.io/managed-by") != OWNER
            or metadata.get("annotations", {}).get("knova-deploy/spec-sha256")
            != journal["spec_sha256"]
            or (
                journal.get("consumer_uid")
                and metadata.get("uid") != journal["consumer_uid"]
            )
        ):
            raise ValueError("Consumer ownership/version differs from saved switch")
        if journal["previous"]:
            previous_target = get("deployment", journal["previous"])
            if (
                not previous_target
                or journal["previous"] == a.name + "-consumer"
                or (
                    journal.get("previous_uid")
                    and previous_target.get("metadata", {}).get("uid")
                    != journal["previous_uid"]
                )
            ):
                raise ValueError("previous Consumer differs from saved switch")

    def restore_switch(journal):
        validate_switch_targets(journal)
        k("scale", "deployment", a.name + "-consumer", "--replicas=0")
        k("rollout", "status", "deployment/" + a.name + "-consumer", "--timeout=120s")
        if journal["previous"]:
            k(
                "scale",
                "deployment",
                journal["previous"],
                f"--replicas={journal['previous_replicas']}",
            )
            k(
                "rollout",
                "status",
                "deployment/" + journal["previous"],
                "--timeout=120s",
            )
        journal["state"] = "rolled_back"
        save_journal(journal)

    if a.action == "rollback":
        restore_switch(read_journal())
        print("Consumer switch rolled back. Ray resources retained for diagnosis.")
        return
    items, job, required, digest = build(a)
    journal = read_journal() if journal_path.exists() else None
    if journal and (
        journal["spec_sha256"] != digest or journal["previous"] != a.previous_consumer
    ):
        raise ValueError("release configuration differs from saved switch")
    manifest = {"apiVersion": "v1", "kind": "List", "items": items}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if a.action == "render":
        print(a.output / "manifest.json")
        return
    if a.action == "status":
        print(
            k(
                "get",
                "pods,rayclusters,deployments,jobs",
                "-l",
                f"knova-release={a.name}",
                "-o",
                "wide",
            )
        )
        return
    k("get", "namespace", a.namespace)
    k("get", "crd", "rayclusters.ray.io")
    nodes = json.loads(
        k("get", "nodes", "-l", f"kubernetes.io/arch={a.architecture}", "-o", "json")
    )["items"]
    if not any(
        not node.get("spec", {}).get("unschedulable")
        and any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in node.get("status", {}).get("conditions", [])
        )
        for node in nodes
    ):
        raise ValueError("no Ready nodes with requested architecture")
    if a.shared_pvc:
        pvc = get("pvc", a.shared_pvc)
        if (
            not pvc
            or pvc.get("status", {}).get("phase") != "Bound"
            or "ReadWriteMany" not in pvc["spec"].get("accessModes", [])
        ):
            raise ValueError("shared PVC must be Bound and ReadWriteMany")
    secret = get("secret", a.secret)
    if not secret:
        raise ValueError(f"Secret not found: {a.secret}")
    missing = [key for key in required if not secret.get("data", {}).get(key)]
    if missing:
        raise ValueError("Secret missing required keys: " + ", ".join(missing))
    previous = get("deployment", a.previous_consumer) if a.previous_consumer else None
    if a.previous_consumer and (
        not previous or a.previous_consumer == a.name + "-consumer"
    ):
        raise ValueError("previous Consumer must be a distinct existing Deployment")
    existing = []
    for item in [*items, job]:
        old = get(item["kind"], item["metadata"]["name"])
        if old:
            if (
                old["metadata"].get("labels", {}).get("app.kubernetes.io/managed-by")
                != OWNER
            ):
                raise ValueError(
                    "refusing to overwrite unmanaged resource: "
                    + item["metadata"]["name"]
                )
            if (
                old["metadata"].get("annotations", {}).get("knova-deploy/spec-sha256")
                != digest
            ):
                raise ValueError(
                    "immutable release differs; use a new --name "
                    "for this configuration/image"
                )
            existing.append(old)
    # Validate against the actual API/CRD before mutation.
    k("apply", "--dry-run=server", "-f", "-", data=json.dumps(manifest))
    print(
        "Preflight configuration passed; required Secret key names: "
        + ", ".join(required)
    )
    if a.action == "check":
        return
    current = get("deployment", a.name + "-consumer")
    legacy_active = (
        journal is not None
        and "state" not in journal
        and current is not None
        and current["spec"].get("replicas", 0) > 0
        and (not previous or previous["spec"].get("replicas", 1) == 0)
    )
    if (
        journal
        and journal.get("state") not in {"active", "rolled_back"}
        and not legacy_active
    ):
        raise ValueError("unfinished switch; run rollback before retrying apply")
    if journal:
        validate_switch_targets(journal)
    if current and current["spec"].get("replicas", 0) > 0:
        if journal and journal.get("state") == "rolled_back":
            raise ValueError("Consumer restarted after rollback; run rollback again")
        k("rollout", "status", "deployment/" + a.name + "-consumer", "--timeout=120s")
        if legacy_active:
            journal["state"] = "active"
            journal["consumer_uid"] = current["metadata"].get("uid")
            journal["previous_uid"] = (
                previous.get("metadata", {}).get("uid") if previous else None
            )
            save_journal(journal)
        print("Same release already active; no resources changed.")
        return
    if journal and journal.get("state") == "active":
        raise ValueError(
            "saved release is inactive; run rollback before retrying apply"
        )
    try:
        print(k("apply", "-f", "-", data=json.dumps(manifest)))
        k(
            "wait",
            "raycluster/" + a.name + "-ray",
            "--for=condition=RayClusterProvisioned",
            "--timeout=300s",
            timeout=330,
        )
        k(
            "wait",
            "pod",
            "-l",
            f"ray.io/cluster={a.name}-ray",
            "--for=condition=Ready",
            "--timeout=300s",
            timeout=330,
        )
        k("delete", "job", a.name + "-check", "--ignore-not-found", "--wait=true")
        k("create", "-f", "-", data=json.dumps(job))
        deadline = time.monotonic() + 250
        while time.monotonic() < deadline:
            state = get("job", a.name + "-check")
            if not state:
                raise RuntimeError("preflight Job disappeared")
            conditions = {
                c["type"]: c["status"]
                for c in state.get("status", {}).get("conditions", [])
            }
            if conditions.get("Failed") == "True" or state.get("status", {}).get(
                "failed", 0
            ):
                raise RuntimeError("preflight Job failed; inspect preflight.log")
            if conditions.get("Complete") == "True":
                break
            time.sleep(1)
        else:
            raise TimeoutError("preflight Job did not complete within 250 seconds")
        log = k("logs", "job/" + a.name + "-check")
        (a.output / "preflight.log").write_text(log)
        print(log)
        if previous and previous["spec"].get("replicas", 1):
            k(
                "rollout",
                "status",
                "deployment/" + a.previous_consumer,
                "--timeout=120s",
            )
            probe = (
                "import json,sys; from pathlib import Path; "
                "from ray.job_submission import JobSubmissionClient; "
                "c=json.loads(Path(sys.argv[1]).read_text()); "
                "client=JobSubmissionClient(c['execution']['ray_dashboard_url']); "
                "jobs=client.list_jobs(); "
                "active=[j.submission_id for j in jobs "
                "if str(j.status) in ('RUNNING','PENDING')]; "
                "print(json.dumps(active))"
            )
            active = json.loads(
                k(
                    "exec",
                    "deployment/" + a.previous_consumer,
                    "--",
                    "python",
                    "-c",
                    probe,
                    a.previous_config,
                )
            )
            if active:
                raise ValueError(
                    "old Ray cluster has active jobs; "
                    "drain it before switching Consumer"
                )
        if journal is None:
            current = get("deployment", a.name + "-consumer")
            journal = {
                "context": a.context,
                "namespace": a.namespace,
                "name": a.name,
                "spec_sha256": digest,
                "consumer_uid": current["metadata"].get("uid"),
                "previous": a.previous_consumer,
                "previous_uid": previous.get("metadata", {}).get("uid")
                if previous
                else None,
                "previous_replicas": previous["spec"].get("replicas", 1)
                if previous
                else 0,
            }
        validate_switch_targets(journal)
        journal["state"] = "switching"
        save_journal(journal)
        try:
            if previous:
                k("scale", "deployment", a.previous_consumer, "--replicas=0")
                k(
                    "rollout",
                    "status",
                    "deployment/" + a.previous_consumer,
                    "--timeout=120s",
                )
            k("scale", "deployment", a.name + "-consumer", "--replicas=1")
            k(
                "rollout",
                "status",
                "deployment/" + a.name + "-consumer",
                "--timeout=180s",
                timeout=210,
            )
        except Exception:
            restore_switch(journal)
            raise
        journal["state"] = "active"
        save_journal(journal)
        print(
            "Deployment ready. Submit a real training task "
            "as the final business acceptance."
        )
    except Exception:
        for filename, command in [
            ("pods.txt", ("get", "pods", "-o", "wide")),
            ("events.txt", ("get", "events", "--sort-by=.lastTimestamp")),
            ("preflight.log", ("logs", "job/" + a.name + "-check")),
        ]:
            with suppress(Exception):
                (a.output / filename).write_text(k(*command))
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DEPLOY FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
