#!/usr/bin/env python3
"""Local Kind fixture for validating consumer-only deployment against existing services.

Never runs against a general kubeconfig/context. This script creates disposable
test infrastructure; it is NOT part of the production deployment package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

NAMESPACE = "knova-existing"
CONTEXT = "kind-knova-existing-accept"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/extended-acceptance-20260925/kubernetes"
KUBECONFIG = OUT / "kubeconfig"
RAY_IMAGE = "knova-existing-ray-fixture:20260925"
REDIS_IMAGE = "redis:8-bookworm"
CK_IMAGE = "artifact.nc.rdcloud.4c.hq.cmcc:80/knovapriv-libs/library/clickhouse:26.3.17"
S3_IMAGE = (
    "artifact.nc.rdcloud.4c.hq.cmcc:80/knovapriv-libs/rustfs/rustfs:1.0.0-beta.8-glibc"
)


def obj(kind, name, **body):
    api = "apps/v1" if kind == "Deployment" else "v1"
    return {
        "apiVersion": api,
        "kind": kind,
        "metadata": {"name": name, "namespace": NAMESPACE},
        **body,
    }


def service(name, ports):
    return obj(
        "Service",
        name,
        spec={
            "selector": {"fixture": name},
            "ports": [
                {"name": f"p{port}", "port": port, "targetPort": port} for port in ports
            ],
        },
    )


def deployment(
    name,
    image,
    command,
    *,
    env=None,
    ports=(),
    replicas=1,
    memory="256Mi",
    cpu="250m",
    ray=False,
):
    container = {
        "name": name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "envFrom": [{"secretRef": {"name": "fixture-env"}}],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": cpu, "memory": memory},
        },
    }
    if env:
        container["env"] = [{"name": key, "value": value} for key, value in env.items()]
    if ports:
        container["ports"] = [{"containerPort": port} for port in ports]
    spec = {"containers": [container], "nodeSelector": {"kubernetes.io/arch": "arm64"}}
    if ray:
        container["resources"]["requests"] = {"cpu": "500m", "memory": "1Gi"}
        container["volumeMounts"] = [{"name": "dshm", "mountPath": "/dev/shm"}]
        spec["volumes"] = [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}
        ]
    if name == "ray-head":
        container["readinessProbe"] = {
            "httpGet": {"path": "/api/version", "port": 8265},
            "periodSeconds": 3,
            "failureThreshold": 60,
        }
    if name == "clickhouse":
        container["readinessProbe"] = {
            "httpGet": {"path": "/ping", "port": 8123},
            "periodSeconds": 3,
            "failureThreshold": 60,
        }
    return obj(
        "Deployment",
        name,
        spec={
            "replicas": replicas,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"fixture": name}},
            "template": {"metadata": {"labels": {"fixture": name}}, "spec": spec},
        },
    )


def manifest():
    # Public, disposable fixture credentials; never use production credentials here.
    secret = obj(
        "Secret",
        "fixture-env",
        stringData={
            "KNOVA_REDIS_PASSWORD": "local-repro-only",
            "KNOVA_REDIS_SENTINEL_PASSWORD": "local-sentinel-only",
            "KNOVA_CLICKHOUSE_PASSWORD": "local-test-only",
            "KNOVA_OUTPUT_CLICKHOUSE_PASSWORD": "local-test-only",
            "TRIBUTO_CLICKHOUSE_PASSWORD": "local-test-only",
            "CLICKHOUSE_PASSWORD": "local-test-only",
            "AWS_ACCESS_KEY_ID": "localtestkey",
            "AWS_SECRET_ACCESS_KEY": "local-test-secret",
            "AWS_ENDPOINT_URL": "http://rustfs:9000",
            "AWS_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
        },
    )
    items = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NAMESPACE}},
        secret,
    ]
    items += [
        service("redis-master", [6379]),
        service("redis-replica", [6379]),
        service("redis-sentinel", [26379]),
        service("clickhouse", [8123, 9000]),
        service("rustfs", [9000]),
        service("ray-head", [8265, 6379]),
    ]
    items.append(
        deployment(
            "redis-master",
            REDIS_IMAGE,
            ["sh", "-ec", 'exec redis-server --requirepass "$KNOVA_REDIS_PASSWORD"'],
        )
    )
    items.append(
        deployment(
            "redis-replica",
            REDIS_IMAGE,
            [
                "sh",
                "-ec",
                "exec redis-server --replicaof redis-master 6379 "
                '--requirepass "$KNOVA_REDIS_PASSWORD" '
                '--masterauth "$KNOVA_REDIS_PASSWORD"',
            ],
        )
    )
    sentinel = (
        'printf "%s\\n" "port 26379" "sentinel resolve-hostnames yes" '
        '"sentinel announce-hostnames yes" '
        '"sentinel monitor mymaster redis-master 6379 2" '
        '"sentinel auth-pass mymaster $KNOVA_REDIS_PASSWORD" '
        '"requirepass $KNOVA_REDIS_SENTINEL_PASSWORD" '
        '"sentinel down-after-milliseconds mymaster 5000" '
        '"sentinel failover-timeout mymaster 15000" > /tmp/sentinel.conf; '
        "exec redis-server /tmp/sentinel.conf --sentinel"
    )
    items.append(
        deployment("redis-sentinel", REDIS_IMAGE, ["sh", "-ec", sentinel], replicas=3)
    )
    items.append(
        deployment(
            "clickhouse",
            CK_IMAGE,
            ["/entrypoint.sh"],
            env={
                "CLICKHOUSE_USER": "knova",
                "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT": "1",
            },
            ports=[8123, 9000],
            memory="2Gi",
            cpu="2",
        )
    )
    items.append(
        deployment(
            "rustfs",
            S3_IMAGE,
            ["/entrypoint.sh"],
            env={
                "RUSTFS_ACCESS_KEY": "localtestkey",
                "RUSTFS_SECRET_KEY": "local-test-secret",
                "RUSTFS_VOLUMES": "/data",
            },
            ports=[9000],
            memory="1Gi",
            cpu="1",
        )
    )
    items.append(
        deployment(
            "ray-head",
            RAY_IMAGE,
            [
                "ray",
                "start",
                "--head",
                "--dashboard-host=0.0.0.0",
                "--runtime-env-agent-port=52366",
                "--num-cpus=0",
                "--object-store-memory=268435456",
                "--disable-usage-stats",
                "--block",
            ],
            ports=[8265, 6379],
            memory="6Gi",
            cpu="2",
            ray=True,
        )
    )
    items.append(
        deployment(
            "ray-worker",
            RAY_IMAGE,
            [
                "sh",
                "-ec",
                'until python -c "import urllib.request; '
                "urllib.request.urlopen('http://ray-head:8265/api/version', "
                'timeout=2)"; do sleep 2; done; '
                "exec ray start --address=ray-head:6379 --num-cpus=4 "
                "--object-store-memory=536870912 --disable-usage-stats --block",
            ],
            memory="6Gi",
            cpu="4",
            ray=True,
        )
    )
    return {"apiVersion": "v1", "kind": "List", "items": items}


def kubectl(*args, **kwargs):
    # Explicit private kubeconfig and hard-coded local Kind context.
    command = [
        "kubectl",
        "--kubeconfig",
        str(KUBECONFIG),
        "--context",
        CONTEXT,
        "--namespace",
        NAMESPACE,
        *args,
    ]
    return subprocess.run(command, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "up", "snapshot", "down"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action in {"render", "up"}:
        target = OUT / "fixtures.json"
        target.write_text(json.dumps(manifest(), indent=2) + "\n")
        print(target)
        if args.action == "up":
            kubectl("apply", "-f", str(target))
    elif args.action == "snapshot":
        result = kubectl(
            "get",
            "deployments,pods,services,configmaps",
            "-o",
            "json",
            capture_output=True,
        )
        (OUT / "fixture-snapshot.json").write_text(result.stdout)
    else:
        kubectl("delete", "namespace", NAMESPACE, "--wait=true", "--timeout=180s")


if __name__ == "__main__":
    main()
