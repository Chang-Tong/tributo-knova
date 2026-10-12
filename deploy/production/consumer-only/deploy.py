#!/usr/bin/env python3
"""Existing-cluster Consumer deployment. Requires Python 3.10+ and kubectl only."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
LABEL = "tributo.knova/managed-by"
OWNER = "consumer-only"


class Error(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def report(stage, status="PASS", **details):
    print(
        json.dumps({"stage": stage, "status": status, **details}, ensure_ascii=False),
        flush=True,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


def load_settings(path):
    settings = json.loads(path.read_text())
    allowed = {
        "context",
        "namespace",
        "name",
        "image",
        "architecture",
        "secret",
        "config",
        "probe",
        "old_deployment",
        "timeout_seconds",
        "image_pull_policy",
        "node_selector",
        "image_pull_secrets",
        "dns_policy",
        "dns_config",
        "resources",
        "output",
        "tolerations",
        "service_account",
        "volumes",
        "volume_mounts",
    }
    if set(settings) - allowed:
        raise Error(
            "E_SETTINGS", "未知配置字段: " + ", ".join(sorted(set(settings) - allowed))
        )
    for key in (
        "context",
        "namespace",
        "name",
        "image",
        "secret",
        "config",
        "probe",
        "output",
    ):
        if (
            not isinstance(settings.get(key), str)
            or not settings[key].strip()
            or "YOUR_" in settings[key]
        ):
            raise Error("E_SETTINGS", "需要填写 settings." + key)
    for key in ("namespace", "name", "secret", "old_deployment"):
        if settings.get(key) and not re.fullmatch(
            r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", settings[key]
        ):
            raise Error("E_SETTINGS", "非法 Kubernetes 名称: " + key)
    if len(settings["name"]) > 40:
        raise Error("E_SETTINGS", "name 最多 40 字符，为哈希后缀预留空间")
    if settings.get("old_deployment") == settings["name"]:
        raise Error(
            "E_SETTINGS", "原地更新省略 old_deployment；它只能指向另一个 Deployment"
        )
    if settings.get("architecture", "arm64") not in {"arm64", "amd64"}:
        raise Error("E_SETTINGS", "architecture 仅支持 arm64/amd64")
    for key in ("config", "probe", "output"):
        settings[key] = str((path.parent / settings[key]).resolve())
    config = json.loads(Path(settings["config"]).read_text())
    probe = json.loads(Path(settings["probe"]).read_text())
    validate_config(config)
    spec = importlib.util.spec_from_file_location("consumer_doctor", HERE / "doctor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate_probe(probe)
    except Exception:
        raise Error(
            "E_SETTINGS", "probe.json 校验失败；检查字段、算法、凭据引用和拓扑约束"
        ) from None
    if int(settings.get("timeout_seconds", 360)) < 30:
        raise Error("E_SETTINGS", "timeout_seconds 不能小于 30")
    return settings, config, probe


def validate_config(config):
    transport, execution = config.get("transport", {}), config.get("execution", {})
    if (
        transport.get("mode") != "sentinel"
        or not transport.get("sentinel_nodes")
        or not transport.get("sentinel_master")
    ):
        raise Error("E_CONFIG", "本部署器要求 Redis Sentinel 地址及 master 名称")
    if not isinstance(config.get("channels"), dict) or set(config["channels"]) != {
        "training",
        "batch_inference",
    }:
        raise Error("E_CONFIG", "channels 必须包含 training 和 batch_inference")
    for value in config["channels"].values():
        if not all(
            value.get(key)
            for key in ("task_stream_key", "consumer_group", "consumer_name")
        ):
            raise Error("E_CONFIG", "channel 缺少 stream/group/consumer_name")
    env = execution.get("env_vars", {})
    if any(name in env for name in ("PATH", "PYTHONPATH")):
        raise Error("E_CONFIG", "不要覆盖 Ray runtime_env PATH/PYTHONPATH")
    for name in env:
        if re.search(r"PASSWORD|SECRET|TOKEN|ACCESS_KEY|CREDENTIAL", name, re.I):
            raise Error("E_CONFIG", "敏感值不能进入 execution.env_vars: " + name)
    if not execution.get("project_root") or not execution.get("extra_py_modules"):
        raise Error("E_CONFIG", "需要 Consumer 内的 project_root 与 extra_py_modules")
    for node in transport["sentinel_nodes"] + (
        transport.get("driver_sentinel_nodes") or []
    ):
        parts = urlsplit("redis://" + node)
        if not parts.hostname or not parts.port or parts.username or parts.password:
            raise Error("E_CONFIG", "Sentinel 节点只能填写 host:port")
    for name in ("password_ref", "sentinel_password_ref"):
        ref = transport.get(name)
        if ref and not ref.startswith(("env:", "mount:")):
            raise Error("E_CONFIG", name + " 必须是 env:/mount: 引用")
    endpoint = urlsplit(execution.get("ray_dashboard_url", ""))
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.query
    ):
        raise Error("E_CONFIG", "ray_dashboard_url 必须是无内嵌凭据的 HTTP/HTTPS 地址")
    if "YOUR_" in json.dumps(config):
        raise Error("E_CONFIG", "config.json 仍包含 YOUR_ 占位符")

    # Catch common accidental plaintext credentials anywhere in config.
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    re.fullmatch(
                        r"password|secret|secret_key|access_key|token", key, re.I
                    )
                    and item
                ):
                    raise Error("E_CONFIG", "config.json 不得包含凭据正文: " + key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)


def manifests(settings, config, probe):
    data = {
        "config.json": json.dumps(config, indent=2),
        "probe.json": json.dumps(probe, indent=2),
        "doctor.py": (HERE / "doctor.py").read_text(),
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
    name, namespace = settings["name"], settings["namespace"]
    cm_name = name + "-" + digest
    labels = {LABEL: OWNER, "app.kubernetes.io/name": name}
    meta = {"name": cm_name, "namespace": namespace, "labels": labels}
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": meta,
        "immutable": True,
        "data": data,
    }
    container = {
        "name": "consumer",
        "image": settings["image"],
        "imagePullPolicy": settings.get("image_pull_policy", "IfNotPresent"),
        "command": ["tributo-knova"],
        "args": ["consume", "--config", "/etc/tributo-knova/config.json"],
        "envFrom": [{"secretRef": {"name": settings["secret"]}}],
        "env": [
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "RAY_USAGE_STATS_ENABLED", "value": "0"},
        ],
        "resources": settings.get(
            "resources",
            {
                "requests": {"cpu": "250m", "memory": "512Mi"},
                "limits": {"cpu": "2", "memory": "2Gi"},
            },
        ),
        "readinessProbe": {
            "exec": {
                "command": [
                    "tributo-knova",
                    "health",
                    "--config",
                    "/etc/tributo-knova/config.json",
                ]
            },
            "periodSeconds": 10,
            "timeoutSeconds": 15,
            "failureThreshold": 3,
        },
        "volumeMounts": [
            {"name": "config", "mountPath": "/etc/tributo-knova", "readOnly": True}
        ]
        + settings.get("volume_mounts", []),
    }
    node_selector = {
        "kubernetes.io/arch": settings.get("architecture", "arm64"),
        **settings.get("node_selector", {}),
    }
    pod = {
        "containers": [container],
        "terminationGracePeriodSeconds": 60,
        "nodeSelector": node_selector,
        "volumes": [{"name": "config", "configMap": {"name": cm_name}}]
        + settings.get("volumes", []),
    }
    for key, target in (
        ("dns_policy", "dnsPolicy"),
        ("dns_config", "dnsConfig"),
        ("tolerations", "tolerations"),
        ("service_account", "serviceAccountName"),
    ):
        if key in settings:
            pod[target] = settings[key]
    if settings.get("image_pull_secrets"):
        pod["imagePullSecrets"] = [
            {"name": name} for name in settings["image_pull_secrets"]
        ]
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "progressDeadlineSeconds": int(settings.get("timeout_seconds", 360)),
            "selector": {"matchLabels": {"app.kubernetes.io/name": name}},
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }
    return cm, deployment


class Kube:
    def __init__(self, settings):
        self.settings = settings
        self.prefix = [
            "kubectl",
            "--context",
            settings["context"],
            "--namespace",
            settings["namespace"],
        ]

    def run(self, *args, data=None, missing_ok=False):
        completed = subprocess.run(
            self.prefix + list(args),
            input=json.dumps(data) if data is not None else None,
            text=True,
            capture_output=True,
            timeout=int(self.settings.get("timeout_seconds", 360)) + 15,
        )
        if completed.returncode:
            if missing_ok and "NotFound" in completed.stderr:
                return None
            # Extract a bounded API reason, never the raw response/message.
            for reason in ("Forbidden", "Unauthorized", "Invalid", "AlreadyExists"):
                if "(" + reason + ")" in completed.stderr:
                    raise Error(
                        "E_KUBE_" + reason.upper(),
                        "kubectl " + " ".join(args[:2]) + " 被 API 拒绝: " + reason,
                    )
            # API response/exec errors can include env values. Do not echo raw stderr.
            raise Error(
                "E_KUBECTL",
                "kubectl "
                + " ".join(args[:2])
                + " 失败；检查 context/RBAC/namespace/对象状态，原始 stderr 未打印",
            )
        return completed.stdout

    def get(self, kind, name):
        result = self.run("get", kind, name, "-o", "json", missing_ok=True)
        return json.loads(result) if result else None

    def apply(self, document):
        return self.run("apply", "-f", "-", data=document)

    def delete(self, kind, name):
        return self.run(
            "delete",
            kind,
            name,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=90s",
        )

    def scale_down(self, name):
        if not self.get("deployment", name):
            return
        self.run("scale", "deployment", name, "--replicas=0")
        deadline = time.monotonic() + int(self.settings.get("timeout_seconds", 360))
        while time.monotonic() < deadline:
            deployment = self.get("deployment", name)
            selector = ",".join(
                key + "=" + value
                for key, value in deployment["spec"]["selector"]["matchLabels"].items()
            )
            pods = json.loads(self.run("get", "pods", "-l", selector, "-o", "json"))[
                "items"
            ]
            if not pods:
                return
            time.sleep(1)
        raise Error("E_DRAIN", "旧 Consumer Pod 未完全停止；不要启动新 Consumer")

    def rollout(self, name):
        self.run(
            "rollout",
            "status",
            "deployment/" + name,
            "--timeout=" + str(self.settings.get("timeout_seconds", 360)) + "s",
        )


def clean_document(obj):
    obj = copy.deepcopy(obj)
    obj.pop("status", None)
    for key in (
        "resourceVersion",
        "uid",
        "creationTimestamp",
        "generation",
        "managedFields",
    ):
        obj["metadata"].pop(key, None)
    obj["metadata"].get("annotations", {}).pop(
        "kubectl.kubernetes.io/last-applied-configuration", None
    )
    return obj


def diagnose_pod(pod):
    result = {"phase": pod.get("status", {}).get("phase", "Unknown")}
    conditions = pod.get("status", {}).get("conditions", [])
    result["conditions"] = [
        {"type": c["type"], "status": c["status"], "reason": c.get("reason")}
        for c in conditions
    ]
    result["containers"] = []
    for container in pod.get("status", {}).get("containerStatuses", []):
        state = container.get("state", {})
        result["containers"].append(
            {
                "name": container["name"],
                "restartCount": container.get("restartCount"),
                "waitingReason": state.get("waiting", {}).get("reason"),
                "exitCode": state.get("terminated", {}).get("exitCode"),
                "terminationReason": state.get("terminated", {}).get("reason"),
            }
        )
    return result


def check(kube, settings, cm, deployment, idle_only=False, keep=False):
    if kube.get("namespace", settings["namespace"]) is None:
        raise Error("E_NAMESPACE", "预先创建目标 namespace；本脚本不创建基础组件")
    if kube.get("secret", settings["secret"]) is None:
        raise Error(
            "E_SECRET",
            "目标 namespace 缺少 settings.secret；请由运维准备。脚本不打印 Secret 值。",
        )
    # get secret uses API but never prints or persists its data.
    existing_cm = kube.get("configmap", cm["metadata"]["name"])
    if (
        existing_cm
        and existing_cm.get("metadata", {}).get("labels", {}).get(LABEL) != OWNER
    ):
        raise Error("E_OWNERSHIP", "同名 ConfigMap 不属于此部署器，拒绝覆盖")
    kube.apply(cm)
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": settings["name"] + "-check-" + uuid.uuid4().hex[:8],
            "namespace": settings["namespace"],
            "labels": {LABEL: OWNER, "tributo.knova/role": "doctor"},
        },
        "spec": copy.deepcopy(deployment["spec"]["template"]["spec"]),
    }
    pod["spec"]["restartPolicy"] = "Never"
    container = pod["spec"]["containers"][0]
    container.pop("readinessProbe", None)
    container["command"] = ["python", "/etc/tributo-knova/doctor.py"]
    container["args"] = [
        "--timeout",
        str(max(10, int(settings.get("timeout_seconds", 360)) - 45)),
    ]
    if idle_only:
        container["args"].append("--idle-only")
    name = pod["metadata"]["name"]
    report("doctor.created", pod=name, consumes_business_tasks=False)
    kube.apply(pod)
    complete = False
    try:
        deadline = time.monotonic() + int(settings.get("timeout_seconds", 360))
        while time.monotonic() < deadline:
            current = kube.get("pod", name)
            if current and current.get("status", {}).get("phase") in {
                "Succeeded",
                "Failed",
            }:
                lines = kube.run("logs", name) or ""
                records = []
                for line in lines.splitlines():
                    if line.startswith("KNOVA_DOCTOR "):
                        value = json.loads(line[len("KNOVA_DOCTOR ") :])
                        records.append(value)
                        print(json.dumps(value, ensure_ascii=False), flush=True)
                write_json(
                    Path(settings["output"]) / (name + ".json"),
                    {"pod": diagnose_pod(current), "records": records},
                )
                complete = current["status"]["phase"] == "Succeeded" and any(
                    row.get("stage") == ("ray.idle" if idle_only else "complete")
                    and row.get("status") == "PASS"
                    for row in records
                )
                if not complete:
                    raise Error(
                        "E_PREFLIGHT",
                        "自查失败；按上述 code/stage 修正配置；Consumer 未开始切换",
                    )
                return records
            time.sleep(1)
        details = diagnose_pod(kube.get("pod", name) or {})
        report("doctor.timeout", "FAIL", pod=name, **details)
        raise Error(
            "E_PREFLIGHT_TIMEOUT",
            "检查 Pod 调度/拉镜像/RBAC/secret/资源/连接；仅打印安全状态字段",
        )
    finally:
        if not keep:
            kube.delete("pod", name)
        report("doctor.cleanup", pod=name, retained=keep, passed=complete)


def assert_identity(kube, name, expected_uid, *, managed=False, missing_ok=False):
    current = kube.get("deployment", name)
    if not current and missing_ok:
        return
    if not current or current["metadata"].get("uid") != expected_uid:
        raise Error("E_IDENTITY", "Deployment 身份已变更，拒绝自动覆盖: " + name)
    if managed and current["metadata"].get("labels", {}).get(LABEL) != OWNER:
        raise Error("E_OWNERSHIP", "目标 Deployment 所有权已变化，拒绝操作: " + name)


def classify_consumer_log(text):
    """Return fixed classifications only; never include a log line or match text."""
    patterns = (
        (
            r"(?:redis\.exceptions\.)?NoPermissionError:",
            "E_REDIS_ACL",
            "Redis 业务 key/命令权限不足；核对 task/event/cancel 前缀 ACL",
        ),
        (
            r"(?:redis\.exceptions\.)?AuthenticationError:",
            "E_REDIS_AUTH",
            "核对 Redis 数据节点与 Sentinel 的独立认证配置",
        ),
        (
            r"RedisCredentialUnavailable:",
            "E_SECRET",
            "Consumer 缺少配置引用的 Redis 环境变量或凭据文件",
        ),
        (
            r"(?:ModuleNotFoundError|ImportError):",
            "E_IMPORT",
            "核对镜像模块、原生依赖库与实际 Python 导入路径",
        ),
        (
            r"(?:ValidationError|_TrainingConfigurationError):",
            "E_CONFIG",
            "配置或任务协议校验失败；核对 config.json 与本版本字段",
        ),
        (r"MemoryError:", "E_MEMORY", "检查 Pod 内存限制、OOM 事件及当前数据批大小"),
    )
    for pattern, code, hint in patterns:
        if re.search(pattern, text):
            return {"code": code, "hint": hint, "source": "known_exception_pattern"}
    return {
        "code": "E_LOG_UNCLASSIFIED",
        "hint": "近期日志未命中安全分类；结合 Pod 状态及人工日志排查",
    }


def recent_log_diagnostic(kube, pod_name):
    for previous in (False, True):
        try:
            args = ["logs", pod_name, "--tail=80", "--limit-bytes=16384"]
            if previous:
                args.append("--previous")
            raw = kube.run(*args)
            classified = classify_consumer_log(raw or "")
            if classified["code"] != "E_LOG_UNCLASSIFIED":
                return classified
        except Error:
            continue
    return {
        "code": "E_LOG_UNCLASSIFIED",
        "hint": "没有可分类的近期日志；核对 Pod 状态和 logs RBAC",
    }


def save_diagnostics(kube, settings):
    name = settings["name"]
    current = kube.get("deployment", name)
    if not current:
        return
    selector = ",".join(
        key + "=" + value
        for key, value in current["spec"]["selector"]["matchLabels"].items()
    )
    pods = json.loads(kube.run("get", "pods", "-l", selector, "-o", "json"))["items"]
    details = {
        "deployment": name,
        "pods": [
            {
                "name": pod["metadata"]["name"],
                **diagnose_pod(pod),
                "diagnostic": recent_log_diagnostic(kube, pod["metadata"]["name"]),
            }
            for pod in pods
        ],
    }
    write_json(Path(settings["output"]) / "rollout-failure.json", details)
    report("rollout.diagnostics", "FAIL", **details)


def restore(kube, settings, state, cm, deployment):
    for name, uid in state.get("snapshot_uids", {}).items():
        assert_identity(kube, name, uid, managed=name == settings["name"])
    if state.get("new_uid"):
        assert_identity(kube, settings["name"], state["new_uid"], managed=True)
    elif settings["name"] not in state.get("snapshot_uids", {}) and kube.get(
        "deployment", settings["name"]
    ):
        raise Error("E_IDENTITY", "无法确认新 Deployment 身份，停止自动回滚")
    # Stop the replacement first, even if health or preflight failed.
    kube.scale_down(settings["name"])
    state["phase"] = "rollback_stopped_new"
    write_json(Path(settings["output"]) / "state.json", state)
    check(kube, settings, cm, deployment, idle_only=True)
    for old in state["snapshots"]:
        kube.apply(old)
        if old["spec"].get("replicas", 1):
            kube.rollout(old["metadata"]["name"])
    if not any(
        old["metadata"]["name"] == settings["name"] for old in state["snapshots"]
    ):
        kube.delete("deployment", settings["name"])
    state["phase"] = "rolled_back"
    write_json(Path(settings["output"]) / "state.json", state)
    report("rollback.complete")


def deploy(kube, settings, cm, deployment, upstream_paused):
    if not upstream_paused:
        raise Error(
            "E_QUIESCE",
            "apply 必须带 --upstream-paused，表示上游已暂停；"
            "预检还会检查活动 Tributo Ray Jobs。",
        )
    state_path = Path(settings["output"]) / "state.json"
    fingerprint = hashlib.sha256(
        json.dumps([settings["context"], cm, deployment], sort_keys=True).encode()
    ).hexdigest()
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if any(
            previous.get(key) != settings[key]
            for key in ("context", "namespace", "name")
        ):
            raise Error(
                "E_JOURNAL",
                "output 中已有另一个 context/namespace/name 的快照，选择独立 output",
            )
        if (
            previous.get("phase") == "complete"
            and previous.get("fingerprint") == fingerprint
        ):
            assert_identity(
                kube, settings["name"], previous.get("new_uid"), managed=True
            )
            current = kube.get("deployment", settings["name"])
            if current["spec"] != previous.get("applied_spec"):
                raise Error(
                    "E_DRIFT",
                    "已部署对象发生人工变更；先 status，确认差异后使用独立 output",
                )
            if current.get("status", {}).get("readyReplicas", 0) < current["spec"].get(
                "replicas", 1
            ):
                raise Error(
                    "E_UNHEALTHY",
                    "相同配置已部署但未就绪；执行 status 定位，不重复停启",
                )
            report(
                "deploy.unchanged",
                deployment=settings["name"],
                original_rollback_snapshot_preserved=True,
            )
            return
        if previous.get("phase") not in {"complete", "rolled_back"}:
            raise Error(
                "E_JOURNAL", "上次切换未完成；先 status，再 rollback；不要覆盖恢复快照"
            )
    existing = kube.get("deployment", settings["name"])
    if existing and existing.get("metadata", {}).get("labels", {}).get(LABEL) != OWNER:
        raise Error(
            "E_OWNERSHIP",
            "目标 Deployment 不属于此部署器；用新 name 并通过 old_deployment 明确迁移",
        )
    check(kube, settings, cm, deployment)
    snapshots = []
    snapshot_uids = {}
    for name in (settings.get("old_deployment"), settings["name"]):
        if name:
            found = kube.get("deployment", name)
            if found:
                snapshots.append(clean_document(found))
                snapshot_uids[name] = found["metadata"].get("uid")
            elif name == settings.get("old_deployment"):
                raise Error(
                    "E_OLD_MISSING",
                    "指定 old_deployment 不存在；修正拼写或首次部署时移除此字段",
                )
    state = {
        "schema": 1,
        "context": settings["context"],
        "namespace": settings["namespace"],
        "name": settings["name"],
        "phase": "prepared",
        "snapshots": snapshots,
        "configmap": cm["metadata"]["name"],
        "snapshot_uids": snapshot_uids,
        "fingerprint": fingerprint,
    }
    if state_path.exists():
        state_path.replace(
            state_path.with_name("state-previous-" + uuid.uuid4().hex[:8] + ".json")
        )
    write_json(state_path, state)
    try:
        for old in snapshots:
            old_name = old["metadata"]["name"]
            assert_identity(
                kube,
                old_name,
                snapshot_uids[old_name],
                managed=old_name == settings["name"],
            )
            kube.scale_down(old_name)
        state["phase"] = "old_stopped"
        write_json(state_path, state)
        # Recheck after old Pods exit; catches jobs admitted during the first check.
        check(kube, settings, cm, deployment, idle_only=True)
        kube.apply(deployment)
        applied = kube.get("deployment", settings["name"])
        state["new_uid"] = applied["metadata"]["uid"]
        state["phase"] = "new_started"
        write_json(state_path, state)
        kube.rollout(settings["name"])
        state["applied_spec"] = kube.get("deployment", settings["name"])["spec"]
        state["phase"] = "complete"
        write_json(state_path, state)
        report(
            "deploy.complete",
            deployment=settings["name"],
            image=settings["image"],
            context=settings["context"],
            namespace=settings["namespace"],
        )
    except BaseException:
        try:
            save_diagnostics(kube, settings)
        except Exception:
            report(
                "rollout.diagnostics",
                "UNAVAILABLE",
                hint="保留 state.json，并手工检查 Consumer Pod 安全状态字段",
            )
        try:
            restore(kube, settings, state, cm, deployment)
        except Exception as exc:
            report(
                "rollback.blocked",
                "FAIL",
                code=getattr(exc, "code", "E_ROLLBACK"),
                hint=(
                    "自动回滚未完成；保留 state.json 并执行 status。"
                    "确认资源身份和活动作业后 rollback，不同时启动两套 Consumer。"
                ),
            )
        raise


def status(kube, settings):
    state_path = Path(settings["output"]) / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        report("journal", phase=state.get("phase"))
    for name in filter(None, (settings.get("old_deployment"), settings["name"])):
        item = kube.get("deployment", name)
        if not item:
            report("deployment", "ABSENT", name=name)
            continue
        report(
            "deployment",
            name=name,
            replicas=item["spec"].get("replicas", 1),
            ready=item.get("status", {}).get("readyReplicas", 0),
            images=[
                container["image"]
                for container in item["spec"]["template"]["spec"]["containers"]
            ],
        )
        selector = ",".join(
            key + "=" + value
            for key, value in item["spec"]["selector"]["matchLabels"].items()
        )
        pods = json.loads(kube.run("get", "pods", "-l", selector, "-o", "json"))[
            "items"
        ]
        for pod in pods:
            details = diagnose_pod(pod)
            if any(
                container.get("restartCount", 0) for container in details["containers"]
            ):
                details["diagnostic"] = recent_log_diagnostic(
                    kube, pod["metadata"]["name"]
                )
            report("pod", name=pod["metadata"]["name"], **details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "action", choices=("render", "check", "apply", "status", "rollback")
    )
    parser.add_argument("--upstream-paused", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings, config, probe = load_settings(args.settings.resolve())
        if args.action == "apply" and not probe.get("require_idle", True):
            raise Error(
                "E_QUIESCE",
                "apply 要求 probe.require_idle=true；检查模式可显式设 false",
            )
        cm, deployment = manifests(settings, config, probe)
        if args.action == "render":
            print(
                json.dumps(
                    {"apiVersion": "v1", "kind": "List", "items": [cm, deployment]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        output = Path(settings["output"])
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Error("E_LOCK", "该 output 目录已有部署命令运行") from None
            kube = Kube(settings)
            if args.action == "check":
                check(kube, settings, cm, deployment)
            elif args.action == "apply":
                deploy(kube, settings, cm, deployment, args.upstream_paused)
            elif args.action == "status":
                status(kube, settings)
            else:
                if not args.upstream_paused:
                    raise Error("E_QUIESCE", "rollback 必须带 --upstream-paused")
                state = json.loads((output / "state.json").read_text())
                if any(
                    state.get(key) != settings[key]
                    for key in ("context", "namespace", "name")
                ):
                    raise Error(
                        "E_JOURNAL",
                        "恢复快照的 context/namespace/name 与本次参数不一致",
                    )
                restore(kube, settings, state, cm, deployment)
        return 0
    except Exception as exc:
        report(
            "command",
            "FAIL",
            code=getattr(exc, "code", "E_LOCAL"),
            message=str(exc) if isinstance(exc, Error) else type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
