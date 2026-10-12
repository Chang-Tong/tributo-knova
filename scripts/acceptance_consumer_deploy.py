#!/usr/bin/env python3
"""Exercise the real deployment CLI against the disposable local Kind cluster."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/extended-acceptance-20260925/kubernetes"
CONTEXT = "kind-knova-existing-accept"
NAMESPACE = "knova-existing"
DEPLOY = ROOT / "deploy/production/consumer-only/deploy.py"
ENV = {**os.environ, "KUBECONFIG": str(OUT / "kubeconfig")}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def kubectl(*args):
    return subprocess.run(
        ["kubectl", "--context", CONTEXT, "--namespace", NAMESPACE, *args],
        env=ENV,
        check=True,
        text=True,
        capture_output=True,
        timeout=300,
    ).stdout


def objects(kind, *args):
    return json.loads(kubectl("get", kind, *args, "-o", "json"))


def fixture_objects():
    result = objects("deployments")
    result["items"] = [
        item
        for item in result["items"]
        if item["spec"]["template"]["metadata"].get("labels", {}).get("fixture")
    ]
    expected = {
        "redis-master",
        "redis-replica",
        "redis-sentinel",
        "clickhouse",
        "rustfs",
        "ray-head",
        "ray-worker",
    }
    assert {item["metadata"]["name"] for item in result["items"]} == expected
    return result


def run_case(name, settings, action="check", expected_code=None, flags=()):
    case_dir = OUT / "cases" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # Paths were normalized to the original local fixture directory by main().
    path = case_dir / "settings.json"
    write(path, settings)
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(DEPLOY), "--settings", str(path), action, *flags],
        env=ENV,
        text=True,
        capture_output=True,
        timeout=int(settings.get("timeout_seconds", 240)) * 4 + 30,
    )
    (case_dir / "stdout.log").write_text(proc.stdout)
    (case_dir / "stderr.log").write_text(proc.stderr)
    records = []
    if action != "render":
        for line in proc.stdout.splitlines():
            with suppress(json.JSONDecodeError):
                records.append(json.loads(line))
    codes = {row.get("code") for row in records if isinstance(row, dict)}
    passed = (
        proc.returncode == 0
        if expected_code is None
        else (proc.returncode != 0 and expected_code in codes)
    )
    result = {
        "case": name,
        "action": action,
        "passed": passed,
        "exit_code": proc.returncode,
        "expected_code": expected_code,
        "observed_codes": sorted(code for code in codes if code),
        "seconds": round(time.monotonic() - start, 2),
    }
    results_path = OUT / "deployment-results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    results.append(result)
    write(results_path, results)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError("deployment acceptance failed: " + name)
    return proc, records


def variant_config(name, settings, transform):
    config = json.loads(Path(settings["config"]).read_text())
    transform(config)
    path = OUT / (name + "-config.json")
    write(path, config)
    return {**settings, "config": str(path), "output": str(OUT / ("release-" + name))}


def variant_probe(name, settings, transform):
    probe = json.loads(Path(settings["probe"]).read_text())
    transform(probe)
    path = OUT / (name + "-probe.json")
    write(path, probe)
    return {**settings, "probe": str(path), "output": str(OUT / ("release-" + name))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=["early", "preflight", "switch", "faults", "rollback", "auto-rollback"],
    )
    args = parser.parse_args()
    settings = json.loads((OUT / "settings.json").read_text())
    assert settings["context"] == CONTEXT and settings["namespace"] == NAMESPACE
    for key in ("config", "probe", "output"):
        settings[key] = str((OUT / settings[key]).resolve())
    if args.phase == "early":
        proc, _ = run_case("render_only", settings, action="render")
        manifest = json.loads(proc.stdout)
        assert sorted(item["kind"] for item in manifest["items"]) == [
            "ConfigMap",
            "Deployment",
        ]
        run_case(
            "requires_upstream_pause",
            settings,
            action="apply",
            expected_code="E_QUIESCE",
        )
        run_case(
            "missing_namespace",
            {**settings, "namespace": "knova-absent-fixture"},
            expected_code="E_NAMESPACE",
        )
        run_case(
            "missing_secret",
            {**settings, "secret": "knova-absent-secret"},
            expected_code="E_SECRET",
        )
        bad = variant_config(
            "path_override",
            settings,
            lambda c: c["execution"]["env_vars"].update(PATH="/wrong/bin"),
        )
        run_case("reject_path_override", bad, expected_code="E_CONFIG")
    elif args.phase == "preflight":
        run_case("existing_cluster_preflight", settings)
    elif args.phase == "switch":
        before = fixture_objects()
        write(OUT / "existing-components-before.json", before)
        run_case(
            "apply_new_consumer", settings, action="apply", flags=("--upstream-paused",)
        )
        assert (
            objects("deployment", settings["old_deployment"])["spec"]["replicas"] == 0
        )
        current = objects("deployment", settings["name"])
        assert current["status"].get("readyReplicas") == 1
        pods_before = objects(
            "pods", "-l", "app.kubernetes.io/name=" + settings["name"]
        )
        uids_before = {pod["metadata"]["uid"] for pod in pods_before["items"]}
        run_case(
            "idempotent_reapply", settings, action="apply", flags=("--upstream-paused",)
        )
        pods_after = objects("pods", "-l", "app.kubernetes.io/name=" + settings["name"])
        assert {pod["metadata"]["uid"] for pod in pods_after["items"]} == uids_before
        after = fixture_objects()
        write(OUT / "existing-components-after.json", after)
        original = {
            item["metadata"]["name"]: (item["metadata"]["uid"], item["spec"])
            for item in before["items"]
        }
        current = {
            item["metadata"]["name"]: (item["metadata"]["uid"], item["spec"])
            for item in after["items"]
        }
        assert original == current
        write(
            OUT / "switch-assertions.json",
            {
                "old_replicas": 0,
                "new_ready_replicas": 1,
                "idempotent_pod_uids": sorted(uids_before),
                "existing_component_specs_unchanged": True,
            },
        )
    elif args.phase == "faults":
        baseline = objects("pods", "-l", "app.kubernetes.io/name=" + settings["name"])

        def bad_master(c):
            c["transport"]["sentinel_master"] = "master-does-not-exist"

        run_case(
            "wrong_sentinel_master",
            variant_config("bad_master", settings, bad_master),
            expected_code="E_SENTINEL",
        )

        def absent_driver_secret(c):
            c["transport"]["password_ref"] = "env:ONLY_CONSUMER_REDIS_PASSWORD"

        secret = objects("secret", "fixture-env")
        data = copy.deepcopy(secret["data"])
        data["ONLY_CONSUMER_REDIS_PASSWORD"] = data["KNOVA_REDIS_PASSWORD"]
        temp_secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "consumer-only-secret", "namespace": NAMESPACE},
            "data": data,
        }
        subprocess.run(
            [
                "kubectl",
                "--context",
                CONTEXT,
                "--namespace",
                NAMESPACE,
                "apply",
                "-f",
                "-",
            ],
            env=ENV,
            input=json.dumps(temp_secret),
            text=True,
            capture_output=True,
            check=True,
        )
        bad = variant_config("missing_driver_secret", settings, absent_driver_secret)
        run_case(
            "secret_not_propagated_to_ray",
            {**bad, "secret": "consumer-only-secret"},
            expected_code="E_SECRET",
        )
        bad = variant_probe(
            "bad_ck_password",
            settings,
            lambda p: p["clickhouse"].update(password_ref="env:KNOVA_REDIS_PASSWORD"),
        )
        run_case("wrong_clickhouse_password", bad, expected_code="E_CK")
        bad = variant_probe(
            "missing_bucket",
            settings,
            lambda p: p["s3"].update(bucket="bucket-does-not-exist"),
        )
        run_case("missing_s3_bucket", bad, expected_code="E_S3")
        bad = variant_probe(
            "insufficient_topology", settings, lambda p: p.update(min_alive_nodes=5)
        )
        run_case("insufficient_ray_nodes", bad, expected_code="E_TOPOLOGY")
        final = objects("pods", "-l", "app.kubernetes.io/name=" + settings["name"])
        assert {p["metadata"]["uid"] for p in baseline["items"]} == {
            p["metadata"]["uid"] for p in final["items"]
        }
        write(OUT / "fault-isolation.json", {"live_consumer_pod_unchanged": True})
    elif args.phase == "auto-rollback":
        # Only the disposable fixture default user is modified. Preserve the
        # complete ACL in memory and verify it unchanged after the key-rule test.
        def acl_record():
            raw = json.loads(
                kubectl(
                    "exec",
                    "deployment/redis-master",
                    "--",
                    "sh",
                    "-ec",
                    'REDISCLI_AUTH="$KNOVA_REDIS_PASSWORD" '
                    "redis-cli --json ACL GETUSER default",
                )
            )
            return (
                raw
                if isinstance(raw, dict)
                else dict(zip(raw[::2], raw[1::2], strict=True))
            )

        before_acl = acl_record()
        assert before_acl["keys"] == "~*", "only the known fixture ACL is allowed"

        def reset_keys(restricted):
            patterns = "'%R~*' '%RW~tributo-doctor:*'" if restricted else "'~*'"
            command = (
                'REDISCLI_AUTH="$KNOVA_REDIS_PASSWORD" redis-cli '
                "ACL SETUSER default resetkeys " + patterns
            )
            assert (
                kubectl(
                    "exec", "deployment/redis-master", "--", "sh", "-ec", command
                ).strip()
                == "OK"
            )

        failed_settings = {
            **settings,
            "timeout_seconds": 120,
            "output": str(OUT / "release-auto-rollback"),
        }
        diagnostics = Path(failed_settings["output"]) / "rollout-failure.json"
        diagnostics.unlink(missing_ok=True)
        restored_after_failure = False
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            try:
                reset_keys(True)
                future = executor.submit(
                    run_case,
                    "auto_rollback_stream_acl",
                    failed_settings,
                    action="apply",
                    expected_code="E_KUBECTL",
                    flags=("--upstream-paused",),
                )
                deadline = time.monotonic() + 420
                while time.monotonic() < deadline and not future.done():
                    if diagnostics.exists():
                        # Real rollout failed. Restore ACL before the old
                        # Consumer restarts; verify actual readiness afterwards.
                        reset_keys(False)
                        restored_after_failure = True
                        break
                    time.sleep(0.25)
                if not restored_after_failure:
                    reset_keys(False)
                future.result(timeout=300)
            finally:
                reset_keys(False)
                assert acl_record() == before_acl, "fixture ACL must be fully restored"
        assert restored_after_failure
        previous = objects("deployment", settings["old_deployment"])
        assert (
            previous["spec"]["replicas"] == previous["status"].get("readyReplicas") == 1
        )
        state = json.loads((Path(failed_settings["output"]) / "state.json").read_text())
        assert state["phase"] == "rolled_back"
        remaining = json.loads(
            kubectl(
                "get",
                "deployment",
                settings["name"],
                "--ignore-not-found=true",
                "-o",
                "json",
            )
            or "null"
        )
        assert remaining is None
        write(
            OUT / "auto-rollback-assertions.json",
            {
                "rollout_failure_recorded": True,
                "acl_restored_after_rollout_failure": True,
                "full_acl_restored": True,
                "old_ready_replicas": 1,
                "replacement_deleted": True,
                "journal_phase": state["phase"],
            },
        )
    else:
        run_case("rollback", settings, action="rollback", flags=("--upstream-paused",))
        previous = objects("deployment", settings["old_deployment"])
        assert (
            previous["spec"]["replicas"] == previous["status"].get("readyReplicas") == 1
        )
        write(OUT / "rollback-assertions.json", {"old_ready_replicas": 1})


if __name__ == "__main__":
    main()
