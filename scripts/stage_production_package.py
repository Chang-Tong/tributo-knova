#!/usr/bin/env python3
"""Stage the reviewed v2.5 deployment package from an explicit file allowlist.

The default only stages files and audits them. Pass --archive only after the
final workload, Sentinel, and release records have been completed and approved.
This script does not start Docker, Ray, Kubernetes, or application tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "dist/comprehensive-recheck"
STAGE = ROOT / "dist/knova-production-v2.5"
IMAGE_ID = "sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101"

DEPLOYMENT_FILES = (
    "ACCEPTANCE.md",
    "COMPREHENSIVE-V25.md",
    "LOGGING-RECHECK.md",
    "OFFLINE-V25.md",
    "PREFLIGHT-V25.md",
    "README.md",
    "check_cluster.py",
    "config.example.json",
    "deploy.py",
    "import-image.sh",
    "release.json",
    "settings.example.json",
    "consumer-only/.gitignore",
    "consumer-only/README.md",
    "consumer-only/config.example.json",
    "consumer-only/consumer.yaml",
    "consumer-only/kustomization.yaml",
)
SCRIPT_FILES = (
    "acceptance_local.sh",
    "acceptance.compose.yaml",
    "acceptance_workloads.py",
    "acceptance_sentinel.sh",
    "acceptance_sentinel.py",
    "local_smoke.py",
    "package_image.sh",
)
EVIDENCE_FILES = (
    "FINAL-QUALITY-REVIEW.md",
    "INDEPENDENT-REVIEW.md",
    "archive-verification.json",
    "offline-context-source-verification.json",
    "package-failure-regression.json",
    "auth/root-cause.md",
    "backend/knova-complete-pytest.log",
    "backend/knova-complete-junit.xml",
    "backend/packaged-dependencies-pytest.log",
    "backend/packaged-dependencies-junit.xml",
    "empty-shards/review.md",
    "empty-shards/commands.json",
    "empty-shards/wheel-provenance.json",
    "empty-shards/native-ray-job.json",
    "empty-shards/native-ray-lazy-job.json",
    "frontend/UI-ACCEPTANCE.md",
    "frontend/backend-evidence.md",
    "frontend/failed-training-query-sanitized.json",
    "frontend/inference-shap-verification.json",
    "frontend/inference-six-execution-events.json",
    "frontend/inference-six-job-summary.json",
    "frontend/inference-threshold-comparison.json",
    "frontend/v5-s3-artifact-verification.json",
    "frontend/v5-training-summary.json",
    "frontend/v5-ray-job-sanitized.log",
    "frontend/iexec-15afh75up0hhbk-ray-job-sanitized.log",
    "frontend/iexec-9exndmcz9kf99c-ray-job-sanitized.log",
    "frontend/iexec-evpab58i9ui5mn-ray-job-sanitized.log",
    "frontend/iexec-gz27lidphnjhwu-ray-job-sanitized.log",
    "frontend/iexec-igxeoa9yjqmk97-ray-job-sanitized.log",
    "frontend/iexec-zc7w8opdy13dwn-ray-job-sanitized.log",
    "preflight/final-9f9c-image-preflight.log",
    "preflight/final-9f9c-image-preflight.json",
    "preflight/final-9f9c-image-preflight-cleanup.json",
    "preflight/final-9f9c-image-preflight-secret-scan.json",
    "vendor-provenance/broker-generic-training-limit.md",
    "vendor-provenance/broker-rebuild.json",
    "vendor-provenance/generic-training-final-image-probe.json",
    "vendor-provenance/generic-training-final-image-probe-metadata.json",
    "vendor-provenance/generic-training-wheel-static-check.json",
    "vendor-provenance/ray-clickhouse-source-check.json",
)
# These records may be absent while the final acceptance is still running.
# A final archive requires all of them, in addition to successful case counts.
FINAL_EVIDENCE_FILES = (
    "workloads-final/command.json",
    "workloads-final/running-image-provenance.json",
    "workloads-final/workload-results.json",
    "workloads-final/summary.json",
    "workloads-final/final-summary.md",
    "sentinel-9f9c-final/command.json",
    "sentinel-9f9c-final/sentinel-results.json",
    "sentinel-9f9c-final/summary.json",
    "sentinel-9f9c-final/final-summary.md",
)
FORBIDDEN_PARTS = {
    "runtime.env",
    ".env",
    "config.json",
    "settings.json",
    "packaged-venv",
    "shared",
    "site-packages",
    "__pycache__",
    ".DS_Store",
}
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
SYNTHETIC_FIXTURE_VALUES = {
    "local-repro-only",
    "local-sentinel-only",
    "local-test-only",
    "local-test-secret",
    "localtestkey",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files() -> tuple[dict[str, Path], list[str]]:
    selected = {name: ROOT / "deploy/production" / name for name in DEPLOYMENT_FILES}
    selected.update(
        {"scripts/" + name: ROOT / "scripts" / name for name in SCRIPT_FILES}
    )
    selected.update({"evidence/" + name: EVIDENCE / name for name in EVIDENCE_FILES})
    missing_final = []
    for name in FINAL_EVIDENCE_FILES:
        source = EVIDENCE / name
        if source.exists():
            selected["evidence/" + name] = source
        else:
            missing_final.append(name)
    for target, source in selected.items():
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(
                f"allowlisted source is missing or not a regular file: {target}"
            )
        if not source.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"allowlisted source escapes repository: {target}")
        if any(parent.is_symlink() for parent in source.parents if parent != ROOT):
            raise RuntimeError(f"allowlisted source traverses a symlink: {target}")
    return selected, missing_final


def local_secret_values() -> dict[str, str]:
    # Read only in memory. Never copy or print the environment file or its values.
    source = EVIDENCE / "frontend/runtime.env"
    values = {}
    for line in source.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value and re.search(r"SECRET|PASSWORD|TOKEN|ACCESS_KEY_ID", key):
            values[key] = value
    return values


def scan_content(target: str, data: bytes, secrets: dict[str, str]) -> None:
    if FORBIDDEN_PARTS.intersection(PurePosixPath(target).parts):
        raise RuntimeError(f"forbidden archive path: {target}")
    if PurePosixPath(target).suffix in {".env", ".pem", ".key", ".p12", ".whl"}:
        raise RuntimeError(f"forbidden archive file type: {target}")
    content = data.decode("utf-8")
    if JWT.search(content):
        raise RuntimeError(f"JWT-shaped credential detected in {target}")
    for key, value in secrets.items():
        if target.startswith("scripts/") and value in SYNTHETIC_FIXTURE_VALUES:
            continue
        if key.endswith("ACCESS_KEY_ID"):
            # The local public access ID is also a common project name. Match
            # assignments to its environment key, not unrelated prose/paths.
            matched = re.search(
                re.escape(key) + r"[\"'\s:=]+[\"']?" + re.escape(value) + r"\b",
                content,
            )
        else:
            matched = value in content
        if matched:
            raise RuntimeError(f"runtime credential detected in {target} ({key})")


def readiness(stage: Path, missing_final: list[str]) -> list[str]:
    issues = ["final evidence missing: " + name for name in missing_final]
    release = json.loads((stage / "release.json").read_text())
    if (
        release.get("image") != "knova-tributo-devbox:v2.5"
        or release.get("image_id") != IMAGE_ID
    ):
        issues.append("release.json does not identify the final v2.5 image")
    if not re.fullmatch(r"[a-f0-9]{64}", str(release.get("archive_sha256", ""))):
        issues.append("release.json has no final archive SHA-256")
    if release.get("archive") != "knova-tributo-devbox-arm64-v2.5.tar.zst":
        issues.append("release.json has no final v2.5 image archive name")
    if release.get("kubernetes", {}).get("apply") is True:
        issues.append(
            "release.json still attributes historical Kubernetes apply to this release"
        )
    for filename, expected_count in (
        ("workloads-final/workload-results.json", 17),
        ("sentinel-9f9c-final/sentinel-results.json", 8),
    ):
        path = stage / "evidence" / filename
        cases = json.loads(path.read_text()) if path.exists() else []
        if len(cases) != expected_count or any(
            case.get("passed") is not True for case in cases
        ):
            issues.append(f"{filename} is not {expected_count}/{expected_count} passed")
    for folder in ("workloads-final", "sentinel-9f9c-final"):
        path = stage / "evidence" / folder / "command.json"
        command = json.loads(path.read_text()) if path.exists() else {}
        if command.get("image_id") != IMAGE_ID or command.get("returncode") != 0:
            issues.append(f"{folder} did not successfully run against the final image")
        summary_path = stage / "evidence" / folder / "final-summary.md"
        if not summary_path.exists() or IMAGE_ID not in summary_path.read_text():
            issues.append(f"{folder} final summary does not identify the final image")
    report = (stage / "COMPREHENSIVE-V25.md").read_text()
    if "复测进行中" in report or "完成后更新" in report:
        issues.append("COMPREHENSIVE-V25.md still contains pending final results")
    preflight = json.loads(
        (stage / "evidence/preflight/final-9f9c-image-preflight.json").read_text()
    )
    if preflight.get("image_id") != IMAGE_ID or preflight.get("status") != "SUCCEEDED":
        issues.append("final image preflight is missing or unsuccessful")
    image_archive = json.loads(
        (stage / "evidence/archive-verification.json").read_text()
    )
    if (
        image_archive.get("repo_tags") != ["knova-tributo-devbox:v2.5"]
        or image_archive.get("verified") is not True
        or image_archive.get("image_config") != "blobs/" + IMAGE_ID.replace(":", "/")
        or image_archive.get("platform") != "linux/arm64"
        or image_archive.get("archive_bytes") != release.get("archive_bytes")
    ):
        issues.append("image archive verification does not match the deployed tag")
    return issues


def stage_files() -> tuple[dict[str, str], list[str]]:
    selected, missing_final = collect_files()
    secrets = local_secret_values()
    with tempfile.TemporaryDirectory(
        prefix="knova-production-stage-", dir=ROOT / "dist"
    ) as temporary:
        destination = Path(temporary)
        records = []
        for target, source in sorted(selected.items()):
            data = source.read_bytes()
            scan_content(target, data, secrets)
            output = destination / target
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            output.chmod(source.stat().st_mode & 0o777)
            records.append(
                {
                    "path": target,
                    "source": str(source.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": digest(data),
                }
            )
        issues = readiness(destination, missing_final)
        audit = {
            "created_at": datetime.now(UTC).isoformat(),
            "expected_image_id": IMAGE_ID,
            "status": "awaiting-final-records" if issues else "ready-for-final-archive",
            "readiness_issues": issues,
            "payload_file_count": len(records),
            "payload_bytes": sum(record["bytes"] for record in records),
            "credentials_scan": {
                "passed": True,
                "runtime_secret_values_checked": len(secrets),
                "jwt_shape_check": True,
                "access_key_id_check": (
                    "environment-key assignments; "
                    "public project-name substrings excluded"
                ),
                "synthetic_fixture_values_allowed_only_in_scripts": sorted(
                    SYNTHETIC_FIXTURE_VALUES
                ),
            },
            "contains_container_images": False,
            "contains_real_runtime_configuration": False,
            "contains_unsanitized_application_logs": False,
            "forbidden_archive_path_parts": sorted(FORBIDDEN_PARTS),
            "files": records,
        }
        (destination / "PACKAGE-AUDIT.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        hashes = {
            str(path.relative_to(destination)): digest(path.read_bytes())
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        (destination / "SHA256SUMS").write_text(
            "".join(f"{value}  {name}\n" for name, value in sorted(hashes.items()))
        )
        hashes["SHA256SUMS"] = digest((destination / "SHA256SUMS").read_bytes())
        if STAGE.exists():
            if STAGE.is_symlink() or not (STAGE / "PACKAGE-AUDIT.json").is_file():
                raise RuntimeError(
                    "refusing to replace a staging directory not owned by this script"
                )
            shutil.rmtree(STAGE)
        shutil.copytree(destination, STAGE)
    audit_directory(hashes)
    return hashes, issues


def audit_directory(hashes: dict[str, str]) -> None:
    files = {
        str(path.relative_to(STAGE)): path
        for path in STAGE.rglob("*")
        if path.is_file()
    }
    if set(files) != set(hashes):
        raise RuntimeError("staging contains files outside the generated manifest")
    for name, path in files.items():
        if path.is_symlink() or digest(path.read_bytes()) != hashes[name]:
            raise RuntimeError(f"staged file changed or became a symlink: {name}")


def make_archive(hashes: dict[str, str]) -> Path:
    release = json.loads((STAGE / "release.json").read_text())
    image_archive = ROOT / release["archive"]
    if not image_archive.exists():
        image_archive = ROOT / "dist" / release["archive"]
    if image_archive.stat().st_size != release["archive_bytes"]:
        raise RuntimeError("actual image archive size does not match release.json")
    checksum = hashlib.sha256()
    with image_archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != release["archive_sha256"]:
        raise RuntimeError("actual image archive SHA-256 does not match release.json")
    archive = STAGE.parent / (STAGE.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as package:
        package.add(STAGE, arcname=STAGE.name)
    archived = {}
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != STAGE.name:
                raise RuntimeError("archive contains an unsafe member path")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("archive contains a non-regular file")
            relative = str(name.relative_to(STAGE.name))
            if relative in archived:
                raise RuntimeError("archive contains duplicate members")
            archived[relative] = digest(package.extractfile(member).read())
    if archived != hashes:
        raise RuntimeError("archive contents do not match the audited staging")
    checksum = digest(archive.read_bytes())
    archive.with_name(archive.name + ".sha256").write_text(
        f"{checksum}  {archive.name}\n"
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="store_true",
        help="create tar.gz only when all final records are ready",
    )
    args = parser.parse_args()
    hashes, issues = stage_files()
    if args.archive and issues:
        raise SystemExit("archive blocked: " + "; ".join(issues))
    archive = make_archive(hashes) if args.archive else None
    print(
        json.dumps(
            {
                "staging": str(STAGE),
                "files": len(hashes),
                "readiness_issues": issues,
                "archive_created": str(archive) if archive else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
