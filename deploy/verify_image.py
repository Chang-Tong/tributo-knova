"""Check that Ray py_modules contain the same Python sources as the consumer."""

import hashlib
import sysconfig
from pathlib import Path


def sources(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }


for package in (
    "tributo_knova",
    "tributo_broker_redis",
    "tributo_algorithms_boosting",
):
    installed = sources(Path(sysconfig.get_paths()["purelib"]) / package)
    distributed = sources(Path("/opt/tributo-runtime/extensions") / package)
    if not installed or installed != distributed:
        raise SystemExit(f"Ray/consumer source mismatch: {package}")
    print(f"Identical consumer/Ray sources: {package} ({len(installed)} files)")
