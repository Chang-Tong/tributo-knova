"""KnoVa provider validation and production consume CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from tributo_knova.broker import KnovaBrokerPlugin

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _configure_logging(level: str | None) -> None:
    """Configure human-readable consumer logs without exposing config values."""
    selected = (
        level
        or os.environ.get("KNOVA_LOG_LEVEL")
        or os.environ.get("TRIBUTO_LOG_LEVEL")
        or "INFO"
    ).upper()
    if selected not in _LOG_LEVELS:
        raise ValueError(
            "log level must be one of: " + ", ".join(_LOG_LEVELS)
        )
    logging.basicConfig(
        level=getattr(logging, selected),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # basicConfig is a no-op when an embedding process already installed a
    # handler. Still raise the root threshold so provider INFO logs are visible.
    logging.getLogger().setLevel(getattr(logging, selected))


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provider config root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tributo KnoVa provider")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate provider config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--check-connectivity", action="store_true")
    health = commands.add_parser(
        "health", help="check configuration, Redis, and Ray readiness"
    )
    health.add_argument("--config", type=Path, required=True)
    consume = commands.add_parser("consume", help="run the provider consume loop")
    consume.add_argument("--config", type=Path, required=True)
    consume.add_argument("--once", action="store_true")
    consume.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default=None,
        type=str.upper,
        help="consumer log level (default: KNOVA_LOG_LEVEL or INFO)",
    )
    return parser


def _check_ray(config: dict[str, Any]) -> None:
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("provider execution config must be an object")
    dashboard_url = execution.get("ray_dashboard_url")
    if not isinstance(dashboard_url, str) or not dashboard_url.strip():
        raise ValueError("execution.ray_dashboard_url must be configured")
    with urlopen(f"{dashboard_url.rstrip('/')}/api/version", timeout=3) as response:
        if response.status != 200:
            raise ConnectionError("Ray dashboard readiness check failed")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = _read_config(args.config)
    plugin = KnovaBrokerPlugin()
    if args.command in {"validate", "health"}:
        plugin.validate_config(
            raw,
            check_connectivity=(args.command == "health" or args.check_connectivity),
        )
        if args.command == "health":
            _check_ray(raw)
            print("KnoVa consumer dependencies are healthy")
            return 0
        print("KnoVa broker configuration is valid")
        return 0

    _configure_logging(args.log_level)
    runtime = plugin.create_runtime(raw)

    def _stop(_signum: int, _frame: object) -> None:
        runtime.close()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        runtime.start()
        if args.once:
            runtime.run_once()
        else:
            runtime.run_forever()
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
