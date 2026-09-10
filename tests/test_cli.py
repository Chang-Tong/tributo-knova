from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

from tributo_knova import cli


class _Runtime:
    def __init__(self) -> None:
        self.started = 0
        self.run_once_calls = 0
        self.run_forever_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.started += 1

    def run_once(self) -> None:
        self.run_once_calls += 1

    def run_forever(self) -> None:
        self.run_forever_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _config_file(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    value: dict[str, object] = {"broker_id": "tributo-knova"}
    path = tmp_path / "knova.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_validate_uses_knova_plugin_config_validation(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    path, value = _config_file(tmp_path)
    calls: list[tuple[dict[str, object], bool]] = []

    class Plugin:
        def validate_config(
            self,
            config: dict[str, object],
            *,
            check_connectivity: bool = False,
        ) -> None:
            calls.append((config, check_connectivity))

        def create_runtime(self, _config: dict[str, object]) -> _Runtime:
            raise AssertionError("validate must not create a runtime")

    monkeypatch.setattr(cli, "KnovaBrokerPlugin", Plugin)

    result = cli.main(
        ["validate", "--config", str(path), "--check-connectivity"]
    )

    assert result == 0
    assert calls == [(value, True)]
    assert capsys.readouterr().out == "KnoVa broker configuration is valid\n"


def test_consume_once_uses_knova_runtime_and_installs_shutdown_handlers(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    path, value = _config_file(tmp_path)
    runtime = _Runtime()
    created_with: list[dict[str, object]] = []
    handlers: dict[int, Any] = {}

    class Plugin:
        def create_runtime(self, config: dict[str, object]) -> _Runtime:
            created_with.append(config)
            return runtime

    monkeypatch.setattr(cli, "KnovaBrokerPlugin", Plugin)
    monkeypatch.setattr(
        cli.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    assert cli.main(["consume", "--config", str(path), "--once"]) == 0
    assert created_with == [value]
    assert runtime.started == 1
    assert runtime.run_once_calls == 1
    assert runtime.run_forever_calls == 0
    assert runtime.close_calls == 1
    assert set(handlers) == {signal.SIGINT, signal.SIGTERM}

    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert runtime.close_calls == 2


def test_consume_without_once_runs_forever(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    path, _value = _config_file(tmp_path)
    runtime = _Runtime()

    class Plugin:
        def create_runtime(self, _config: dict[str, object]) -> _Runtime:
            return runtime

    monkeypatch.setattr(cli, "KnovaBrokerPlugin", Plugin)
    monkeypatch.setattr(cli.signal, "signal", lambda *_args: None)

    assert cli.main(["consume", "--config", str(path)]) == 0
    assert runtime.started == 1
    assert runtime.run_once_calls == 0
    assert runtime.run_forever_calls == 1
    assert runtime.close_calls == 1


def test_config_root_must_be_an_object(tmp_path: Path) -> None:
    path = tmp_path / "knova.json"
    path.write_text("[]", encoding="utf-8")

    try:
        cli.main(["validate", "--config", str(path)])
    except ValueError as exc:
        assert str(exc) == "provider config root must be an object"
    else:
        raise AssertionError("non-object config unexpectedly accepted")
