"""Non-network tests for safe per-node existing Ray diagnostics."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "consumer_doctor", ROOT / "deploy/production/consumer-only/doctor.py"
)
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


def test_secret_missing_reports_name_not_value(monkeypatch):
    monkeypatch.delenv("LOCAL_SECRET", raising=False)
    with pytest.raises(doctor.Fault) as caught:
        doctor.secret("env:LOCAL_SECRET", "test")
    result = doctor.failure(caught.value)
    assert result["code"] == "E_SECRET"
    assert result["reference"] == "env:LOCAL_SECRET"


def test_unknown_exception_does_not_expose_secrets():
    error = ValueError("password=super-secret")
    error.__cause__ = RuntimeError("nested-secret")
    assert "secret" not in json.dumps(doctor.failure(error))
    assert doctor.failure(error)["exception_type"] == "ValueError"


def inventory():
    return {
        "python": "3.13.5",
        "packages": {"ray": "2.55.1"},
        "sources": {
            "tributo_knova": {"sha256": "expected", "path": "/source", "files": 8}
        },
    }


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda item: item.update(python="3.12.5"), "E_IMPORT"),
        (lambda item: item["packages"].update(ray="2.54.0"), "E_IMPORT"),
        (
            lambda item: item["sources"]["tributo_knova"].update(sha256="old"),
            "E_SOURCE",
        ),
    ],
)
def test_node_mismatch_identifies_kind(change, code):
    expected = inventory()
    actual = copy.deepcopy(expected)
    change(actual)
    with pytest.raises(doctor.Fault) as caught:
        doctor.compare_inventory(actual, expected)
    assert caught.value.code == code


def test_layout_does_not_need_opt_path():
    expected, actual = inventory(), inventory()
    actual["sources"]["tributo_knova"]["path"] = (
        "/tmp/ray/session/runtime_resources/py_modules/tributo_knova"
    )
    doctor.compare_inventory(actual, expected)


def test_node_error_is_structured_and_has_no_traceback(monkeypatch):
    monkeypatch.setattr(doctor, "inventory", lambda p: inventory())
    monkeypatch.setattr(doctor, "check_environment", lambda p: None)

    def fail(*a, **k):
        raise doctor.Fault(
            "E_SECRET", "redis.driver", reference="env:KNOVA_REDIS_PASSWORD"
        )

    monkeypatch.setattr(doctor, "check_redis", fail)
    result = doctor.node_probe({}, {}, inventory())
    assert result["status"] == "FAIL"
    assert result["stage"] == "redis.driver"
    assert result["code"] == "E_SECRET"


def test_unconfigured_storage_is_explicit_not_checked():
    result = doctor.check_storage({})
    assert all(value.startswith("NOT_CHECKED:") for value in result.values())


@pytest.mark.parametrize(
    "probe",
    [
        {"s3": {"secret_key": "do-not-print"}},
        {"s3": {"secret_key_ref": "do-not-print"}},
        {"s3": {"endpoint_url": "http://user:secret@host"}},
        {"clickhouse": {"password": "do-not-print"}},
        {"algorithms": ["unknown"]},
        {"inference_formats": ["unknown"]},
        {"min_alive_nodes": 0},
    ],
)
def test_probe_rejects_plaintext_or_unknown_fields(probe):
    with pytest.raises(doctor.Fault) as caught:
        doctor.validate_probe(probe)
    assert caught.value.code == "E_CONFIG"
    assert "do-not-print" not in json.dumps(doctor.failure(caught.value))


def test_driver_failure_preserves_stage_without_duplicate_keyword(
    monkeypatch, tmp_path, capsys
):
    import ray

    for name in ("config.json", "probe.json", "expected.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(ray, "init", lambda **k: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(
        doctor,
        "node_probe",
        lambda *args: {"status": "FAIL", "stage": "redis.driver", "code": "E_REDIS"},
    )
    assert doctor.driver(tmp_path) == 1
    row = json.loads(capsys.readouterr().out.removeprefix(doctor.PREFIX))
    assert row == {
        "status": "FAIL",
        "stage": "ray.driver",
        "check_stage": "redis.driver",
        "code": "E_REDIS",
    }


def test_active_business_job_blocks_migration():
    busy = SimpleNamespace(
        submission_id="tributo-redis-training-test",
        status=SimpleNamespace(is_terminal=lambda: False),
    )
    with pytest.raises(doctor.Fault) as caught:
        doctor.active_jobs(SimpleNamespace(list_jobs=lambda: [busy]))
    assert caught.value.code == "E_BUSY"


@pytest.mark.parametrize(
    "name, result", [("table", "`table`"), ("db.table", "`db`.`table`")]
)
def test_qualified_table_quoted_by_segment(name, result):
    assert doctor.quote_table(name) == result


@pytest.mark.parametrize(
    "name", ["db.table;DROP TABLE secrets", "db.`table`", "x.y.z", "db..table"]
)
def test_probe_table_rejects_sql_fragments(name):
    with pytest.raises(doctor.Fault):
        doctor.quote_table(name)


def test_clickhouse_read_and_insert_grant(monkeypatch):
    import clickhouse_connect

    commands = []

    class Client:
        def command(self, query):
            commands.append(query)
            return "26.3.17" if query == "SELECT version()" else 1

        def query(self, query):
            commands.append(query)

        def close(self):
            commands.append("close")

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **kwargs: Client())
    result = doctor.check_storage(
        {
            "clickhouse": {
                "host": "local",
                "input_table": "db.input",
                "output_table": "db.output",
            }
        }
    )
    assert result["clickhouse_insert_grant"].startswith("PASS")
    assert "SELECT * FROM `db`.`input` LIMIT 0" in commands
    assert "CHECK GRANT INSERT ON `db`.`output`" in commands
    assert commands[-1] == "close"


def test_clickhouse_missing_insert_grant_is_error(monkeypatch):
    import clickhouse_connect

    client = SimpleNamespace(
        command=lambda q: "26.3.17" if q == "SELECT version()" else 0,
        query=lambda q: [],
        close=lambda: None,
    )
    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **kwargs: client)
    with pytest.raises(doctor.Fault) as caught:
        doctor.check_storage(
            {"clickhouse": {"host": "local", "output_table": "db.output"}}
        )
    assert caught.value.code == "E_CK"
    assert caught.value.stage == "clickhouse.insert_grant"


def s3_factory(
    *,
    endpoint="http://s3:9000",
    credentials=True,
    access="native-access",
    secret="native-secret",
    token=None,
):
    closed = []
    native = SimpleNamespace(
        meta=SimpleNamespace(endpoint_url=endpoint), close=lambda: closed.append(True)
    )
    frozen = SimpleNamespace(access_key=access, secret_key=secret, token=token)
    session = SimpleNamespace(
        get_credentials=lambda: (
            SimpleNamespace(get_frozen_credentials=lambda: frozen)
            if credentials
            else None
        )
    )
    sdk = SimpleNamespace(client=lambda *a, **k: native, Session=lambda: session)
    return sdk, closed


def test_s3_probe_must_match_native_endpoint():
    sdk, closed = s3_factory()
    probe = SimpleNamespace(meta=SimpleNamespace(endpoint_url="http://another:9000"))
    with pytest.raises(doctor.Fault) as caught:
        doctor.check_native_s3_configuration(sdk, probe, {}, None)
    assert caught.value.stage == "s3.native_endpoint"
    assert closed == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aws_access_key_id": "probe-access"},
        {"aws_secret_access_key": "probe-secret"},
        {"aws_session_token": "probe-token"},
    ],
)
def test_s3_explicit_probe_credentials_cannot_hide_missing_native_credentials(kwargs):
    sdk, _ = s3_factory()
    probe = SimpleNamespace(meta=SimpleNamespace(endpoint_url="http://s3:9000"))
    with pytest.raises(doctor.Fault) as caught:
        doctor.check_native_s3_configuration(sdk, probe, kwargs, None)
    assert caught.value.stage == "s3.native_credentials"
    report = json.dumps(doctor.failure(caught.value))
    assert "native-secret" not in report
    assert "probe-secret" not in report
    assert "probe-token" not in report


def test_s3_without_default_chain_is_clear_secret_failure():
    sdk, _ = s3_factory(credentials=False)
    probe = SimpleNamespace(meta=SimpleNamespace(endpoint_url="http://s3:9000"))
    with pytest.raises(doctor.Fault) as caught:
        doctor.check_native_s3_configuration(sdk, probe, {}, None)
    assert caught.value.code == "E_SECRET"
    assert caught.value.stage == "s3.native_credentials"


def test_s3_native_iam_or_env_chain_is_allowed():
    sdk, closed = s3_factory()
    probe = SimpleNamespace(meta=SimpleNamespace(endpoint_url="http://s3:9000/"))
    doctor.check_native_s3_configuration(
        sdk,
        probe,
        {
            "aws_access_key_id": "native-access",
            "aws_secret_access_key": "native-secret",
        },
        None,
    )
    assert closed == [True]
