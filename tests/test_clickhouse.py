from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa
import pytest
from tributo.data.engine_binding import BindingCompileRequest
from tributo.data.ingestion import (
    IngestionRuntimeContext,
    RayDataHandle,
    ReadHint,
    ReadOptions,
)
from tributo.data.scan_plan import (
    ScanKind,
    SourceCapability,
    SqlScan,
    SqlShardMode,
    SqlShardRequirement,
    SqlTableRead,
)
from tributo.data.transform_ir import FilterEq, TransformPipeline
from tributo.exceptions import JobConfigurationError

from tributo_knova.clickhouse import (
    RayClickHouseBinding,
    _discover_table_layout,
    _dsn,
    _qualified_table,
    _simple_sorting_key_columns,
    _TableLayout,
    clickhouse_binding_descriptor,
)


class _Dataset:
    def __init__(self) -> None:
        self.schema_value = pa.schema(
            [pa.field("feature", pa.float64()), pa.field("label", pa.int64())]
        )

    def schema(self) -> pa.Schema:
        return self.schema_value


def _layout(
    *,
    engine: str = "MergeTree",
    sorting_columns: tuple[str, ...] = ("feature",),
    active_partitions: int | None = 0,
    first_order_type: str | None = "Float64",
) -> _TableLayout:
    return _TableLayout(
        engine=engine,
        sorting_columns=sorting_columns,
        active_partitions=active_partitions,
        first_order_type=first_order_type,
    )


def _request(
    *,
    parallel: bool = False,
    transforms: TransformPipeline | None = None,
) -> BindingCompileRequest:
    return BindingCompileRequest(
        plan=SqlScan(
            provider_id="tributo.clickhouse",
            connector_id="clickhouse",
            target=SqlTableRead(
                table="training_features",
                schema="analytics",
                projection=("feature", "label"),
            ),
            sharding=SqlShardRequirement(
                mode=(SqlShardMode.PARALLEL if parallel else SqlShardMode.AUTO),
                columns=(("feature",) if parallel else ()),
                target_partitions=4,
            ),
        ),
        runtime_options={
            "host": "clickhouse",
            "port": 8123,
            "database": "analytics",
            "user": "reader@example.com",
            "password": "p@ss/word",
        },
        transforms=transforms or TransformPipeline(),
        read_options=ReadOptions(
            batch_size=2048,
            target_split_size_bytes=4 * 1024 * 1024,
            concurrency=3,
        ),
        source_ref="a" * 64,
        runtime_context=IngestionRuntimeContext(),
    )


def test_descriptor_declares_bounded_pushdown_contract() -> None:
    descriptor = clickhouse_binding_descriptor()

    assert descriptor.key.engine_id == "tributo.ray_data"
    assert descriptor.key.scan_kind is ScanKind.SQL
    assert descriptor.key.connector_id == "clickhouse"
    assert descriptor.key.binding_id == "tributo.knova.ray.clickhouse"
    assert descriptor.factory is RayClickHouseBinding
    assert SourceCapability.PREDICATE_PUSHDOWN in descriptor.capabilities
    assert ReadHint.TARGET_SPLIT_SIZE_BYTES in descriptor.supported_read_hints
    assert "ray-clickhouse" in descriptor.dependency_distributions


def test_dsn_escapes_credentials_and_ipv6() -> None:
    assert _dsn(
        host="2001:db8::1",
        port=8123,
        database="analytics",
        user="reader@example.com",
        password="p@ss/word",
    ) == (
        "clickhouse+http://reader%40example.com:p%40ss%2Fword@"
        "[2001:db8::1]:8123/analytics"
    )


@pytest.mark.parametrize("table", ["bad-name", "db.table.extra", "db;drop"])
def test_qualified_table_rejects_uncontrolled_identifiers(table: str) -> None:
    with pytest.raises(JobConfigurationError, match="table or database.table"):
        _qualified_table("analytics", table)


def test_non_integer_explicit_key_preserves_native_ray_parallel_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(),
    )
    monkeypatch.setattr(
        "ray.data.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )

    result = RayClickHouseBinding().compile(_request(parallel=True))

    assert isinstance(result.handle, RayDataHandle)
    assert result.handle.dataset is dataset
    assert result.reader_api == "ray.data.read_clickhouse"
    assert result.transport_id == "clickhouse_http"
    assert calls == [
        {
            "table": "analytics.training_features",
            "dsn": (
                "clickhouse+http://reader%40example.com:p%40ss%2Fword@"
                "clickhouse:8123/analytics"
            ),
            "columns": ["feature", "label"],
            "order_by": (["feature"], False),
            "client_settings": {"max_block_size": 2048},
            "concurrency": 3,
            "override_num_blocks": 4,
        }
    ]
    assert "native parallel OFFSET reader" in result.diagnostics[0]


def test_auto_integer_key_uses_disjoint_range_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(
            sorting_columns=("label", "feature"),
            first_order_type="UInt64",
        ),
    )
    monkeypatch.setattr(
        "ray_clickhouse.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )

    result = RayClickHouseBinding().compile(_request())

    assert result.reader_api == "ray_clickhouse.read_clickhouse"
    assert calls[0]["split"] == "range"
    assert calls[0]["range_column"] == "label"
    assert calls[0]["target_tasks"] == 4
    assert calls[0]["override_num_blocks"] == 4
    assert calls[0]["batch_rows"] == 2048
    assert calls[0]["batch_bytes"] == 4 * 1024 * 1024
    assert "remaining sorting columns" in result.diagnostics[0]


def test_physical_partitions_take_priority_over_range_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(
            sorting_columns=("label",),
            active_partitions=12,
            first_order_type="UInt64",
        ),
    )
    monkeypatch.setattr(
        "ray_clickhouse.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )

    result = RayClickHouseBinding().compile(_request())

    assert calls[0]["split"] == "partition"
    assert calls[0]["range_column"] is None
    assert "physical partitions" in result.physical_splits.detail
    assert "ORDER BY/OFFSET" in result.diagnostics[0]


def test_integer_range_is_used_when_physical_partitions_cannot_fill_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(
            sorting_columns=("label",),
            active_partitions=2,
            first_order_type="UInt64",
        ),
    )
    monkeypatch.setattr(
        "ray_clickhouse.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )

    RayClickHouseBinding().compile(_request())

    assert calls[0]["split"] == "range"
    assert calls[0]["range_column"] == "label"


def test_leading_equality_filters_are_pushed_with_bound_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(sorting_columns=(), first_order_type=None),
    )
    monkeypatch.setattr(
        "ray_clickhouse.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )
    transforms = TransformPipeline(steps=(FilterEq(column="label", value=1),))

    result = RayClickHouseBinding().compile(_request(transforms=transforms))

    assert calls[0]["split"] == "single"
    assert calls[0]["filter"] == "`label` = %(knova_filter_0)s"
    assert calls[0]["query_parameters"] == {"knova_filter_0": 1}
    assert result.transform_decisions[0].pushdown_level == "exact"
    assert result.transform_decisions[0].residual_required is False
    assert result.handle.dataset is dataset


def test_missing_safe_split_warns_about_single_worker_pressure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: _layout(sorting_columns=(), first_order_type=None),
    )
    monkeypatch.setattr(
        "ray_clickhouse.read_clickhouse",
        lambda **kwargs: calls.append(kwargs) or dataset,
    )

    with caplog.at_level(logging.WARNING, logger="tributo_knova.clickhouse"):
        result = RayClickHouseBinding().compile(_request())

    assert calls[0]["split"] == "single"
    assert "one worker" in caplog.text
    assert "memory-pressure hotspot" in caplog.text
    assert "bounded streaming" in result.physical_splits.detail


def test_table_layout_metadata_uses_bound_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, dict[str, str]]] = []

    class _Result:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.result_rows = rows
            self.first_row = rows[0] if rows else ()

    class _Client:
        def query(self, query: str, *, parameters: dict[str, str]) -> _Result:
            observed.append((query, parameters))
            if "system.tables" in query:
                return _Result(
                    [
                        (
                            "MergeTree",
                            "tenant_id, event_time, user_id",
                            "toYYYYMM(ts)",
                        )
                    ]
                )
            if "system.parts" in query:
                return _Result([(6,)])
            return _Result([("UInt64",)])

        def close(self) -> None:
            pass

    monkeypatch.setattr("clickhouse_connect.get_client", lambda **_kwargs: _Client())

    layout = _discover_table_layout(
        dsn="clickhouse+http://host:8123/analytics",
        qualified_table="analytics.training_features",
    )

    assert layout == _TableLayout(
        engine="MergeTree",
        sorting_columns=("tenant_id", "event_time", "user_id"),
        active_partitions=6,
        first_order_type="UInt64",
    )
    assert all(
        parameters["database"] == "analytics"
        and parameters["table"] == "training_features"
        for _query, parameters in observed
    )
    assert observed[-1][1]["column"] == "tenant_id"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("tenant_id, event_time, user_id", ["tenant_id", "event_time", "user_id"]),
        ("event_date, `table`, event_time", ["event_date", "`table`", "event_time"]),
        ("toDate(event_time), user_id", None),
        ("tuple()", None),
    ],
)
def test_sorting_key_parser_preserves_composite_key_order(
    expression: str,
    expected: list[str] | None,
) -> None:
    assert _simple_sorting_key_columns(expression) == expected


def test_binding_failure_does_not_include_clickhouse_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo_knova.clickhouse._discover_table_layout",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "ray.data.read_clickhouse",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("p@ss/word")),
    )

    with pytest.raises(Exception) as captured:
        RayClickHouseBinding().compile(_request())

    assert "p@ss/word" not in str(captured.value)
