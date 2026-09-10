from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from tributo.data.engine_binding import BindingCompileRequest
from tributo.data.ingestion import IngestionRuntimeContext, RayDataHandle, ReadOptions
from tributo.data.scan_plan import (
    ScanKind,
    SqlScan,
    SqlShardMode,
    SqlShardRequirement,
    SqlTableRead,
)
from tributo.data.transform_ir import TransformPipeline
from tributo.exceptions import JobConfigurationError

from tributo_knova.clickhouse import (
    RayClickHouseBinding,
    _dsn,
    _qualified_table,
    clickhouse_binding_descriptor,
)


class _Dataset:
    def __init__(self) -> None:
        self.schema_value = pa.schema(
            [pa.field("feature", pa.float64()), pa.field("label", pa.int64())]
        )

    def schema(self) -> pa.Schema:
        return self.schema_value


def _request(*, parallel: bool = False) -> BindingCompileRequest:
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
        transforms=TransformPipeline(),
        read_options=ReadOptions(batch_size=2048, concurrency=3),
        source_ref="a" * 64,
        runtime_context=IngestionRuntimeContext(),
    )


def test_descriptor_targets_public_ray_clickhouse_contract() -> None:
    descriptor = clickhouse_binding_descriptor()

    assert descriptor.key.engine_id == "tributo.ray_data"
    assert descriptor.key.scan_kind is ScanKind.SQL
    assert descriptor.key.connector_id == "clickhouse"
    assert descriptor.key.binding_id == "tributo.knova.ray.clickhouse"
    assert descriptor.factory is RayClickHouseBinding


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


def test_binding_delegates_structured_read_to_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    dataset = _Dataset()

    def read_clickhouse(**kwargs: Any) -> _Dataset:
        calls.append(kwargs)
        return dataset

    import ray.data

    monkeypatch.setattr(ray.data, "read_clickhouse", read_clickhouse)
    result = RayClickHouseBinding().compile(_request(parallel=True))

    assert isinstance(result.handle, RayDataHandle)
    assert result.handle.dataset is dataset
    assert result.reader_api == "ray.data.read_clickhouse"
    assert result.transport_id == "clickhouse_http"
    assert result.physical_splits.split_count is None
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


def test_binding_failure_does_not_include_clickhouse_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray.data

    monkeypatch.setattr(
        ray.data,
        "read_clickhouse",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("p@ss/word")),
    )

    with pytest.raises(Exception) as captured:
        RayClickHouseBinding().compile(_request())

    assert "p@ss/word" not in str(captured.value)
