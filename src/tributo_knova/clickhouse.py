"""Ray-native ClickHouse ingestion Binding for KnoVa workloads."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pyarrow as pa
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    residual_decisions,
)
from tributo.data.bindings._sql_shared import require_sql_table, resolve_sql_target
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingDescriptor,
    BindingKey,
    binding_stage,
)
from tributo.data.ingestion import (
    PhysicalSplitSummary,
    RayDataHandle,
    ReadHint,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import ScanKind, SourceCapability, SqlScan, SqlShardMode
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError

_BINDING_ID = "tributo.knova.ray.clickhouse"
_TABLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
_COLUMN_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUOTED_COLUMN_IDENTIFIER = re.compile(r"^`[A-Za-z_][A-Za-z0-9_]*`$")
_LOGGER = logging.getLogger(__name__)


def _qualified_table(database: str, table: str) -> str:
    value = table if "." in table else f"{database}.{table}"
    if _TABLE_IDENTIFIER.fullmatch(value) is None:
        raise JobConfigurationError(
            "ClickHouse table must be a table or database.table identifier"
        )
    return value


def _dsn(*, host: str, port: int, database: str, user: str, password: str) -> str:
    safe_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    authority = quote(user, safe="")
    if password:
        authority += f":{quote(password, safe='')}"
    return (
        f"clickhouse+http://{authority}@{safe_host}:{port}/{quote(database, safe='')}"
    )


@dataclass(frozen=True)
class _NativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    order_by: tuple[list[str], bool] | None


def _safe_order_columns(columns: tuple[str, ...]) -> list[str]:
    if not columns or any(
        _COLUMN_IDENTIFIER.fullmatch(value) is None for value in columns
    ):
        raise JobConfigurationError(
            "ClickHouse order columns must be simple column identifiers"
        )
    return list(columns)


def _simple_sorting_key_columns(expression: object) -> list[str] | None:
    """Parse a ClickHouse sorting key only when it is a plain column tuple.

    ``system.tables.sorting_key`` preserves composite-key order, unlike the
    physical column positions in ``system.columns``. Expressions such as
    ``toDate(event_time)`` deliberately fall back to a serial read because
    Ray documents ``order_by`` as a list of columns, not arbitrary SQL.
    """
    if not isinstance(expression, str) or not expression.strip():
        return None
    columns = [item.strip() for item in expression.split(",")]
    if not columns or any(
        _COLUMN_IDENTIFIER.fullmatch(item) is None
        and _QUOTED_COLUMN_IDENTIFIER.fullmatch(item) is None
        for item in columns
    ):
        return None
    return columns


def _discover_sorting_key(
    *, dsn: str, qualified_table: str
) -> tuple[list[str], bool] | None:
    """Resolve AUTO sharding from ClickHouse's physical sorting-key metadata.

    Ray can estimate rows, bytes, and schema, but its ClickHouse datasource does
    not discover a deterministic ``order_by`` key. Without one Ray intentionally
    collapses the read to one task. Keep that connector-specific lookup here so
    both training and inference retain the public Tributo ingestion contract.
    """
    import clickhouse_connect

    database, table = qualified_table.split(".", 1)
    client = None
    try:
        client = clickhouse_connect.get_client(dsn=dsn)
        result = client.query(
            "SELECT sorting_key FROM system.tables "
            "WHERE database = {database:String} AND name = {table:String} LIMIT 1",
            parameters={"database": database, "table": table},
        )
        if not result.result_rows:
            return None
        columns = _simple_sorting_key_columns(result.result_rows[0][0])
        return (columns, False) if columns is not None else None
    except Exception:
        # Metadata discovery is an optimization. Ray's safe single-task fallback
        # remains correct when the server does not expose system.tables.
        return None
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()


class RayClickHouseBinding:
    """Compile a structured ClickHouse table read through Ray Data itself."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "clickhouse")
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _NativePlan:
        import ray.data

        target = resolve_sql_target(plan, request.runtime_options)
        qualified_table = _qualified_table(target.database, target.table)
        dsn = _dsn(
            host=target.host,
            port=target.port,
            database=target.database,
            user=target.username,
            password=target.password,
        )
        target_parallelism = (
            plan.sharding.target_partitions or request.read_options.target_parallelism
        )
        settings: dict[str, Any] = {}
        if request.read_options.batch_size is not None:
            settings["max_block_size"] = request.read_options.batch_size

        order_by: tuple[list[str], bool] | None = None
        if plan.sharding.mode is SqlShardMode.PARALLEL:
            order_by = (_safe_order_columns(plan.sharding.columns), False)
        elif plan.sharding.mode is SqlShardMode.AUTO:
            order_by = _discover_sorting_key(
                dsn=dsn,
                qualified_table=qualified_table,
            )
        if order_by is None:
            _LOGGER.warning(
                "ClickHouse table %s has no usable simple sorting key for Ray "
                "read_clickhouse(order_by=...); Ray will fall back to one read "
                "task, so a large result can exhaust memory on a single Ray "
                "worker node. Define a simple ClickHouse ORDER BY key or configure "
                "explicit sharding columns.",
                qualified_table,
            )

        dataset = ray.data.read_clickhouse(
            table=qualified_table,
            dsn=dsn,
            columns=list(target.columns) or None,
            order_by=order_by,
            client_settings=settings or None,
            concurrency=request.read_options.concurrency,
            override_num_blocks=target_parallelism,
        )
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _NativePlan(dataset, schema, transforms, order_by)

    @staticmethod
    def _wrap(
        native_plan: _NativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        transformed = apply_pipeline_to_ray_ds(
            native_plan.transforms, native_plan.dataset
        )
        output_schema = (
            native_plan.transforms.steps[-1].output_schema
            if native_plan.transforms.steps
            else native_plan.input_schema
        )
        return BindingCompilation(
            handle=RayDataHandle(transformed),
            engine_version=importlib.metadata.version("ray"),
            reader_api="ray.data.read_clickhouse",
            transport_id="clickhouse_http",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail=(
                    "ClickHouse ordered read tasks are delegated to Ray Data"
                    if native_plan.order_by is not None
                    else "ClickHouse read uses Ray Data's safe single-task fallback"
                ),
            ),
            diagnostics=(
                "database metadata I/O was used for schema inference; "
                + (
                    "an order key enables Ray parallel read tasks"
                    if native_plan.order_by is not None
                    else "no safe order key was available for parallel reads"
                ),
            ),
        )


def clickhouse_binding_descriptor() -> BindingDescriptor:
    """Return the descriptor discovered by Tributo's Binding registry."""
    return BindingDescriptor(
        key=BindingKey(
            "tributo.ray_data",
            ScanKind.SQL,
            "clickhouse",
            _BINDING_ID,
        ),
        factory=RayClickHouseBinding,
        capabilities=frozenset({SourceCapability.PROJECTION}),
        distribution_name="tributo-knova",
        distribution_version=importlib.metadata.version("tributo-knova"),
        engine_version_spec="==2.55.1",
        dependency_distributions=("clickhouse-connect",),
        supported_read_hints=frozenset(
            {
                ReadHint.TARGET_PARALLELISM,
                ReadHint.BATCH_SIZE,
                ReadHint.CONCURRENCY,
            }
        ),
        install_hint="pip install tributo-knova",
    )


__all__ = [
    "RayClickHouseBinding",
    "clickhouse_binding_descriptor",
]
