"""Ray-native ClickHouse ingestion Binding for KnoVa workloads."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import pyarrow as pa
from tributo.data.bindings._shared import canonical_engine_schema
from tributo.data.bindings._sql_shared import require_sql_table, resolve_sql_target
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingDescriptor,
    BindingKey,
    binding_stage,
    classify_transform_decisions,
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
from tributo.data.transform_ir import FilterEq, TransformPipeline
from tributo.exceptions import JobConfigurationError

_BINDING_ID = "tributo.knova.ray.clickhouse"
_TABLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
_COLUMN_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUOTED_COLUMN_IDENTIFIER = re.compile(r"^`[A-Za-z_][A-Za-z0-9_]*`$")
_INTEGER_CLICKHOUSE_TYPE = re.compile(r"^U?Int(?:8|16|32|64|128|256)$")
_DEFAULT_BATCH_ROWS = 65_536
_DEFAULT_BATCH_BYTES = 64 * 1024 * 1024
_DEFAULT_TARGET_TASKS = 8
_DEFAULT_MAX_TASKS = 256
_MAX_TASKS = 1_024
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
    reader_api: str
    split_detail: str
    diagnostic: str
    pushed_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class _TableLayout:
    engine: str
    sorting_columns: tuple[str, ...]
    active_partitions: int | None
    first_order_type: str | None


@dataclass(frozen=True)
class _ReadStrategy:
    reader: Literal["ray", "ray_clickhouse"]
    split: Literal["single", "partition", "range"]
    range_column: str | None


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


def _discover_table_layout(
    *,
    dsn: str,
    qualified_table: str,
    explicit_order_columns: tuple[str, ...] = (),
) -> _TableLayout | None:
    """Read only the metadata needed to choose a safe physical split."""
    import clickhouse_connect

    database, table = qualified_table.split(".", 1)
    client = None
    try:
        client = clickhouse_connect.get_client(dsn=dsn)
        result = client.query(
            "SELECT engine, sorting_key, partition_key FROM system.tables "
            "WHERE database = {database:String} AND name = {table:String} LIMIT 1",
            parameters={"database": database, "table": table},
        )
        if not result.result_rows:
            return None
        engine, sorting_key, partition_key = result.result_rows[0]
        sorting_columns = tuple(_simple_sorting_key_columns(sorting_key) or ())
        order_columns = explicit_order_columns or sorting_columns

        active_partitions: int | None = None
        if (
            isinstance(engine, str)
            and engine.endswith("MergeTree")
            and isinstance(partition_key, str)
            and partition_key.strip()
        ):
            try:
                partition_result = client.query(
                    "SELECT uniqExact(partition_id) FROM system.parts "
                    "WHERE active AND database = {database:String} "
                    "AND table = {table:String}",
                    parameters={"database": database, "table": table},
                )
                active_partitions = int(partition_result.first_row[0])
            except Exception:
                active_partitions = None

        first_order_type: str | None = None
        if order_columns:
            try:
                type_result = client.query(
                    "SELECT type FROM system.columns "
                    "WHERE database = {database:String} AND table = {table:String} "
                    "AND name = {column:String} LIMIT 1",
                    parameters={
                        "database": database,
                        "table": table,
                        "column": order_columns[0].strip("`"),
                    },
                )
                if type_result.result_rows:
                    first_order_type = str(type_result.result_rows[0][0])
            except Exception:
                first_order_type = None

        return _TableLayout(
            engine=str(engine),
            sorting_columns=sorting_columns,
            active_partitions=active_partitions,
            first_order_type=first_order_type,
        )
    except Exception:
        # Metadata discovery is an optimization. The native reader fallback
        # remains correct when the account cannot inspect system tables.
        return None
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()


def _is_integer_type(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip()
    if normalized.startswith("Nullable(") and normalized.endswith(")"):
        normalized = normalized[9:-1]
    return _INTEGER_CLICKHOUSE_TYPE.fullmatch(normalized) is not None


def _leading_filter_pushdown(
    pipeline: TransformPipeline,
) -> tuple[str | None, dict[str, Any], tuple[int, ...], TransformPipeline]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {}
    pushed = 0
    for ordinal, step in enumerate(pipeline.steps):
        if not isinstance(step, FilterEq):
            break
        column = step.column
        if _COLUMN_IDENTIFIER.fullmatch(column) is None:
            break
        parameter = f"knova_filter_{ordinal}"
        clauses.append(f"`{column}` = %({parameter})s")
        parameters[parameter] = step.value
        pushed += 1
    predicate = " AND ".join(clauses) or None
    residual = TransformPipeline(steps=pipeline.steps[pushed:])
    return predicate, parameters, tuple(range(pushed)), residual


def _read_strategy(
    *,
    layout: _TableLayout | None,
    order_columns: tuple[str, ...],
    has_filters: bool,
    target_parallelism: int | None,
) -> _ReadStrategy:
    has_partitions = layout is not None and (
        layout.active_partitions is not None and layout.active_partitions > 1
    )
    has_integer_range = (
        layout is not None
        and order_columns
        and _is_integer_type(layout.first_order_type)
    )
    if has_partitions and (
        not has_integer_range
        or target_parallelism is None
        or layout.active_partitions >= target_parallelism
    ):
        return _ReadStrategy("ray_clickhouse", "partition", None)
    if has_integer_range:
        return _ReadStrategy(
            "ray_clickhouse",
            "range",
            order_columns[0].strip("`"),
        )
    if has_partitions:
        return _ReadStrategy("ray_clickhouse", "partition", None)
    if has_filters and layout is not None and (
        layout.engine.endswith("MergeTree") or layout.engine in {"View", "Distributed"}
    ):
        return _ReadStrategy("ray_clickhouse", "single", None)
    if order_columns:
        return _ReadStrategy("ray", "single", None)
    if layout is not None and (
        layout.engine.endswith("MergeTree") or layout.engine in {"View", "Distributed"}
    ):
        return _ReadStrategy("ray_clickhouse", "single", None)
    return _ReadStrategy("ray", "single", None)


def _task_limits(value: int | None) -> tuple[int, int]:
    requested = value or _DEFAULT_TARGET_TASKS
    target = min(requested, _MAX_TASKS)
    if requested > _MAX_TASKS:
        _LOGGER.warning(
            "ClickHouse target parallelism %s exceeds ray-clickhouse's safe cap; "
            "using %s read tasks",
            requested,
            target,
        )
    return target, max(_DEFAULT_MAX_TASKS, target)


class RayClickHouseBinding:
    """Compile a structured ClickHouse table read through Ray Data itself."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "clickhouse")
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("classify_transforms"):
            decisions = classify_transform_decisions(
                request.transforms,
                {ordinal: "exact" for ordinal in native_plan.pushed_ordinals},
            )
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _NativePlan:
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
        explicit_order_columns: tuple[str, ...] = ()
        if plan.sharding.mode is SqlShardMode.PARALLEL:
            explicit_order_columns = tuple(_safe_order_columns(plan.sharding.columns))
        layout = _discover_table_layout(
            dsn=dsn,
            qualified_table=qualified_table,
            explicit_order_columns=explicit_order_columns,
        )
        order_columns = explicit_order_columns or (
            layout.sorting_columns if layout is not None else ()
        )
        predicate, parameters, pushed_ordinals, residual = _leading_filter_pushdown(
            request.transforms
        )
        strategy = _read_strategy(
            layout=layout,
            order_columns=order_columns,
            has_filters=predicate is not None,
            target_parallelism=target_parallelism,
        )

        if strategy.reader == "ray_clickhouse":
            from ray_clickhouse import read_clickhouse

            database, table = qualified_table.split(".", 1)
            target_tasks, max_tasks = _task_limits(target_parallelism)
            dataset = read_clickhouse(
                host=target.host,
                port=target.port,
                database=database,
                table=table,
                username=target.username,
                password=target.password,
                columns=list(target.columns) or None,
                filter=predicate,
                query_parameters=parameters or None,
                split=strategy.split,
                range_column=strategy.range_column,
                discovery_policy="single",
                batch_rows=request.read_options.batch_size or _DEFAULT_BATCH_ROWS,
                batch_bytes=(
                    request.read_options.target_split_size_bytes
                    or _DEFAULT_BATCH_BYTES
                ),
                target_tasks=target_tasks,
                max_tasks=max_tasks,
                concurrency=request.read_options.concurrency,
                override_num_blocks=target_tasks,
            )
            transforms_to_compile = residual
            reader_api = "ray_clickhouse.read_clickhouse"
            order_by = None
            if strategy.split == "partition":
                split_detail = (
                    "ClickHouse physical partitions are balanced across Ray read tasks"
                )
                diagnostic = (
                    "physical partition pruning avoids global ORDER BY/OFFSET scans"
                )
            elif strategy.split == "range":
                split_detail = (
                    f"ClickHouse integer ranges on {strategy.range_column!r} are "
                    "delegated to Ray read tasks"
                )
                diagnostic = (
                    "the first usable sorting column provides disjoint range reads; "
                    "remaining sorting columns do not affect row coverage"
                )
            else:
                split_detail = (
                    "ClickHouse read uses one bounded streaming Ray read task"
                )
                diagnostic = (
                    "no safe distributed split was available; row and byte bounds "
                    "limit each emitted block"
                )
        else:
            import ray.data

            order_by = (list(order_columns), False) if order_columns else None
            settings: dict[str, Any] = {}
            if request.read_options.batch_size is not None:
                settings["max_block_size"] = request.read_options.batch_size
            dataset = ray.data.read_clickhouse(
                table=qualified_table,
                dsn=dsn,
                columns=list(target.columns) or None,
                order_by=order_by,
                client_settings=settings or None,
                concurrency=request.read_options.concurrency,
                override_num_blocks=target_parallelism,
            )
            transforms_to_compile = request.transforms
            pushed_ordinals = ()
            reader_api = "ray.data.read_clickhouse"
            if order_by is not None:
                split_detail = "ClickHouse ordered read tasks are delegated to Ray Data"
                diagnostic = (
                    "a composite or non-integer order key preserves Ray's native "
                    "parallel OFFSET reader"
                )
            else:
                split_detail = "ClickHouse read uses Ray Data's single-task fallback"
                diagnostic = (
                    "ClickHouse metadata was unavailable for safe split planning"
                )

        if strategy.split == "single" and order_by is None:
            _LOGGER.warning(
                "ClickHouse table %s has no safe distributed partition or integer "
                "range split; the read will use one Ray task. Emitted blocks are "
                "bounded when ray-clickhouse is available, but one worker still "
                "carries the scan and can become a throughput or memory-pressure "
                "hotspot. Define physical partitions or an integer sorting key.",
                qualified_table,
            )
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            transforms_to_compile, TransformBackend.RAY, schema
        )
        return _NativePlan(
            dataset=dataset,
            input_schema=schema,
            transforms=transforms,
            reader_api=reader_api,
            split_detail=split_detail,
            diagnostic=diagnostic,
            pushed_ordinals=pushed_ordinals,
        )

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
            reader_api=native_plan.reader_api,
            transport_id="clickhouse_http",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail=native_plan.split_detail,
            ),
            diagnostics=(
                "database metadata I/O was used for schema inference and split "
                f"planning; {native_plan.diagnostic}",
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
        capabilities=frozenset(
            {SourceCapability.PROJECTION, SourceCapability.PREDICATE_PUSHDOWN}
        ),
        distribution_name="tributo-knova",
        distribution_version=importlib.metadata.version("tributo-knova"),
        engine_version_spec="==2.55.1",
        dependency_distributions=("clickhouse-connect", "ray-clickhouse"),
        supported_read_hints=frozenset(
            {
                ReadHint.TARGET_PARALLELISM,
                ReadHint.TARGET_SPLIT_SIZE_BYTES,
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
