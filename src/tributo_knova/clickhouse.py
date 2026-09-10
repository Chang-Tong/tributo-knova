"""Ray-native ClickHouse ingestion Binding for KnoVa workloads."""

from __future__ import annotations

import importlib.metadata
import re
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
_TABLE_IDENTIFIER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$"
)


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
        f"clickhouse+http://{authority}@{safe_host}:{port}/"
        f"{quote(database, safe='')}"
    )


@dataclass(frozen=True)
class _NativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


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
        target_parallelism = (
            plan.sharding.target_partitions or request.read_options.target_parallelism
        )
        settings: dict[str, Any] = {}
        if request.read_options.batch_size is not None:
            settings["max_block_size"] = request.read_options.batch_size

        order_by: tuple[list[str], bool] | None = None
        if plan.sharding.mode is SqlShardMode.PARALLEL:
            if not plan.sharding.columns:
                raise JobConfigurationError(
                    "parallel ClickHouse reads require an order/shard column"
                )
            order_by = (list(plan.sharding.columns), False)

        dataset = ray.data.read_clickhouse(
            table=_qualified_table(target.database, target.table),
            dsn=_dsn(
                host=target.host,
                port=target.port,
                database=target.database,
                user=target.username,
                password=target.password,
            ),
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
        return _NativePlan(dataset, schema, transforms)

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
                detail="ClickHouse read tasks are delegated to Ray Data",
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
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
