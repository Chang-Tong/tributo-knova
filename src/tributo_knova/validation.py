"""Credential-safe structural validation of ai-model-design v2 requests.

Unknown fields, including opaque extensions, are intentionally ignored.  This
module validates the protocol; execution adapters gate their own capabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class ProtocolValidationError(ValueError):
    """A field-only diagnostic safe to expose in a protocol failure."""


def _fail(field: str, rule: str) -> None:
    raise ProtocolValidationError(f"{field} {rule}")


def _obj(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "must be an object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(field, "must be a non-empty string")
    return value


def _enum(value: object, choices: set[str], field: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(field, "has an unsupported value")
    return value


def _int(value: object, field: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(field, f"must be an integer >= {minimum}")
    return value


def _number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _fail(field, "must be a finite number")
    return float(value)


def _datasource(value: object, field: str) -> None:
    obj = _obj(value, field)
    _enum(obj.get("type"), {"CLICKHOUSE", "DORIS", "HIVE", "JDBC"}, field + ".type")
    for name in ("host", "database_name"):
        _text(obj.get(name), field + "." + name)
    if _int(obj.get("port"), field + ".port") > 65535:
        _fail(field + ".port", "must be <= 65535")
    if obj.get("properties") is not None:
        _obj(obj["properties"], field + ".properties")
    for name in ("username", "password", "credential_ref", "connection_string"):
        if obj.get(name) is not None and not isinstance(obj[name], str):
            _fail(field + "." + name, "must be a string")


def _topology(value: Mapping[str, Any], field: str = "") -> set[str]:
    tables = value.get("tables")
    if not isinstance(tables, list) or not tables:
        _fail(field + "tables", "must be a non-empty array")
    aliases: set[str] = set()
    primary = None
    for raw in tables:
        table = _obj(raw, field + "tables[]")
        alias = _text(table.get("table_alias"), field + "tables[].table_alias")
        if alias in aliases:
            _fail(field + "tables[].table_alias", "must be unique")
        aliases.add(alias)
        for name in ("database_name", "table_name"):
            _text(table.get(name), field + "tables[]." + name)
        role = _enum(table.get("role"), {"PRIMARY", "JOINED"}, field + "tables[].role")
        if role == "PRIMARY":
            if primary is not None:
                _fail(field + "tables", "must have exactly one PRIMARY")
            primary = alias
    if primary is None:
        _fail(field + "tables", "must have exactly one PRIMARY")
    relations = value.get("relations", [])
    if not isinstance(relations, list) or (len(aliases) > 1 and not relations):
        _fail(field + "relations", "must connect the declared tables")
    connected = {primary}
    for raw in relations:
        relation = _obj(raw, field + "relations[]")
        left, right = relation.get("left_alias"), relation.get("right_alias")
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left not in connected
            or right not in aliases
            or right in connected
        ):
            _fail(
                field + "relations",
                "must expand once from PRIMARY using declared aliases",
            )
        _enum(
            relation.get("join_type"),
            {"INNER", "LEFT"},
            field + "relations[].join_type",
        )
        conditions = relation.get("on")
        if not isinstance(conditions, list) or not conditions:
            _fail(field + "relations[].on", "must be non-empty")
        for raw_condition in conditions:
            condition = _obj(raw_condition, field + "relations[].on[]")
            for name in ("left_column", "right_column"):
                _text(condition.get(name), field + "relations[].on[]." + name)
        connected.add(right)
    if connected != aliases:
        _fail(field + "relations", "must connect all tables to PRIMARY")
    return aliases


def _column(value: object, aliases: set[str], field: str) -> str:
    obj = _obj(value, field)
    origin = _obj(obj.get("origin"), field + ".origin")
    alias = _text(origin.get("table_alias"), field + ".origin.table_alias")
    if alias not in aliases:
        _fail(field + ".origin.table_alias", "must reference tables")
    for name in ("column_name", "column_type"):
        _text(origin.get(name), field + ".origin." + name)
    return _text(obj.get("result_column"), field + ".result_column")


def _features(value: object, aliases: set[str], field: str = "features") -> set[str]:
    if not isinstance(value, list) or not value:
        _fail(field, "must be a non-empty array")
    ids, columns = set(), set()
    for raw in value:
        feature = _obj(raw, field + "[]")
        feature_id = _text(feature.get("feature_id"), field + "[].feature_id")
        column = _column(feature, aliases, field + "[]")
        if feature_id in ids or column in columns:
            _fail(field, "feature_id and result_column must each be unique")
        ids.add(feature_id)
        columns.add(column)
        _text(feature.get("display_name"), field + "[].display_name")
        _enum(
            feature.get("feature_role"),
            {"REGULAR", "TEMPORAL_KEY", "KNOWN_FUTURE"},
            field + "[].feature_role",
        )
        treatment = feature.get("treatment")
        if treatment is not None:
            _obj(treatment, field + "[].treatment")
    return columns


def _label_mapping(value: object, field: str) -> Mapping[str, int]:
    if value is None:
        return {}
    mapping = _obj(value, field)
    codes = tuple(mapping.values())
    if any(not isinstance(code, int) or isinstance(code, bool) for code in codes):
        _fail(field, "values must be non-boolean integers")
    if len(set(codes)) != len(codes) or set(codes) != set(range(len(codes))):
        _fail(field, "must uniquely encode labels with consecutive integers from zero")
    return mapping


def _query(value: object, field: str) -> None:
    query = _obj(value, field)
    _text(query.get("sql"), field + ".sql")
    params = _obj(query.get("params", {}), field + ".params")
    if any(not isinstance(v, str) for v in params.values()):
        _fail(field + ".params", "values must be strings")


_CATEGORIES = {
    "CLASSIFICATION": {"BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"},
    "REGRESSION": {"REGRESSION"},
    "TIMESERIES": {"TIMESERIES_FORECAST"},
    "CLUSTERING": {"CLUSTERING"},
}
_XGB_SEARCH = {
    "n_estimators": ("INT", 10, 1000),
    "max_depth": ("INT", 1, 15),
    "learning_rate": ("FLOAT", 0.001, 1.0),
    "subsample": ("FLOAT", 0.1, 1.0),
    "colsample_bytree": ("FLOAT", 0.1, 1.0),
    "min_child_weight": ("INT", 0, 10),
    "gamma": ("FLOAT", 0.0, 5.0),
    "reg_alpha": ("FLOAT", 0.0, 10.0),
    "reg_lambda": ("FLOAT", 0.0, 10.0),
}


def _tuning(value: object, algorithm: Mapping[str, Any]) -> None:
    tuning = _obj(value, "tuning")
    mode = _enum(tuning.get("mode"), {"MANUAL", "AUTO"}, "tuning.mode")
    if mode == "MANUAL":
        if tuning.get("auto_config") is not None:
            _fail("tuning.auto_config", "must be null in MANUAL mode")
        return
    auto = _obj(tuning.get("auto_config"), "tuning.auto_config")
    sampler = _obj(auto.get("sampler"), "tuning.auto_config.sampler")
    sampler_type = _enum(
        sampler.get("type"),
        {"GRID", "RANDOM", "BAYESIAN", "CMAES"},
        "tuning.auto_config.sampler.type",
    )
    budget = _obj(auto.get("budget"), "tuning.auto_config.budget")
    _int(budget.get("max_trials"), "tuning.auto_config.budget.max_trials")
    for name in ("patience", "parallelism"):
        if budget.get(name) is not None:
            _int(budget[name], "tuning.auto_config.budget." + name)
    space = _obj(auto.get("search_space"), "tuning.auto_config.search_space")
    if not space:
        _fail("tuning.auto_config.search_space", "must be non-empty")
    for name, raw in space.items():
        domain = _obj(raw, "tuning.auto_config.search_space domain")
        kind = _enum(
            domain.get("type"),
            {"INT", "FLOAT", "ENUM", "BOOL"},
            "search_space domain.type",
        )
        bounds = (
            _XGB_SEARCH.get(name)
            if algorithm.get("algorithm_key") in {"xgboost", "xgboost_reg"}
            else None
        )
        if (
            algorithm.get("algorithm_key") in {"xgboost", "xgboost_reg"}
            and bounds is None
        ):
            _fail(
                "tuning.auto_config.search_space",
                "contains a parameter outside the algorithm catalog",
            )
        if bounds and kind != bounds[0]:
            _fail("search_space domain.type", "must match the algorithm catalog")
        if kind in {"INT", "FLOAT"}:
            low, high = (
                _number(domain.get(x), "search_space domain." + x)
                for x in ("min", "max")
            )
            if low > high or (bounds and (low < bounds[1] or high > bounds[2])):
                _fail("search_space domain", "range must fit the algorithm catalog")
            if kind == "INT" and (not low.is_integer() or not high.is_integer()):
                _fail("search_space domain", "INT bounds must be integers")
            if (
                domain.get("step") is not None
                and _number(domain["step"], "search_space domain.step") <= 0
            ):
                _fail("search_space domain.step", "must be positive")
            if sampler_type == "GRID" and domain.get("step") is None:
                _fail("search_space domain.step", "is required for GRID")
            _enum(
                domain.get("scale", "LINEAR"),
                {"LINEAR", "LOG"},
                "search_space domain.scale",
            )
        elif not isinstance(domain.get("choices"), list) or not domain["choices"]:
            _fail("search_space domain.choices", "must be non-empty")
    scheduler = _obj(
        auto.get("scheduler") or {"type": "NONE"}, "tuning.auto_config.scheduler"
    )
    schedule_type = _enum(
        scheduler.get("type"),
        {"NONE", "MEDIAN", "ASHA", "HYPERBAND", "PBT"},
        "tuning.auto_config.scheduler.type",
    )
    if schedule_type in {"ASHA", "HYPERBAND"}:
        resource = scheduler.get("resource_param")
        if resource not in {"n_estimators", "data_fraction"} or resource in space:
            _fail(
                "tuning.auto_config.scheduler.resource_param",
                "must be an independent fidelity dimension",
            )
        lo = _number(scheduler.get("min_resource"), "scheduler.min_resource")
        hi = _number(scheduler.get("max_resource"), "scheduler.max_resource")
        if lo <= 0 or hi < lo:
            _fail(
                "tuning.auto_config.scheduler",
                "resource bounds must be positive and ordered",
            )
    if schedule_type == "PBT" and algorithm.get("algorithm_key") in {
        "xgboost",
        "xgboost_reg",
    }:
        _fail(
            "tuning.auto_config.scheduler", "PBT requires resumable training capability"
        )


def validate_training_payload(payload: Mapping[str, Any]) -> None:
    algorithm = _obj(payload.get("algorithm"), "algorithm")
    category = _enum(algorithm.get("category"), set(_CATEGORIES), "algorithm.category")
    key = _text(algorithm.get("algorithm_key"), "algorithm.algorithm_key")
    expected_family = {
        "xgboost": "CLASSIFICATION",
        "xgboost_reg": "REGRESSION",
        "random_forest": "CLASSIFICATION",
        "random_forest_reg": "REGRESSION",
        "logistic_regression": "CLASSIFICATION",
        "kmeans": "CLUSTERING",
    }.get(key)
    if expected_family is not None and category != expected_family:
        _fail("algorithm.category", "does not match algorithm_key")
    _obj(algorithm.get("hyper_params"), "algorithm.hyper_params")
    _datasource(payload.get("datasource"), "datasource")
    aliases = _topology(payload)
    columns = _features(payload.get("features"), aliases)
    query = _obj(payload.get("data_query"), "data_query")
    mode = _enum(
        query.get("mode"), {"DIRECT_QUERY", "TIMESERIES_PIVOT"}, "data_query.mode"
    )
    if mode == "DIRECT_QUERY":
        _query(query.get("query"), "data_query.query")
    else:
        for name in ("entity_sampling_query", "feature_query", "target_query"):
            _query(query.get(name), "data_query." + name)
        if not _obj(query.get("timeseries_pivot"), "data_query.timeseries_pivot"):
            _fail("data_query.timeseries_pivot", "must be non-empty")
    if (
        query.get("entity_key") is not None
        or payload.get("relations")
        or mode == "TIMESERIES_PIVOT"
    ):
        _column(query.get("entity_key"), aliases, "data_query.entity_key")
    target = payload.get("target")
    if category != "CLUSTERING":
        target = _obj(target, "target")
        _enum(
            target.get("task_type"), _CATEGORIES[category], "target.task_type/category"
        )
        if _column(target, aliases, "target") in columns:
            _fail("target.result_column", "must not also be a feature")
        _text(target.get("display_name"), "target.display_name")
        mapping = _label_mapping(target.get("label_mapping"), "target.label_mapping")
        if mapping:
            positive = target.get("positive_label_value")
            if positive is not None and (
                not isinstance(positive, (str, int))
                or isinstance(positive, bool)
                or str(positive) not in mapping
            ):
                _fail(
                    "target.positive_label_value", "must identify a label_mapping key"
                )
    sampling = _obj(payload.get("data_sampling", {}), "data_sampling")
    if (
        sampling.get("sample_ratio") is not None
        and _int(sampling["sample_ratio"], "data_sampling.sample_ratio") > 99
    ):
        _fail("data_sampling.sample_ratio", "must be within [1, 99]")
    if sampling.get("sample_limit") is not None:
        _int(sampling["sample_limit"], "data_sampling.sample_limit")
    balance = sampling.get("class_balance")
    if balance is not None:
        if category != "CLASSIFICATION":
            _fail(
                "data_sampling.class_balance",
                "must be null for non-classification tasks",
            )
        balance = _obj(balance, "data_sampling.class_balance")
        strategy = _enum(
            balance.get("strategy"),
            {"NONE", "UNDERSAMPLE", "CUSTOM_AMOUNT", "OVERSAMPLE_SMOTE"},
            "data_sampling.class_balance.strategy",
        )
        amounts = balance.get("class_amounts")
        if amounts is not None or strategy == "CUSTOM_AMOUNT":
            amounts = _obj(amounts, "data_sampling.class_balance.class_amounts")
            if not amounts:
                _fail("data_sampling.class_balance.class_amounts", "must be non-empty")
            for count in amounts.values():
                _int(count, "data_sampling.class_balance.class_amounts value")
        if strategy == "OVERSAMPLE_SMOTE":
            oversample = _obj(
                balance.get("oversample_config"),
                "data_sampling.class_balance.oversample_config",
            )
            _enum(
                oversample.get("method"),
                {"SMOTE", "RANDOM_OVERSAMPLE", "ADASYN", "BORDERLINE_SMOTE"},
                "oversample_config.method",
            )
        elif balance.get("oversample_config") is not None:
            _fail("data_sampling.class_balance.oversample_config", "must be null")
    split = _obj(payload.get("data_split", {}), "data_split")
    ratios = [
        _number(split.get(name, default), "data_split." + name)
        for name, default in (
            ("train_ratio", 0.7),
            ("validation_ratio", 0.15),
            ("test_ratio", 0.15),
        )
    ]
    if any(not 0 <= r <= 1 for r in ratios) or not math.isclose(
        sum(ratios), 1, abs_tol=1e-6, rel_tol=0
    ):
        _fail("data_split", "ratios must be within [0, 1] and sum to 1")
    strategy = _enum(
        split.get("strategy", "RANDOM"),
        {"RANDOM", "TIME_ORDERED"},
        "data_split.strategy",
    )
    stratify = split.get("stratify", False)
    if not isinstance(stratify, bool) or (
        stratify and (strategy == "TIME_ORDERED" or category != "CLASSIFICATION")
    ):
        _fail(
            "data_split.stratify",
            "must be false for TIME_ORDERED and non-classification tasks",
        )
    _tuning(payload.get("tuning", {"mode": "MANUAL"}), algorithm)
    resources = _obj(payload.get("resource_limits", {}), "resource_limits")
    for name in ("max_epochs", "early_stopping_patience", "max_training_time_seconds"):
        if resources.get(name) is not None:
            _int(resources[name], "resource_limits." + name)
    storage = _obj(payload.get("storage_context"), "storage_context")
    for name in ("type", "bucket", "prefix"):
        _text(storage.get(name), "storage_context." + name)
    prefix = storage["prefix"]
    if (
        prefix.startswith("/")
        or not prefix.endswith("/")
        or any(part in {".", "..", ""} for part in prefix[:-1].split("/"))
    ):
        _fail(
            "storage_context.prefix",
            "must be a relative directory ending in / without traversal",
        )


def validate_inference_payload(payload: Mapping[str, Any]) -> None:
    model = _obj(payload.get("model"), "model")
    for name in ("model_id", "version_id", "algorithm_key"):
        _text(model.get(name), "model." + name)
    category = _enum(
        model.get("algorithm_category"), set(_CATEGORIES), "model.algorithm_category"
    )
    _enum(model.get("task_type"), _CATEGORIES[category], "model.task_type/category")
    _label_mapping(model.get("label_mapping"), "model.label_mapping")
    _obj(model.get("storage"), "model.storage")
    if not _obj(model.get("model_artifacts"), "model.model_artifacts"):
        _fail("model.model_artifacts", "must be non-empty")
    inputs = _obj(payload.get("input"), "input")
    _datasource(inputs.get("datasource"), "input.datasource")
    aliases = _topology(inputs, "input.")
    _features(inputs.get("features"), aliases, "input.features")
    _column(inputs.get("entity_key"), aliases, "input.entity_key")
    _query(inputs.get("query"), "input.query")
    if inputs.get("expected_total_rows") is not None:
        _int(inputs["expected_total_rows"], "input.expected_total_rows", 0)
    output = _obj(payload.get("output"), "output")
    _datasource(output.get("datasource"), "output.datasource")
    _text(output.get("table_name"), "output.table_name")
    execution = _obj(payload.get("execution", {}), "execution")
    for name in (
        "batch_size",
        "max_execution_time_seconds",
        "progress_report_interval_rows",
    ):
        if execution.get(name) is not None:
            _int(execution[name], "execution." + name)
