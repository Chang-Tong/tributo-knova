"""Internal Ray/XGBoost runtime for the KnoVa training protocol adapter."""

from __future__ import annotations

import hashlib
import math
import queue as stdlib_queue
import tempfile
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from tributo.algorithms import (
    AlgorithmExecutionResult,
    AlgorithmRunResult,
    WorkerResources,
)
from tributo.data import IngestionGateway, IngestionRequest, RayDataHandle
from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig

_METRIC_NAMES = {
    "logloss": "loss",
    "mlogloss": "loss",
    "error": "accuracy",
    "merror": "accuracy",
    "aucpr": "average_precision",
}
_ERROR_METRICS = frozenset({"error", "merror"})


class _EvidenceCollector:
    def __init__(self) -> None:
        self._records: dict[int, dict[str, Any]] = {}

    def record(self, value: dict[str, Any]) -> None:
        self._records[int(value["rank"])] = dict(value)

    def snapshot(self) -> list[dict[str, Any]]:
        return [self._records[index] for index in sorted(self._records)]


def _metric_value(raw_name: str, value: object) -> float:
    result = float(value)
    if raw_name in _ERROR_METRICS:
        result = 1.0 - result
    if not math.isfinite(result):
        raise ValueError("XGBoost produced a non-finite metric")
    return result


def _metrics_item(
    round_number: int,
    total_rounds: int,
    evals_log: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, Any] | None:
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for scope, raw_metrics in evals_log.items():
        for raw_name, values in raw_metrics.items():
            if round_number > len(values):
                continue
            try:
                value = _metric_value(raw_name, values[round_number - 1])
            except (TypeError, ValueError):
                continue
            name = _METRIC_NAMES.get(raw_name, raw_name)
            if name not in by_name:
                by_name[name] = {"metric_name": name}
                order.append(name)
            target = "train" if scope == "train" else "eval"
            by_name[name][target] = value
    metrics = [by_name[name] for name in order]
    if not metrics:
        return None
    return {
        "current_round": round_number,
        "total_rounds": total_rounds,
        "progress_percent": round(round_number / total_rounds * 100.0, 1),
        "metrics": metrics,
    }


def _xgboost_train_loop(config: dict[str, Any]) -> None:
    """Train one synchronized booster and report live metrics from rank zero."""
    import json
    import os

    import ray
    import xgboost
    from ray import train
    from ray.train import Checkpoint

    feature_names = list(config["feature_names"])
    label_name = str(config["label_name"])
    train_frame = train.get_dataset_shard("train").materialize().to_pandas()
    if train_frame.empty:
        raise RuntimeError("XGBoost training shard is empty")
    dtrain = xgboost.DMatrix(
        train_frame[feature_names].to_numpy(),
        label=train_frame[label_name].to_numpy(),
    )
    evals: list[tuple[Any, str]] = [(dtrain, "train")]
    try:
        validation_shard = train.get_dataset_shard("validation")
    except KeyError:
        validation_shard = None
    if validation_shard is not None:
        validation_frame = validation_shard.materialize().to_pandas()
        if not validation_frame.empty:
            evals.append(
                (
                    xgboost.DMatrix(
                        validation_frame[feature_names].to_numpy(),
                        label=validation_frame[label_name].to_numpy(),
                    ),
                    "eval",
                )
            )

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    total_rounds = int(config["num_rounds"])
    interval = max(1, math.ceil(total_rounds / 100))
    metrics_queue = config["metrics_queue"]

    class _RoundMetricsCallback(xgboost.callback.TrainingCallback):
        def __init__(self) -> None:
            self.published: set[int] = set()
            self.last_round = 0
            self.last_log: dict[str, dict[str, list[float]]] = {}

        def after_iteration(
            self,
            model: xgboost.Booster,
            epoch: int,
            evals_log: dict[str, dict[str, list[float]]],
        ) -> bool:
            del model
            self.last_round = epoch + 1
            if evals_log:
                self.last_log = evals_log
            if rank == 0 and (
                self.last_round == 1
                or self.last_round % interval == 0
                or self.last_round == total_rounds
            ):
                item = _metrics_item(self.last_round, total_rounds, evals_log)
                if item is not None:
                    try:
                        metrics_queue.put_nowait(item)
                        self.published.add(self.last_round)
                    except Exception:
                        pass
            return False

        def after_training(self, model: xgboost.Booster) -> xgboost.Booster:
            if rank == 0 and self.last_round not in self.published:
                item = _metrics_item(self.last_round, total_rounds, self.last_log)
                if item is not None:
                    with suppress(Exception):
                        metrics_queue.put_nowait(item)
            return model

    histories: dict[str, dict[str, list[float]]] = {}
    booster = xgboost.train(
        dict(config["params"]),
        dtrain,
        num_boost_round=total_rounds,
        evals=evals,
        evals_result=histories,
        early_stopping_rounds=config.get("early_stopping_rounds"),
        verbose_eval=False,
        callbacks=[_RoundMetricsCallback()],
    )
    raw = bytes(booster.save_raw(raw_format="ubj"))
    digest = hashlib.sha256(raw).hexdigest()
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    ray.get(
        config["evidence_actor"].record.remote(
            {
                "worker_id": str(runtime.get_worker_id()),
                "node_id": str(runtime.get_node_id()),
                "rank": rank,
                "world_size": world_size,
                "rows_processed": int(train_frame.shape[0]),
                "model_state_digest": digest,
                "resources": {
                    "num_cpus": float(assigned.get("CPU", 0.0)),
                    "num_gpus": float(assigned.get("GPU", 0.0)),
                },
            }
        )
    )

    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-knova-xgboost-") as root:
            path = Path(root)
            booster.save_model(path / "model.ubj")
            (path / "feature_names.json").write_text(
                json.dumps(feature_names), encoding="utf-8"
            )
            train.report(
                {
                    "metric_history": histories,
                    "model_state_digest": digest,
                    "worker_pid": os.getpid(),
                },
                checkpoint=Checkpoint.from_directory(path),
            )
    else:
        train.report({"model_state_digest": digest})


class _MetricsPump:
    def __init__(self, metrics_queue: Any, reporter: Any, rounds: int) -> None:
        self._queue = metrics_queue
        self._reporter = reporter
        self._rounds = rounds
        self._stop = threading.Event()
        self._published: set[int] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="tributo-knova-training-metrics",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _publish(self, item: object) -> None:
        if not isinstance(item, Mapping):
            return
        current = item.get("current_round")
        metrics = item.get("metrics")
        if (
            not isinstance(current, int)
            or current < 1
            or current > self._rounds
            or current in self._published
            or not isinstance(metrics, list)
            or not metrics
        ):
            return
        self._reporter.publish("METRICS", dict(item), phase="EXECUTING")
        self._published.add(current)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._publish(self._queue.get(timeout=0.2))
            except stdlib_queue.Empty:
                continue
            except Exception:
                if not self._stop.is_set():
                    self._stop.wait(0.2)

    def finish(self, histories: Mapping[str, Mapping[str, list[float]]]) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        while True:
            try:
                self._publish(self._queue.get_nowait())
            except Exception:
                break

        actual_rounds = max(
            (
                len(values)
                for metrics in histories.values()
                for values in metrics.values()
            ),
            default=0,
        )
        for round_number in (1, actual_rounds):
            if round_number < 1 or round_number in self._published:
                continue
            item = _metrics_item(round_number, self._rounds, histories)
            if item is not None:
                self._publish(item)
        with suppress(Exception):
            self._queue.shutdown(force=True)


def _split_dataset(
    dataset: Any,
    training: Mapping[str, Any],
) -> tuple[Any, Any | None, Any | None, dict[str, int]]:
    total = int(dataset.count())
    train_ratio = float(training.get("train_ratio", 0.7))
    validation_ratio = float(training.get("validation_ratio", 0.0))
    test_ratio = float(training.get("test_ratio", 0.3))
    if total < 1:
        raise ValueError("training dataset is empty")
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in (train_ratio, validation_ratio, test_ratio)
    ) or not math.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("data_split ratios must be within [0,1] and sum to 1")
    if train_ratio <= 0:
        raise ValueError("data_split.train_ratio must be positive")

    strategy = str(training.get("split_strategy", "RANDOM")).upper()
    if strategy not in {"RANDOM", "TIME_ORDERED"}:
        raise ValueError("data_split.strategy must be RANDOM or TIME_ORDERED")
    source = (
        dataset.randomize_block_order(seed=int(training.get("seed", 42)))
        if strategy == "RANDOM"
        else dataset
    )
    if validation_ratio > 0 and test_ratio > 0:
        train_dataset, validation_dataset, test_dataset = source.split_proportionately(
            [train_ratio, validation_ratio]
        )
    elif validation_ratio > 0:
        train_dataset, validation_dataset = source.split_proportionately([train_ratio])
        test_dataset = None
    elif test_ratio > 0:
        train_dataset, test_dataset = source.split_proportionately([train_ratio])
        validation_dataset = None
    else:
        train_dataset, validation_dataset, test_dataset = source, None, None

    rows = {
        "total": total,
        "train": int(train_dataset.count()),
        "validation": (
            int(validation_dataset.count()) if validation_dataset is not None else 0
        ),
        "test": int(test_dataset.count()) if test_dataset is not None else 0,
    }
    if sum(rows[name] for name in ("train", "validation", "test")) != total:
        raise RuntimeError("training split did not preserve every input row")
    if any(
        ratio > 0 and rows[name] == 0
        for name, ratio in (
            ("train", train_ratio),
            ("validation", validation_ratio),
            ("test", test_ratio),
        )
    ):
        raise ValueError("training dataset is too small for the requested split")
    return train_dataset, validation_dataset, test_dataset, rows


def _ensure_worker_blocks(
    dataset: Any,
    *,
    worker_count: int,
    row_count: int,
    name: str,
) -> Any:
    """Ensure every distributed XGBoost worker receives a non-empty shard."""
    if row_count < worker_count:
        raise ValueError(
            f"{name} dataset has fewer rows than distributed XGBoost workers"
        )
    try:
        block_count = dataset.num_blocks()
    except NotImplementedError:
        dataset = dataset.materialize()
        block_count = dataset.num_blocks()
    if block_count < worker_count:
        dataset = dataset.repartition(
            worker_count,
            strict=True,
            shuffle=False,
        )
    return dataset


def _load_booster(checkpoint: Any, feature_names: tuple[str, ...]) -> Any:
    import xgboost

    if checkpoint is None:
        raise RuntimeError("XGBoost training returned no checkpoint")
    with checkpoint.as_directory() as directory:
        booster = xgboost.Booster()
        booster.load_model(Path(directory) / "model.ubj")
    # The official boosting package intentionally feeds NumPy arrays to the
    # distributed DMatrix so XGBoost workers do not perform a feature-name
    # collective during setup. Restore business names after the distributed
    # context has closed; export still records the positional model contract.
    booster.feature_names = list(feature_names)
    return booster


def _predict_batch(
    batch: Any,
    *,
    booster_raw: bytes,
    feature_names: list[str],
    label_name: str,
    task_type: str,
    num_class: int | None,
) -> Any:
    import pandas as pd
    import xgboost

    booster = xgboost.Booster()
    booster.load_model(bytearray(booster_raw))
    values = booster.predict(xgboost.DMatrix(batch[feature_names]))
    result: dict[str, Any] = {"label": batch[label_name].to_numpy()}
    if task_type == "MULTICLASS_CLASSIFICATION":
        classes = int(num_class or values.shape[1])
        for index in range(classes):
            result[f"probability_{index}"] = values[:, index]
    elif task_type == "BINARY_CLASSIFICATION":
        result["probability"] = values
    else:
        result["prediction"] = values
    return pd.DataFrame(result)


def _safe_metric(function: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        value = float(function(*args, **kwargs))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _evaluation(
    dataset: Any | None,
    booster: Any,
    *,
    feature_names: tuple[str, ...],
    label_name: str,
    task_type: str,
    num_class: int | None,
    artifacts: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    if dataset is None:
        return {}, {}
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    raw = bytes(booster.save_raw(raw_format="ubj"))
    predictions = dataset.map_batches(
        _predict_batch,
        batch_format="pandas",
        batch_size=65_536,
        fn_kwargs={
            "booster_raw": raw,
            "feature_names": list(feature_names),
            "label_name": label_name,
            "task_type": task_type,
            "num_class": num_class,
        },
    ).to_pandas(limit=None)
    y_true = predictions["label"].to_numpy()
    metrics: dict[str, float | None]
    details: dict[str, Any] = {}
    if task_type == "BINARY_CLASSIFICATION":
        probability = predictions["probability"].to_numpy()
        y_pred = (probability >= 0.5).astype(int)
        metrics = {
            "auc": _safe_metric(roc_auc_score, y_true, probability),
            "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
            "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
            "average_precision": _safe_metric(
                average_precision_score, y_true, probability
            ),
        }
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        details["confusion_matrix"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        if artifacts.get("roc_curve", False):
            fpr, tpr, _ = roc_curve(y_true, probability)
            indices = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)
            details["roc_curve"] = {
                "fpr": [float(fpr[index]) for index in indices],
                "tpr": [float(tpr[index]) for index in indices],
            }
        if artifacts.get("threshold_analysis", False):
            thresholds = np.linspace(0.05, 0.95, 19)
            precision_values: list[float] = []
            recall_values: list[float] = []
            f1_values: list[float] = []
            positive_rows: list[int] = []
            for threshold in thresholds:
                predicted = (probability >= threshold).astype(int)
                precision_values.append(
                    float(precision_score(y_true, predicted, zero_division=0))
                )
                recall_values.append(
                    float(recall_score(y_true, predicted, zero_division=0))
                )
                f1_values.append(float(f1_score(y_true, predicted, zero_division=0)))
                positive_rows.append(int(predicted.sum()))
            details["threshold_analysis"] = {
                "thresholds": [round(float(value), 2) for value in thresholds],
                "precision_values": precision_values,
                "recall_values": recall_values,
                "f1_values": f1_values,
                "predicted_positive_rows": positive_rows,
            }
    elif task_type == "MULTICLASS_CLASSIFICATION":
        probability_columns = [
            f"probability_{index}" for index in range(int(num_class or 0))
        ]
        probabilities = predictions[probability_columns].to_numpy()
        y_pred = probabilities.argmax(axis=1)
        metrics = {
            "auc": _safe_metric(
                roc_auc_score,
                y_true,
                probabilities,
                multi_class="ovr",
                average="weighted",
            ),
            "f1": _safe_metric(
                f1_score, y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision": _safe_metric(
                precision_score,
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall": _safe_metric(
                recall_score,
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        }
        labels = sorted(np.unique(y_true).tolist())
        details["confusion_matrix"] = {
            "labels": [str(value) for value in labels],
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }
    else:
        predicted = predictions["prediction"].to_numpy()
        mean_squared_error_value = _safe_metric(mean_squared_error, y_true, predicted)
        metrics = {
            "rmse": (
                math.sqrt(mean_squared_error_value)
                if mean_squared_error_value is not None
                else None
            ),
            "mae": _safe_metric(mean_absolute_error, y_true, predicted),
            "mape": _safe_metric(mean_absolute_percentage_error, y_true, predicted),
            "r2": _safe_metric(r2_score, y_true, predicted),
        }
    return (
        {name: value for name, value in metrics.items() if value is not None},
        details,
    )


def _feature_importance(
    booster: Any,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    scores = booster.get_score(importance_type="gain")
    by_name = {
        name: float(scores.get(name, scores.get(f"f{index}", 0.0)))
        for index, name in enumerate(feature_names)
    }
    ordered = sorted(feature_names, key=lambda name: (-by_name[name], name))
    return [
        {
            "rank": index,
            "model_feature_name": name,
            "importance_score": by_name[name],
        }
        for index, name in enumerate(ordered, start=1)
    ]


def _correlation_batch_stats(batch: Any, feature_names: list[str]) -> Any:
    """Reduce one Ray batch to pairwise Pearson sufficient statistics."""
    import numpy as np
    import pandas as pd

    width = len(feature_names)
    numeric = batch.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    valid = finite.astype(np.float64, copy=False)
    clean = np.where(finite, values, 0.0)

    pair_count = valid.T @ valid
    pair_sum = clean.T @ valid
    pair_square_sum = np.square(clean).T @ valid
    pair_cross_sum = clean.T @ clean

    def _flat(matrix: Any) -> list[float]:
        return np.asarray(matrix, dtype=np.float64).reshape(width * width).tolist()

    return pd.DataFrame(
        {
            "_corr_count": [_flat(pair_count)],
            "_corr_sum": [_flat(pair_sum)],
            "_corr_square_sum": [_flat(pair_square_sum)],
            "_corr_cross_sum": [_flat(pair_cross_sum)],
        }
    )


def _finalize_correlation_stats(
    feature_names: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Merge compact worker statistics into a finite correlation matrix."""
    import numpy as np

    width = len(feature_names)
    if width == 0 or not rows:
        return None
    shape = (width, width)

    def _sum_field(name: str) -> Any:
        total = np.zeros(shape, dtype=np.float64)
        for row in rows:
            total += np.asarray(row[name], dtype=np.float64).reshape(shape)
        return total

    count = _sum_field("_corr_count")
    pair_sum = _sum_field("_corr_sum")
    square_sum = _sum_field("_corr_square_sum")
    cross_sum = _sum_field("_corr_cross_sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = cross_sum - pair_sum * pair_sum.T / count
        variance_x = square_sum - np.square(pair_sum) / count
        variance_y = variance_x.T
        denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=(count >= 2) & (denominator > 0.0),
        )

    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return {
        "feature_names": feature_names,
        "values": np.round(correlation, 6).tolist(),
    }


def _distributed_correlation(
    dataset: Any,
    feature_names: tuple[str, ...],
) -> dict[str, Any] | None:
    """Compute correlation on Ray workers without collecting source rows."""
    names = list(feature_names)
    rows = (
        dataset.select_columns(names)
        .map_batches(
            _correlation_batch_stats,
            batch_format="pandas",
            batch_size=65_536,
            fn_kwargs={"feature_names": names},
        )
        .take_all()
    )
    return _finalize_correlation_stats(names, rows)


def _export_bundle(
    booster: Any,
    *,
    feature_names: tuple[str, ...],
    bundle_uri: str,
    run_id: str,
) -> dict[str, Any]:
    import json

    import xgboost
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    learner = json.loads(booster.save_config())["learner"]
    objective = str(learner["objective"]["name"])
    classification = objective.startswith(("binary:", "multi:"))
    class_count = max(2, int(learner["learner_model_param"]["num_class"]))
    output_schema = (
        (
            CheckpointField(name="label", dtype="int64", shape=("batch",)),
            CheckpointField(
                name="probabilities",
                dtype="float32",
                shape=("batch", class_count),
            ),
        )
        if classification
        else (CheckpointField(name="prediction", dtype="float32", shape=("batch", 1)),)
    )
    checkpoint_contract = ExportCheckpointV1(
        trainer_type="xgboost",
        architecture_id="xgboost",
        input_schema=(
            CheckpointField(
                name="float_input",
                dtype="float32",
                shape=("batch", len(feature_names)),
            ),
        ),
        output_schema=output_schema,
        preprocessing={"type": "none"},
        task_type="classification" if classification else "regression",
        framework="xgboost",
        framework_version=xgboost.__version__,
        checkpoint_format_version=1,
    )
    raw = bytes(booster.save_raw(raw_format="ubj"))
    # onnxmltools only accepts XGBoost's positional ``f0``/``f1`` feature
    # convention.  Training with a pandas frame records the source column names
    # on the Booster, so export a semantically identical copy without that
    # converter-incompatible metadata.  The original names remain authoritative
    # in feature_schema and in the native artifact's feature_names.json.
    export_booster = xgboost.Booster()
    export_booster.load_model(bytearray(raw))
    export_booster.feature_names = None
    source = ExportSource(
        source_kind="xgboost_result",
        model_object=export_booster,
        architecture_id="xgboost",
        feature_schema={"feature_names": list(feature_names)},
        metadata={
            "framework": "xgboost",
            "framework_versions": {"xgboost": xgboost.__version__},
            "objective": objective,
            "producer_distribution": "tributo-knova",
        },
        source_fingerprint=hashlib.sha256(raw).hexdigest(),
        checkpoint_contract=checkpoint_contract,
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=bundle_uri,
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    exporter_id="official-xgboost-onnx-v1",
                ),
                ExportTarget(
                    name="native-model",
                    format="ubj",
                    exporter_id="official-xgboost-ubj-v1",
                ),
            ],
            roles={"inference": "onnx-model", "native": "native-model"},
        ),
    )
    return {
        "bundle_id": bundle.bundle_id,
        "bundle_uri": bundle.canonical_uri,
        "execution_id": bundle.execution_id,
        "manifest_sha256": bundle.manifest_sha256,
    }


def run_training(
    *,
    ingestion_request: IngestionRequest,
    feature_names: tuple[str, ...],
    label_name: str,
    algorithm_config: Mapping[str, Any],
    worker_count: int,
    resources: WorkerResources,
    run_id: str,
    reporter: Any,
    task_type: str,
    num_class: int | None,
    evaluation_artifacts: Mapping[str, Any],
) -> AlgorithmRunResult:
    """Run a split-aware XGBoost job using imported Tributo components."""
    import importlib.metadata

    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer
    from ray.util.queue import Queue

    gateway = IngestionGateway()
    opened = gateway.open(ingestion_request)
    metrics_queue: Any | None = None
    evidence_actor: Any | None = None
    pump: _MetricsPump | None = None
    histories: Mapping[str, Mapping[str, list[float]]] = {}
    try:
        if not isinstance(opened.handle, RayDataHandle):
            raise TypeError("training ingestion must return a RayDataHandle")
        dataset = opened.handle.dataset.materialize()
        training = algorithm_config.get("training", {})
        if not isinstance(training, Mapping):
            raise TypeError("training configuration must be an object")
        reporter.phase("DATA_SPLITTING")
        train_dataset, validation_dataset, test_dataset, rows = _split_dataset(
            dataset, training
        )
        train_dataset = _ensure_worker_blocks(
            train_dataset,
            worker_count=worker_count,
            row_count=rows["train"],
            name="train",
        )
        if validation_dataset is not None:
            validation_dataset = _ensure_worker_blocks(
                validation_dataset,
                worker_count=worker_count,
                row_count=rows["validation"],
                name="validation",
            )

        rounds = int(training.get("num_rounds", 100))
        metrics_queue = Queue(maxsize=256, actor_options={"num_cpus": 0})
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        evidence_actor = collector_type.remote()
        params = algorithm_config.get("model", {})
        if not isinstance(params, Mapping):
            raise TypeError("model configuration must be an object")
        model_params = dict(params)
        if isinstance(model_params.get("eval_metric"), tuple):
            model_params["eval_metric"] = list(model_params["eval_metric"])
        train_loop_config = {
            "feature_names": list(feature_names),
            "label_name": label_name,
            "params": model_params,
            "num_rounds": rounds,
            "early_stopping_rounds": training.get("early_stopping_rounds"),
            "metrics_queue": metrics_queue,
            "evidence_actor": evidence_actor,
        }
        datasets = {"train": train_dataset}
        datasets_to_split = ["train"]
        if validation_dataset is not None:
            datasets["validation"] = validation_dataset
            datasets_to_split.append("validation")
        ray_config = algorithm_config.get("ray", {})
        if not isinstance(ray_config, Mapping):
            raise TypeError("ray configuration must be an object")
        trainer = XGBoostTrainer(
            train_loop_per_worker=_xgboost_train_loop,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=worker_count,
                use_gpu=resources.num_gpus > 0,
                placement_strategy="SPREAD",
                resources_per_worker={
                    "CPU": resources.num_cpus,
                    "GPU": resources.num_gpus,
                    **dict(resources.custom),
                },
            ),
            datasets=datasets,
            dataset_config=CompleteCoverageDataConfig(
                datasets_to_split=datasets_to_split
            ),
            run_config=RunConfig(
                name=f"tributo-knova-{run_id}",
                storage_path=str(ray_config["storage_path"]),
            ),
        )
        reporter.phase("EXECUTING")
        pump = _MetricsPump(metrics_queue, reporter, rounds)
        pump.start()
        train_result = trainer.fit()
        raw_histories = train_result.metrics.get("metric_history", {})
        if isinstance(raw_histories, Mapping):
            histories = raw_histories
        pump.finish(histories)
        pump = None
        metrics_queue = None

        evidence = ray.get(evidence_actor.snapshot.remote())
        ray.kill(evidence_actor, no_restart=True)
        evidence_actor = None
        digests = {item.get("model_state_digest") for item in evidence}
        if (
            len(evidence) != worker_count
            or len(digests) != 1
            or sum(int(item.get("rows_processed", 0)) for item in evidence)
            != rows["train"]
        ):
            raise RuntimeError("distributed XGBoost worker evidence is incomplete")

        booster = _load_booster(train_result.checkpoint, feature_names)
        reporter.phase("MATERIALIZING")
        evaluation, details = _evaluation(
            test_dataset,
            booster,
            feature_names=feature_names,
            label_name=label_name,
            task_type=task_type,
            num_class=num_class,
            artifacts=evaluation_artifacts,
        )
        correlation_matrix = None
        if evaluation_artifacts.get("correlation_matrix", False):
            correlation_matrix = _distributed_correlation(
                train_dataset,
                feature_names,
            )
            if correlation_matrix is None:
                raise RuntimeError(
                    "requested feature correlation matrix could not be computed"
                )
        output = algorithm_config.get("output", {})
        if not isinstance(output, Mapping):
            raise TypeError("output configuration must be an object")
        outputs = _export_bundle(
            booster,
            feature_names=feature_names,
            bundle_uri=str(output["bundle_uri"]),
            run_id=run_id,
        )
        internal_metrics: dict[str, Any] = {
            "evaluation": evaluation,
            "evaluation_details": details,
            "feature_importance": _feature_importance(booster, feature_names),
            "correlation_matrix": correlation_matrix,
            "sample_rows": rows,
        }
        return AlgorithmRunResult(
            run_id=run_id,
            plan_id=hashlib.sha256(f"tributo-knova:{run_id}".encode()).hexdigest(),
            execution=AlgorithmExecutionResult(
                status="succeeded",
                metrics=internal_metrics,
                outputs=outputs,
            ),
            actual_versions={
                "ray": importlib.metadata.version("ray"),
                "xgboost": importlib.metadata.version("xgboost"),
            },
            input_provenance={"dataset_ref": opened.receipt.dataset_ref},
            worker_metadata={"workers": evidence},
        )
    finally:
        if pump is not None:
            pump.finish(histories)
        elif metrics_queue is not None:
            with suppress(Exception):
                metrics_queue.shutdown(force=True)
        if evidence_actor is not None:
            with suppress(Exception):
                ray.kill(evidence_actor, no_restart=True)
        opened.close()


__all__ = ["run_training"]
