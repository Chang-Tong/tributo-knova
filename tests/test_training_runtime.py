from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost

from tributo_knova import _training_runtime as runtime


class _Dataset:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.random_seed: int | None = None

    def count(self) -> int:
        return self.rows

    def randomize_block_order(self, seed: int) -> _Dataset:
        self.random_seed = seed
        return self

    def split_proportionately(self, proportions: list[float]) -> tuple[_Dataset, ...]:
        boundaries = [round(self.rows * value) for value in proportions]
        sizes: list[int] = []
        consumed = 0
        for boundary in boundaries:
            sizes.append(boundary)
            consumed += boundary
        sizes.append(self.rows - consumed)
        return tuple(_Dataset(size) for size in sizes)


class _BlockedDataset:
    def __init__(self, blocks: int) -> None:
        self.blocks = blocks
        self.repartition_call: tuple[int, bool, bool] | None = None

    def num_blocks(self) -> int:
        return self.blocks

    def repartition(
        self,
        blocks: int,
        *,
        strict: bool,
        shuffle: bool,
    ) -> _BlockedDataset:
        self.repartition_call = (blocks, strict, shuffle)
        self.blocks = blocks
        return self


class _PredictionDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def map_batches(self, function: Any, **kwargs: Any) -> _PredictionDataset:
        self.frame = function(self.frame, **kwargs["fn_kwargs"])
        return self

    def to_pandas(self, limit: int | None = None) -> pd.DataFrame:
        assert limit is None
        return self.frame


def test_split_dataset_preserves_requested_three_way_counts() -> None:
    dataset = _Dataset(3_000_000)

    _, _, _, rows = runtime._split_dataset(
        dataset,
        {
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "split_strategy": "RANDOM",
            "seed": 42,
        },
    )

    assert dataset.random_seed == 42
    assert rows == {
        "total": 3_000_000,
        "train": 2_100_000,
        "validation": 450_000,
        "test": 450_000,
    }


def test_metrics_item_contains_non_empty_train_and_eval_series() -> None:
    item = runtime._metrics_item(
        2,
        10,
        {
            "train": {"logloss": [0.6, 0.5], "auc": [0.7, 0.8]},
            "eval": {"logloss": [0.65, 0.55], "auc": [0.68, 0.77]},
        },
    )

    assert item == {
        "current_round": 2,
        "total_rounds": 10,
        "progress_percent": 20.0,
        "metrics": [
            {"metric_name": "loss", "train": 0.5, "eval": 0.55},
            {"metric_name": "auc", "train": 0.8, "eval": 0.77},
        ],
    }


def test_worker_block_guard_repartitions_small_validation_dataset() -> None:
    dataset = _BlockedDataset(1)

    result = runtime._ensure_worker_blocks(
        dataset,
        worker_count=2,
        row_count=450_000,
        name="validation",
    )

    assert result is dataset
    assert dataset.repartition_call == (2, True, False)


def test_binary_evaluation_uses_all_test_rows_and_builds_dashboard_data() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0] * 10,
            "label": [0, 0, 0, 1, 1, 1] * 10,
        }
    )
    booster = xgboost.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=5,
    )

    metrics, details = runtime._evaluation(
        _PredictionDataset(frame),
        booster,
        feature_names=("feature_a",),
        label_name="label",
        task_type="BINARY_CLASSIFICATION",
        num_class=None,
        artifacts={"roc_curve": True, "threshold_analysis": True},
    )

    assert metrics["auc"] == pytest.approx(1.0)
    assert sum(details["confusion_matrix"].values()) == len(frame)
    assert len(details["roc_curve"]["fpr"]) == len(details["roc_curve"]["tpr"])
    threshold = details["threshold_analysis"]
    assert len(threshold["thresholds"]) == 19
    assert len(threshold["precision_values"]) == 19
    assert len(threshold["recall_values"]) == 19
    assert len(threshold["f1_values"]) == 19
    assert len(threshold["predicted_positive_rows"]) == 19


def test_bundle_export_accepts_source_column_feature_names(tmp_path: Any) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.9, 1.0] * 10,
            "label": [0, 0, 1, 1] * 10,
        }
    )
    booster = xgboost.train(
        {"objective": "binary:logistic", "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=2,
    )
    assert booster.feature_names == ["feature_a"]

    result = runtime._export_bundle(
        booster,
        feature_names=("feature_a",),
        bundle_uri=str(tmp_path / "bundle"),
        run_id="custom-feature-export",
    )

    assert result["bundle_id"]
    assert result["manifest_sha256"]
    assert booster.feature_names == ["feature_a"]


def test_feature_importance_maps_positional_xgboost_names() -> None:
    features = np.asarray([[0.0, 1.0], [0.1, 1.0], [0.9, 0.0], [1.0, 0.0]] * 10)
    labels = np.asarray([0, 0, 1, 1] * 10)
    booster = xgboost.train(
        {"objective": "binary:logistic", "seed": 7},
        xgboost.DMatrix(features, label=labels),
        num_boost_round=3,
    )

    importance = runtime._feature_importance(
        booster,
        ("business_feature_a", "business_feature_b"),
    )

    assert {item["model_feature_name"] for item in importance} == {
        "business_feature_a",
        "business_feature_b",
    }
    assert importance[0]["importance_score"] > 0


def test_distributed_correlation_statistics_match_pandas() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    first = runtime._correlation_batch_stats(frame.iloc[:2], list(frame.columns))
    second = runtime._correlation_batch_stats(frame.iloc[2:], list(frame.columns))

    result = runtime._finalize_correlation_stats(
        list(frame.columns),
        [
            *first.to_dict(orient="records"),
            *second.to_dict(orient="records"),
        ],
    )

    assert result is not None
    assert result["feature_names"] == ["a", "b", "constant"]
    assert result["values"][0][1] == pytest.approx(1.0)
    assert result["values"][0][2] == 0.0
    assert result["values"][1][2] == 0.0
    assert [result["values"][i][i] for i in range(3)] == [1.0, 1.0, 1.0]
