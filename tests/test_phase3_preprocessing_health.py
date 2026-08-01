"""Leakage, balancing, PCA orientation, and health-indicator tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from xjtu_sy_tcc.config.phase3 import TransformationConfig, load_phase3_config
from xjtu_sy_tcc.degradation.health import build_health_indicators, smooth_indicator
from xjtu_sy_tcc.degradation.preprocessing import balanced_training_matrix, fit_preprocessor
from xjtu_sy_tcc.degradation.splits import build_folds


def _features() -> pd.DataFrame:
    rows = []
    for condition in range(1, 4):
        for position in range(1, 6):
            bearing = f"Bearing{condition}_{position}"
            for sequence in range(20):
                rows.append(
                    {
                        "condition_id": condition,
                        "bearing_id": bearing,
                        "acquisition_number": sequence + 1,
                        "sequence_index": sequence,
                        "elapsed_minutes": float(sequence),
                        "rul_minutes": float(19 - sequence),
                        "horizontal_energy": float((sequence + 1) ** 2 + position),
                        "vertical_mean": float(sequence + condition),
                        "horizontal_rms": float(sequence / 10 + 1),
                    }
                )
    return pd.DataFrame(rows)


def _config():
    base = load_phase3_config(Path("configs/phase3.yaml"))
    return replace(
        base,
        balance_points_per_bearing=10,
        pca_components=2,
        selected_feature_count=3,
        minimum_baseline_acquisitions=2,
        transformation=TransformationConfig("auto_log1p_nonnegative", ("energy", "rms")),
    )


def test_balancing_gives_each_training_bearing_equal_weight() -> None:
    features = _features()
    fold = build_folds(features)[0]
    training = features[features["bearing_id"].isin(fold.train_bearings)]
    matrix, progression = balanced_training_matrix(
        training, ("horizontal_energy", "vertical_mean"), 10
    )
    assert matrix.shape == (90, 2)
    assert progression.shape == (90,)


def test_preprocessor_fits_training_only_and_transforms_without_refit() -> None:
    features = _features()
    fold = build_folds(features)[0]
    training = features[features["bearing_id"].isin(fold.train_bearings)]
    selected = ("horizontal_energy", "vertical_mean", "horizontal_rms")
    fitted = fit_preprocessor(training, selected, _config())
    center_before = fitted.scaler.center_.copy()
    transformed = fitted.transform(features)
    assert set(fitted.training_bearings) == set(fold.train_bearings)
    assert np.array_equal(center_before, fitted.scaler.center_)
    assert transformed.shape == (len(features), 2)
    assert np.isfinite(transformed).all()
    assert fitted.orientation in {-1, 1}


def test_pca_orientation_is_deterministic_and_increasing_on_training() -> None:
    features = _features()
    fold = build_folds(features)[0]
    training = features[features["bearing_id"].isin(fold.train_bearings)]
    selected = ("horizontal_energy", "vertical_mean", "horizontal_rms")
    first = fit_preprocessor(training, selected, _config())
    second = fit_preprocessor(training, selected, _config())
    assert first.orientation == second.orientation
    assert np.allclose(first.pca.components_, second.pca.components_)
    component = pd.Series(first.transform(training)[:, 0], index=training.index)
    correlations = []
    for _bearing, indices in training.groupby("bearing_id", sort=True).groups.items():
        correlations.append(
            spearmanr(component.loc[indices], training.loc[indices, "sequence_index"]).statistic
        )
    assert np.median(correlations) > 0


def test_log_transformation_rejects_incompatible_negative_values() -> None:
    features = _features()
    fold = build_folds(features)[0]
    training = features[features["bearing_id"].isin(fold.train_bearings)].copy()
    training.loc[training.index[0], "horizontal_energy"] = -1
    with pytest.raises(ValueError, match="negative"):
        fit_preprocessor(
            training, ("horizontal_energy", "vertical_mean", "horizontal_rms"), _config()
        )


def test_health_baseline_centering_smoothing_and_raw_preservation() -> None:
    features = _features()
    fold = build_folds(features)[0]
    selected = ("horizontal_energy", "vertical_mean", "horizontal_rms")
    training = features[features["bearing_id"].isin(fold.train_bearings)]
    fitted = fit_preprocessor(training, selected, _config())
    health = build_health_indicators(features, fold, fitted, 0.1, 2, "median", 3)
    assert len(health) == len(features)
    assert np.isfinite(health.select_dtypes(include="number").to_numpy()).all()
    for _, group in health.groupby("bearing_id"):
        assert np.median(group["baseline_centered_health_indicator"].iloc[:2]) == pytest.approx(0)
        assert not np.shares_memory(
            group["raw_health_indicator"].to_numpy(),
            group["smoothed_health_indicator"].to_numpy(),
        )


def test_smoothing_edge_policy_and_insufficient_baseline() -> None:
    values = pd.Series([1.0, 100.0, 2.0, 3.0])
    assert smooth_indicator(values, "median", 3).tolist() == [1.0, 50.5, 2.0, 3.0]
    features = _features()
    fold = build_folds(features)[0]
    selected = ("horizontal_energy", "vertical_mean", "horizontal_rms")
    fitted = fit_preprocessor(
        features[features["bearing_id"].isin(fold.train_bearings)], selected, _config()
    )
    with pytest.raises(ValueError, match="Insufficient"):
        build_health_indicators(features, fold, fitted, 0.9, 20, "median", 3)
