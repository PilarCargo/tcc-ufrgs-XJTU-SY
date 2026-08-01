"""End-to-end Phase 3 traceability and leakage validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xjtu_sy_tcc.degradation.splits import Fold, validate_folds


@dataclass(frozen=True, slots=True)
class Phase3Issue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Phase3Validation:
    status: str
    issues: tuple[Phase3Issue, ...]


def validate_phase3(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...],
    folds: tuple[Fold, ...],
    artifacts: dict[int, dict[str, object]],
    health: pd.DataFrame,
    onsets: pd.DataFrame,
    expected_rows: int,
    expected_feature_count: int,
) -> Phase3Validation:
    issues: list[Phase3Issue] = []
    if len(features) != expected_rows:
        issues.append(
            _error("feature_row_count", f"Expected {expected_rows}, observed {len(features)}")
        )
    if len(feature_columns) != expected_feature_count:
        issues.append(
            _error(
                "feature_count",
                f"Expected {expected_feature_count}, observed {len(feature_columns)}",
            )
        )
    if features["bearing_id"].nunique() != 15:
        issues.append(_error("bearing_count", "Feature table must contain 15 bearings"))
    try:
        validate_folds(folds, features)
    except ValueError as exc:
        issues.append(_error("split_validation", str(exc)))
    fold_map = {fold.fold_id: fold for fold in folds}
    for fold_id, artifact in artifacts.items():
        expected_training = set(fold_map[fold_id].train_bearings)
        fitted = set(artifact["training_bearings"])
        if fitted != expected_training:
            issues.append(
                _error(
                    "preprocessing_leakage",
                    f"Fold {fold_id} artifact references non-training bearings",
                )
            )
        forbidden = {
            "rul_minutes",
            "elapsed_minutes",
            "sequence_index",
            "total_acquisitions",
            "life_fraction",
        }
        if set(artifact["selected_features"]) & forbidden:
            issues.append(_error("future_column", f"Fold {fold_id} selected a forbidden column"))
    if len(health) != expected_rows * len(folds):
        issues.append(
            _error(
                "health_row_count",
                f"Expected {expected_rows * len(folds)} health rows, observed {len(health)}",
            )
        )
    health_values = health[
        ["raw_health_indicator", "baseline_centered_health_indicator", "smoothed_health_indicator"]
    ].to_numpy(float)
    if not np.all(np.isfinite(health_values)):
        issues.append(_error("nonfinite_health", "Health indicators contain non-finite values"))
    source_keys = set(
        map(
            tuple,
            features[["condition_id", "bearing_id", "acquisition_number"]].itertuples(
                index=False, name=None
            ),
        )
    )
    health_keys = set(
        map(
            tuple,
            health[["condition_id", "bearing_id", "acquisition_number"]].itertuples(
                index=False, name=None
            ),
        )
    )
    if source_keys != health_keys:
        issues.append(
            _error("traceability", "Health rows do not map exactly to Phase 2 acquisitions")
        )
    detected = onsets[onsets["estimated_onset_status"] == "detected"]
    if len(detected):
        if (
            detected["estimated_onset_sequence_index"].astype(float)
            < detected["baseline_acquisitions"]
        ).any():
            issues.append(
                _error("onset_before_baseline", "An estimated onset precedes calibration")
            )
        if not detected["life_fraction_at_estimated_onset"].astype(float).between(0, 1).all():
            issues.append(
                _error("onset_bounds", "An estimated onset lies outside the bearing life")
            )
    return Phase3Validation("passed" if not issues else "failed", tuple(issues))


def _error(code: str, message: str) -> Phase3Issue:
    return Phase3Issue("error", code, message)
