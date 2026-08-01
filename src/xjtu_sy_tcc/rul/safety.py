"""Frozen-fold verification and predictive-feature leakage barriers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_sy_tcc.degradation.splits import Fold, validate_folds

FORBIDDEN_EXACT = {
    "rul_minutes",
    "total_acquisitions",
    "normalized_life_fraction",
    "life_fraction",
    "remaining_acquisitions",
    "final_file_index",
    "bearing_id",
    "file_path",
    "file_name",
    "fold",
    "subset",
    "estimated_onset_sequence_index",
    "estimated_onset_elapsed_minutes",
    "distance_to_estimated_onset",
    "post_onset",
    "smoothed_health_indicator",
    "baseline_centered_health_indicator",
    "raw_health_indicator",
}
FORBIDDEN_TOKENS = ("onset", "remaining_acquisition", "total_lifetime", "final_lifetime")
TRACE_COLUMNS = (
    "condition_id",
    "bearing_id",
    "acquisition_number",
    "sequence_index",
    "elapsed_minutes",
    "rul_minutes",
    "rotation_rpm",
    "radial_load_kn",
)


def phase3_health_predictive_policy() -> dict[str, object]:
    """Declare why the retrospective Phase 3 indicator is excluded from full-trajectory input."""
    return {
        "smoothing_is_causal": True,
        "baseline_requires_complete_initial_calibration_interval": True,
        "full_trajectory_prediction_allowed": False,
        "earliest_allowed_use": "after baseline_acquisitions",
        "decision": "omit from Phase 4 predictive experiments",
    }


def load_frozen_folds(
    path: Path,
    expected_sha256: str,
    expected_configuration_hash: str,
    features: pd.DataFrame,
) -> tuple[tuple[Fold, ...], dict[str, object]]:
    """Load the Phase 3 manifest only when both physical and logical hashes match."""
    payload_bytes = path.read_bytes()
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen split SHA-256 changed: expected {expected_sha256}, observed {observed}"
        )
    payload = json.loads(payload_bytes)
    if payload.get("configuration_hash") != expected_configuration_hash:
        raise ValueError("Frozen split logical configuration hash changed")
    folds = []
    for item in payload.get("folds", []):
        subsets = item["subsets"]
        folds.append(
            Fold(
                int(item["fold_id"]),
                tuple(subsets["train"]["bearings"]),
                tuple(subsets["validation"]["bearings"]),
                tuple(subsets["test"]["bearings"]),
            )
        )
    result = tuple(folds)
    validate_folds(result, features)
    return result, payload


def load_selected_features(
    path: Path, folds: tuple[Fold, ...], available: tuple[str, ...]
) -> dict[int, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    fold_map = {fold.fold_id: fold for fold in folds}
    for item in payload.get("folds", []):
        fold_id = int(item["fold"])
        if tuple(item["training_bearings"]) != fold_map[fold_id].train_bearings:
            raise ValueError(
                f"Selected-feature manifest training bearings changed for fold {fold_id}"
            )
        selected = tuple(item["selected_features"])
        validate_input_columns(selected, available, allow_time=False)
        result[fold_id] = selected
    if set(result) != set(fold_map):
        raise ValueError("Selected-feature manifest does not cover all frozen folds")
    return result


def validate_input_columns(
    columns: tuple[str, ...], available: tuple[str, ...], *, allow_time: bool
) -> None:
    """Reject targets, identifiers, future-derived values, and undeclared time inputs."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Predictive input columns must be non-empty and unique")
    unknown = set(columns) - set(available)
    forbidden = set(columns) & FORBIDDEN_EXACT
    forbidden.update(name for name in columns if any(token in name for token in FORBIDDEN_TOKENS))
    if not allow_time:
        forbidden.update(set(columns) & {"elapsed_minutes", "sequence_index"})
    if unknown or forbidden:
        raise ValueError(
            f"Unsafe predictive inputs; unknown={sorted(unknown)}, forbidden={sorted(forbidden)}"
        )


def validate_phase2_features(
    features: pd.DataFrame, expected_rows: int, expected_count: int
) -> tuple[str, ...]:
    feature_columns = tuple(
        name for name in features if name.startswith(("horizontal_", "vertical_"))
    )
    if len(features) != expected_rows or len(feature_columns) != expected_count:
        raise ValueError("Phase 2 input row or vibration-feature count changed")
    if features["bearing_id"].nunique() != 15:
        raise ValueError("Phase 2 input must contain 15 bearings")
    if features.duplicated(["condition_id", "bearing_id", "acquisition_number"]).any():
        raise ValueError("Phase 2 input contains duplicate acquisitions")
    numeric = features.loc[:, feature_columns + ("rul_minutes",)].to_numpy(float)
    if not np.all(np.isfinite(numeric)) or (features["rul_minutes"] < 0).any():
        raise ValueError("Phase 2 features or targets contain invalid values")
    return feature_columns


def load_test_onsets(path: Path, folds: tuple[Fold, ...]) -> pd.DataFrame:
    """Return exactly one retrospective test onset per frozen test bearing."""
    table = pd.read_parquet(path)
    test = table[table["subset"] == "test"].copy()
    expected = {(fold.fold_id, bearing) for fold in folds for bearing in fold.test_bearings}
    observed = set(map(tuple, test[["fold", "bearing_id"]].itertuples(index=False, name=None)))
    if len(test) != 15 or test["bearing_id"].nunique() != 15 or observed != expected:
        raise ValueError("Post-hoc onset table must contain exactly 15 unique frozen test records")
    if table[table["subset"] != "test"].empty:
        raise ValueError("Expected Phase 3 onset artifact with explicit non-test rows")
    return test
