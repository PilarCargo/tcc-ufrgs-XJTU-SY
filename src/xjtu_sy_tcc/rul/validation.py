"""End-to-end validation of frozen classical RUL experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xjtu_sy_tcc.degradation.splits import Fold


@dataclass(frozen=True, slots=True)
class Phase4Issue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Phase4Validation:
    status: str
    issues: tuple[Phase4Issue, ...]


def validate_phase4(
    features: pd.DataFrame,
    folds: tuple[Fold, ...],
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    artifact_records: list[dict[str, object]],
    test_onsets: pd.DataFrame,
    experiments: tuple[str, ...],
) -> Phase4Validation:
    issues: list[Phase4Issue] = []
    expected_keys = set(
        map(
            tuple,
            features[["condition_id", "bearing_id", "acquisition_number"]].itertuples(
                index=False, name=None
            ),
        )
    )
    for experiment in experiments:
        scoped = predictions[predictions["experiment"] == experiment]
        observed = set(
            map(
                tuple,
                scoped[["condition_id", "bearing_id", "acquisition_number"]].itertuples(
                    index=False, name=None
                ),
            )
        )
        if len(scoped) != len(features) or observed != expected_keys:
            issues.append(
                _error(
                    "prediction_coverage",
                    f"Experiment {experiment} does not cover every acquisition exactly once",
                )
            )
        if scoped.duplicated(["fold", "bearing_id", "acquisition_number"]).any():
            issues.append(
                _error(
                    "duplicate_prediction",
                    f"Experiment {experiment} has duplicate test predictions",
                )
            )
    numeric = predictions[
        ["rul_minutes", "prediction_raw_minutes", "prediction_non_negative_minutes"]
    ].to_numpy(float)
    if not np.all(np.isfinite(numeric)) or (predictions["rul_minutes"] < 0).any():
        issues.append(
            _error("invalid_predictions", "Predictions or targets are non-finite/invalid")
        )
    if (predictions["prediction_non_negative_minutes"] < 0).any():
        issues.append(
            _error(
                "negative_primary_prediction", "Non-negative predictions contain negative values"
            )
        )
    if len(test_onsets) != 15 or test_onsets["bearing_id"].nunique() != 15:
        issues.append(
            _error("onset_scope", "Post-hoc onset analysis does not use exactly 15 test records")
        )
    fold_map = {fold.fold_id: fold for fold in folds}
    for artifact in artifact_records:
        fold_id = int(artifact["fold"])
        if set(artifact["fitted_bearings"]) != set(fold_map[fold_id].train_bearings):
            issues.append(
                _error(
                    "training_leakage",
                    f"Artifact in fold {fold_id} was not fit on training bearings only",
                )
            )
        forbidden = {
            "rul_minutes",
            "bearing_id",
            "file_path",
            "file_name",
            "estimated_onset_elapsed_minutes",
        }
        if set(artifact["input_columns"]) & forbidden:
            issues.append(
                _error("forbidden_input", f"Artifact in fold {fold_id} contains forbidden inputs")
            )
    selected = candidates[candidates["selected"]]
    if len(selected) != len(folds) * len(experiments):
        issues.append(_error("model_selection", "Every fold/experiment must freeze one candidate"))
    return Phase4Validation("passed" if not issues else "failed", tuple(issues))


def _error(code: str, message: str) -> Phase4Issue:
    return Phase4Issue("error", code, message)
