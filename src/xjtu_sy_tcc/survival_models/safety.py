"""Input integrity and evaluation-truth access guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IDENTIFIERS = {
    "fold",
    "subset",
    "bearing_id",
    "condition_id",
    "landmark_id",
    "file_path",
    "file_name",
    "acquisition_number",
    "scenario_id",
    "duration_minutes",
    "event_observed",
    "landmark_weight",
    "target_censoring_rate",
    "administrative_horizon_minutes",
}
EXTRA_FORBIDDEN = {
    "true_time_to_failure_minutes",
    "rul_minutes",
    "normalized_life_fraction",
    "consumed_life_fraction",
    "final_acquisition_index",
    "total_acquisition_count",
    "retrospective_pelt_onset",
    "causal_vs_pelt_difference",
    "future_event_time",
}


@dataclass(slots=True)
class EvaluationTruthGuard:
    """Prevent evaluation-only truth from being opened before test predictions freeze."""

    path: Path
    frozen: bool = False
    access_count: int = 0

    def freeze_predictions(self) -> None:
        self.frozen = True

    def load(self) -> pd.DataFrame:
        if not self.frozen:
            raise RuntimeError(
                "Evaluation truth cannot be loaded before test predictions are frozen"
            )
        self.access_count += 1
        return pd.read_parquet(self.path)


def verify_split_manifest(path: Path, expected_sha: str, expected_logical: str) -> dict:
    physical = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    logical = manifest.get("configuration_hash")
    if physical != expected_sha or logical != expected_logical:
        raise ValueError("Frozen Phase 3 split manifest hash mismatch")
    folds = manifest.get("folds", [])
    tests = []
    for fold in folds:
        subsets = fold["subsets"]
        train = set(subsets["train"]["bearings"])
        validation = set(subsets["validation"]["bearings"])
        test = set(subsets["test"]["bearings"])
        if (
            len(train) != 9
            or len(validation) != 3
            or len(test) != 3
            or train & validation
            or train & test
            or validation & test
        ):
            raise ValueError("Invalid frozen bearing fold")
        tests.extend(test)
    if len(tests) != 15 or len(set(tests)) != 15:
        raise ValueError("Each bearing must occur once in the test subset")
    return {"physical_sha256": physical, "logical_hash": logical, "test_bearings": sorted(tests)}


def forbidden_set(manifest_path: Path) -> set[str]:
    data = json.loads(manifest_path.read_text())
    return set(data.get("future_derived_forbidden_columns", ())) | IDENTIFIERS | EXTRA_FORBIDDEN


def validate_model_columns(columns: list[str] | tuple[str, ...], forbidden: set[str]) -> None:
    bad = sorted(set(columns) & forbidden)
    if bad:
        raise ValueError(f"Forbidden model columns: {bad}")


def validate_outcome(data: pd.DataFrame) -> None:
    if not set(data.event_observed.dropna().unique()).issubset({0, 1}):
        raise ValueError("Survival event indicators must be binary")
    if not np.isfinite(data.duration_minutes).all() or (data.duration_minutes <= 0).any():
        raise ValueError("Survival durations must be finite and positive")


def assert_traceability(data: pd.DataFrame) -> None:
    required = {"fold", "subset", "bearing_id", "landmark_id", "acquisition_number"}
    if required - set(data):
        raise ValueError("Landmark traceability columns are missing")
    if data[list(required)].isna().any().any() or data.landmark_id.duplicated().any():
        raise ValueError("Landmark traceability is invalid")
