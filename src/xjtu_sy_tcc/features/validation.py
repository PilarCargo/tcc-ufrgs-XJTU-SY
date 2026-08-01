"""Validation of the Phase-2 acquisition-level feature table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FeatureValidation:
    status: str
    issues: tuple[FeatureIssue, ...]
    expected_rows: int
    observed_rows: int
    feature_columns: tuple[str, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity == "error" for item in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.issues)


KEY_COLUMNS = ("condition_id", "bearing_id", "acquisition_number")
TARGET_COLUMNS = ("rul_minutes",)


def validate_feature_table(
    metadata: pd.DataFrame,
    table: pd.DataFrame,
    required_feature_columns: tuple[str, ...],
) -> FeatureValidation:
    """Verify one-to-one manifest alignment, ordering, targets, and numeric values."""
    issues: list[FeatureIssue] = []
    required = (
        set(KEY_COLUMNS)
        | {"sequence_index", "elapsed_minutes", "rul_minutes", "sampling_frequency_hz"}
        | set(required_feature_columns)
    )
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        issues.append(_error("missing_columns", f"Missing columns: {', '.join(missing_columns)}"))
    if len(table) != len(metadata):
        issues.append(_error("row_count", f"Expected {len(metadata)} rows, observed {len(table)}"))
    if set(KEY_COLUMNS).issubset(table.columns):
        duplicate_count = int(table.duplicated(list(KEY_COLUMNS)).sum())
        if duplicate_count:
            issues.append(
                _error("duplicate_acquisitions", f"Found {duplicate_count} duplicate rows")
            )
        expected_keys = set(
            map(tuple, metadata.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        )
        observed_keys = set(
            map(tuple, table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        )
        if expected_keys - observed_keys:
            issues.append(
                _error(
                    "missing_acquisitions",
                    f"Missing {len(expected_keys - observed_keys)} acquisitions",
                )
            )
        if observed_keys - expected_keys:
            issues.append(
                _error(
                    "unexpected_acquisitions",
                    f"Found {len(observed_keys - expected_keys)} unexpected acquisitions",
                )
            )
    available_features = [name for name in required_feature_columns if name in table]
    if available_features:
        values = table[available_features].to_numpy(dtype=np.float64)
        invalid = ~np.isfinite(values)
        if invalid.any():
            affected = sorted(np.asarray(available_features)[np.any(invalid, axis=0)].tolist())
            issues.append(
                _error("nonfinite_features", f"NaN or infinite values in: {', '.join(affected)}")
            )
    nonnegative_tokens = ("energy", "power", "rms", "std", "variance", "spread")
    for column in available_features:
        if any(token in column for token in nonnegative_tokens) and (table[column] < 0).any():
            issues.append(_error("negative_feature", f"Negative values found in {column}"))
        if column.endswith("spectral_entropy") and not table[column].between(0.0, 1.0).all():
            issues.append(_error("entropy_range", f"Values outside [0, 1] in {column}"))
        if column.endswith("_frequency_hz"):
            nyquist = table["sampling_frequency_hz"] / 2.0
            if ((table[column] < 0) | (table[column] > nyquist)).any():
                issues.append(
                    _error("frequency_range", f"Values outside Nyquist range in {column}")
                )
    if {"rul_minutes", "bearing_id", "sequence_index", "elapsed_minutes"}.issubset(table):
        if (table["rul_minutes"] < 0).any():
            issues.append(_error("negative_rul", "RUL contains negative values"))
        for bearing, group in table.groupby("bearing_id", sort=False):
            ordered = group.sort_values("sequence_index")
            if (
                ordered["sequence_index"].duplicated().any()
                or not ordered["sequence_index"].is_monotonic_increasing
            ):
                issues.append(_error("sequence_order", f"Invalid sequence order for {bearing}"))
            if not ordered["elapsed_minutes"].is_monotonic_increasing:
                issues.append(_error("elapsed_order", f"Non-monotonic elapsed time for {bearing}"))
            if float(ordered.iloc[-1]["rul_minutes"]) != 0.0:
                issues.append(_error("final_rul", f"Final RUL is not zero for {bearing}"))
    return FeatureValidation(
        "passed" if not any(item.severity == "error" for item in issues) else "failed",
        tuple(issues),
        len(metadata),
        len(table),
        tuple(required_feature_columns),
    )


def _error(code: str, message: str) -> FeatureIssue:
    return FeatureIssue("error", code, message)
