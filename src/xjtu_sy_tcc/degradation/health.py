"""Fold-specific baseline-centered and explicitly smoothed health indicators."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from xjtu_sy_tcc.degradation.preprocessing import FittedPreprocessor
from xjtu_sy_tcc.degradation.splits import Fold, assign_subset

HEALTH_COLUMNS = (
    "fold",
    "subset",
    "condition_id",
    "bearing_id",
    "acquisition_number",
    "sequence_index",
    "elapsed_minutes",
    "rul_minutes",
    "raw_health_indicator",
    "baseline_center",
    "baseline_acquisitions",
    "baseline_centered_health_indicator",
    "smoothed_health_indicator",
    "smoothing_method",
    "smoothing_window",
)


def build_health_indicators(
    features: pd.DataFrame,
    fold: Fold,
    preprocessor: FittedPreprocessor,
    baseline_fraction: float,
    minimum_baseline: int,
    smoothing_method: str,
    smoothing_window: int,
) -> pd.DataFrame:
    """Transform with frozen parameters and calibrate each bearing on its initial segment."""
    scores = preprocessor.transform(features)[:, 0]
    result = features.loc[
        :,
        [
            "condition_id",
            "bearing_id",
            "acquisition_number",
            "sequence_index",
            "elapsed_minutes",
            "rul_minutes",
        ],
    ].copy()
    result.insert(0, "subset", assign_subset(features, fold).to_numpy())
    result.insert(0, "fold", fold.fold_id)
    result["raw_health_indicator"] = scores
    frames = []
    for _, group in result.groupby("bearing_id", sort=True):
        ordered = group.sort_values("sequence_index").copy()
        baseline_count = max(minimum_baseline, math.ceil(len(ordered) * baseline_fraction))
        if baseline_count >= len(ordered):
            raise ValueError(
                f"Insufficient post-baseline observations for {ordered.iloc[0]['bearing_id']}"
            )
        center = float(np.median(ordered["raw_health_indicator"].iloc[:baseline_count]))
        ordered["baseline_center"] = center
        ordered["baseline_acquisitions"] = baseline_count
        ordered["baseline_centered_health_indicator"] = ordered["raw_health_indicator"] - center
        ordered["smoothed_health_indicator"] = smooth_indicator(
            ordered["baseline_centered_health_indicator"], smoothing_method, smoothing_window
        )
        ordered["smoothing_method"] = smoothing_method
        ordered["smoothing_window"] = smoothing_window
        frames.append(ordered)
    combined = pd.concat(frames, ignore_index=True).loc[:, HEALTH_COLUMNS]
    numeric = combined.select_dtypes(include="number").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Health indicators contain non-finite values")
    return combined


def smooth_indicator(values: pd.Series, method: str, window: int) -> pd.Series:
    """Apply a trailing rolling window with min_periods=1 and unchanged row count."""
    if window <= 0 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer")
    rolling = values.rolling(window=window, min_periods=1, center=False)
    if method == "median":
        return rolling.median()
    if method == "mean":
        return rolling.mean()
    raise ValueError("Smoothing method must be median or mean")
