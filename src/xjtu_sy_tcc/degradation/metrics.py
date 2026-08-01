"""Explicit prognostic feature-quality metrics.

Definitions used here are intentionally fixed rather than mixed across publications:

* monotonicity is ``abs(n_positive - n_negative) / n_non_tied_differences``; ties are excluded;
* trend sign is ``sign(n_positive - n_negative)``;
* temporal association is Spearman rho against normalized sequence position;
* fluctuation is median absolute first difference divided by trajectory IQR;
* trendability is the median absolute Pearson correlation over interpolated bearing pairs;
* prognosability is ``exp(-MAD(final medians) / median(abs(final-initial)))``.

Undefined constant-trajectory correlations are represented by zero and pair counts are reported.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def bearing_feature_scores(
    features: pd.DataFrame, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Calculate within-bearing scores without sharing information between bearings."""
    rows = []
    for (condition, bearing), group in features.groupby(["condition_id", "bearing_id"], sort=True):
        ordered = group.sort_values("sequence_index")
        progression = _normalized_position(len(ordered))
        for feature in feature_columns:
            values = ordered[feature].to_numpy(dtype=np.float64)
            differences = np.diff(values)
            positive = int(np.count_nonzero(differences > 0))
            negative = int(np.count_nonzero(differences < 0))
            valid = positive + negative
            imbalance = positive - negative
            rho = _spearman(values, progression)
            scale = float(np.subtract(*np.percentile(values, [75, 25])))
            fluctuation = (
                float(np.median(np.abs(differences)) / scale)
                if scale > 0
                else (0.0 if np.all(differences == 0) else float("inf"))
            )
            rows.append(
                {
                    "condition_id": int(condition),
                    "bearing_id": str(bearing),
                    "feature": feature,
                    "monotonicity": abs(imbalance) / valid if valid else 0.0,
                    "trend_sign": int(np.sign(imbalance)),
                    "spearman": rho,
                    "spearman_abs": abs(rho),
                    "fluctuation": fluctuation,
                    "valid_differences": valid,
                    "tie_differences": int(len(differences) - valid),
                    "finite": bool(np.all(np.isfinite(values))),
                    "missing_count": int(np.isnan(values).sum()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_feature_scores(
    features: pd.DataFrame,
    by_bearing: pd.DataFrame,
    feature_columns: tuple[str, ...],
    grid_size: int,
    initial_fraction: float,
    final_fraction: float,
    bearings: tuple[str, ...] | None = None,
    condition_id: int | None = None,
) -> pd.DataFrame:
    """Aggregate metrics for an explicitly scoped bearing set."""
    scoped = features
    scores = by_bearing
    if bearings is not None:
        scoped = scoped[scoped["bearing_id"].isin(bearings)]
        scores = scores[scores["bearing_id"].isin(bearings)]
    if condition_id is not None:
        scoped = scoped[scoped["condition_id"] == condition_id]
        scores = scores[scores["condition_id"] == condition_id]
    rows = []
    for feature in feature_columns:
        metric_rows = scores[scores["feature"] == feature]
        trendability, pairs = trendability_score(scoped, feature, grid_size)
        prognosability = prognosability_score(scoped, feature, initial_fraction, final_fraction)
        spearman_values = metric_rows["spearman_abs"].to_numpy(dtype=float)
        stability = float(
            np.clip(1.0 - np.subtract(*np.percentile(spearman_values, [75, 25])), 0, 1)
        )
        rows.append(
            {
                "feature": feature,
                "monotonicity": float(metric_rows["monotonicity"].median()),
                "trendability": trendability,
                "trendability_valid_pairs": pairs,
                "prognosability": prognosability,
                "spearman": float(metric_rows["spearman"].median()),
                "spearman_abs": float(metric_rows["spearman_abs"].median()),
                "fluctuation": float(metric_rows["fluctuation"].replace([np.inf], np.nan).median()),
                "stability": stability,
                "bearing_count": int(metric_rows["bearing_id"].nunique()),
                "finite": bool(metric_rows["finite"].all()),
                "missing_count": int(metric_rows["missing_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def trendability_score(features: pd.DataFrame, feature: str, grid_size: int) -> tuple[float, int]:
    """Median absolute pairwise correlation after interpolation without extrapolation."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    grid = np.linspace(0.0, 1.0, grid_size)
    trajectories = []
    for _, group in features.groupby("bearing_id", sort=True):
        ordered = group.sort_values("sequence_index")
        values = ordered[feature].to_numpy(dtype=float)
        trajectories.append(np.interp(grid, _normalized_position(len(values)), values))
    correlations = []
    for left, right in itertools.combinations(trajectories, 2):
        if np.std(left) == 0 or np.std(right) == 0:
            continue
        correlations.append(abs(float(np.corrcoef(left, right)[0, 1])))
    return (float(np.median(correlations)) if correlations else 0.0, len(correlations))


def prognosability_score(
    features: pd.DataFrame, feature: str, initial_fraction: float, final_fraction: float
) -> float:
    """Robust endpoint consistency across complete bearing trajectories."""
    initial_values, final_values = [], []
    for _, group in features.groupby("bearing_id", sort=True):
        values = group.sort_values("sequence_index")[feature].to_numpy(dtype=float)
        initial_count = max(1, math.ceil(len(values) * initial_fraction))
        final_count = max(1, math.ceil(len(values) * final_fraction))
        initial_values.append(float(np.median(values[:initial_count])))
        final_values.append(float(np.median(values[-final_count:])))
    if not initial_values:
        return 0.0
    final_array = np.asarray(final_values)
    dispersion = float(np.median(np.abs(final_array - np.median(final_array))))
    change = float(np.median(np.abs(final_array - np.asarray(initial_values))))
    if change == 0:
        return 1.0 if dispersion == 0 and len(set(final_values)) == 1 else 0.0
    return float(np.exp(-dispersion / change))


def _normalized_position(size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("At least two acquisitions are required")
    return np.linspace(0.0, 1.0, size)


def _spearman(values: np.ndarray, progression: np.ndarray) -> float:
    if np.ptp(values) == 0:
        return 0.0
    result = float(spearmanr(values, progression).statistic)
    return result if np.isfinite(result) else 0.0
