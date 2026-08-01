"""Fold-specific training-only feature ranking and deterministic selection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

ALLOWED_METRICS = ("monotonicity", "trendability", "prognosability", "spearman_abs", "stability")


def rank_features(scores: pd.DataFrame, weights: Mapping[str, float], count: int) -> pd.DataFrame:
    """Min-max normalize components and rank with feature name as deterministic tie-breaker."""
    if (
        set(weights) != set(ALLOWED_METRICS)
        or any(not np.isfinite(value) or value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Metric weights must be finite, non-negative, and cover all metrics")
    if count <= 0 or count > len(scores):
        raise ValueError("Selected feature count is outside the available range")
    if scores["feature"].duplicated().any():
        raise ValueError("Feature names must be unique")
    ranked = scores.copy()
    composite = np.zeros(len(ranked), dtype=float)
    for metric in ALLOWED_METRICS:
        values = ranked[metric].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite ranking metric: {metric}")
        span = float(values.max() - values.min())
        normalized = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        ranked[f"normalized_{metric}"] = normalized
        composite += normalized * float(weights[metric]) / sum(weights.values())
    ranked["composite_score"] = composite
    ranked = ranked.sort_values(
        ["composite_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= count
    return ranked


def validate_feature_names(feature_names: tuple[str, ...], available: tuple[str, ...]) -> None:
    forbidden = {
        "rul_minutes",
        "elapsed_minutes",
        "sequence_index",
        "total_acquisitions",
        "life_fraction",
    }
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Selected features must be non-empty and unique")
    if set(feature_names) & forbidden or set(feature_names) - set(available):
        raise ValueError("Selected features contain metadata, targets, or unknown names")
