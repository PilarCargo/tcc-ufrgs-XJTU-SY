"""Inverse-bearing-frequency sample weights with equal bearing totals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bearing_balanced_weights(training: pd.DataFrame) -> np.ndarray:
    """Return deterministic weights with mean one and equal total per bearing."""
    if training.empty or "bearing_id" not in training:
        raise ValueError("Training rows with bearing_id are required")
    counts = training["bearing_id"].value_counts()
    if (counts <= 0).any():
        raise ValueError("Every training bearing must contain observations")
    factor = len(training) / len(counts)
    weights = training["bearing_id"].map(lambda name: factor / counts[name]).to_numpy(float)
    totals = pd.Series(weights).groupby(training["bearing_id"].reset_index(drop=True)).sum()
    if not np.allclose(totals, totals.iloc[0], rtol=1e-12, atol=1e-12):
        raise ValueError("Bearing-balanced weights do not have equal totals")
    if not np.isclose(weights.mean(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Bearing-balanced weights do not have mean one")
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or (values <= 0).any()
    ):
        raise ValueError("Effective sample size requires positive finite one-dimensional weights")
    return float(values.sum() ** 2 / np.square(values).sum())
