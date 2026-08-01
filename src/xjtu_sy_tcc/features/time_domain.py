"""Numerically explicit time-domain vibration features."""

from __future__ import annotations

import numpy as np

TIME_FEATURE_NAMES = (
    "mean",
    "std",
    "variance",
    "rms",
    "minimum",
    "maximum",
    "max_abs",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "energy",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
)


def time_features(signal: np.ndarray) -> dict[str, float]:
    """Calculate scalar features; mathematically undefined ratios remain NaN."""
    values = _valid_signal(signal).astype(np.float64, copy=False)
    absolute = np.abs(values)
    mean_abs = float(np.mean(absolute, dtype=np.float64))
    mean_sqrt_abs = float(np.mean(np.sqrt(absolute), dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    maximum_abs = float(np.max(absolute))
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    variance = float(np.var(values, ddof=0, dtype=np.float64))
    skewness = (
        float(np.mean(np.power(centered, 3), dtype=np.float64) / variance**1.5)
        if variance != 0.0
        else float("nan")
    )
    pearson_kurtosis = (
        float(np.mean(np.power(centered, 4), dtype=np.float64) / variance**2)
        if variance != 0.0
        else float("nan")
    )
    return {
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "variance": variance,
        "rms": rms,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "max_abs": maximum_abs,
        "peak_to_peak": float(np.ptp(values)),
        "skewness": skewness,
        "kurtosis": pearson_kurtosis,
        "energy": float(np.sum(np.square(values), dtype=np.float64)),
        "crest_factor": _safe_ratio(maximum_abs, rms),
        "shape_factor": _safe_ratio(rms, mean_abs),
        "impulse_factor": _safe_ratio(maximum_abs, mean_abs),
        "clearance_factor": _safe_ratio(maximum_abs, mean_sqrt_abs**2),
    }


def _valid_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite numeric values")
    return values


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0.0 else float("nan")
