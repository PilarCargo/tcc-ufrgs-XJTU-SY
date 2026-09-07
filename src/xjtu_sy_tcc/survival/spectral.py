"""Detector-specific binned Welch probability distributions and KL divergence."""

from __future__ import annotations

import numpy as np

from xjtu_sy_tcc.features.frequency_domain import compute_welch_psd


def binned_distribution(signal, fs, nperseg, overlap, minimum, maximum, bins, epsilon):
    f, p = compute_welch_psd(signal, fs, nperseg, overlap)
    edges = np.linspace(minimum, maximum, bins + 1)
    values = np.zeros(bins, float)
    for i in range(bins):
        mask = (f >= edges[i]) & (f < edges[i + 1] if i < bins - 1 else f <= edges[i + 1])
        values[i] = np.trapezoid(p[mask], f[mask]) if mask.sum() >= 2 else 0
    values = np.maximum(values, 0) + epsilon
    values /= values.sum()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite spectral distribution")
    return values.astype("float32")


def shaft_order_distribution(signal, fs, nperseg, overlap, rpm, max_order, bins, epsilon):
    """Welch power in shaft-order bins; this is mechanical context, not a fault-frequency claim."""
    if rpm <= 0 or max_order <= 1:
        raise ValueError("Positive shaft speed and a valid order range are required")
    shaft_hz = rpm / 60.0
    return binned_distribution(
        signal, fs, nperseg, overlap, 0.0, shaft_hz * max_order, bins, epsilon
    )


def validate_distribution(value, tolerance=1e-6):
    x = np.asarray(value, float)
    if (
        x.ndim != 1
        or np.any(x < 0)
        or not np.isfinite(x).all()
        or not np.isclose(x.sum(), 1, atol=tolerance)
    ):
        raise ValueError("Input must be a normalized finite probability distribution")
    return x


def kl_divergence(p, q, epsilon=1e-12, symmetric=False):
    p = validate_distribution(p)
    q = validate_distribution(q)
    a = np.maximum(p, epsilon)
    b = np.maximum(q, epsilon)
    a /= a.sum()
    b /= b.sum()
    forward = float(np.sum(a * np.log(a / b)))
    return 0.5 * (forward + float(np.sum(b * np.log(b / a)))) if symmetric else forward
