"""Strictly trailing normalization, aggregation, and persistent alarm rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_aggregate(values, window, method="median"):
    s = pd.Series(values, dtype=float)
    if method == "ewma":
        return s.ewm(span=window, adjust=False).mean().to_numpy()
    rolling = s.rolling(window, min_periods=1)
    return (rolling.median() if method == "median" else rolling.mean()).to_numpy()


def robust_normalize(values, baseline, epsilon=1e-12):
    values = np.asarray(values, float)
    base = values[:baseline]
    center = float(np.median(base))
    mad = float(np.median(np.abs(base - center)))
    scale = max(mad, epsilon)
    return (values - center) / scale, center, mad


def persistent_alarm(
    values, baseline, threshold, persistence, maximum_gap, minimum_post, minimum_effect
):
    x = np.asarray(values, float)
    crossings = np.flatnonzero((np.arange(len(x)) >= baseline) & (x > threshold))
    first = int(crossings[0]) if len(crossings) else None
    for end in range(baseline, len(x)):
        start = max(baseline, end - (persistence + maximum_gap) + 1)
        segment = x[start : end + 1]
        hits = np.flatnonzero(segment > threshold)
        if (
            len(hits) >= persistence
            and (hits[-1] - hits[0] + 1) <= persistence + maximum_gap
            and len(x) - 1 - end >= minimum_post
            and np.median(segment[hits]) >= minimum_effect
        ):
            return {
                "status": "detected",
                "first_threshold_crossing_index": first,
                "alarm_confirmation_index": end,
                "persistence_duration": len(hits),
                "rejection_reasons": "",
            }
    return {
        "status": "not_detected",
        "first_threshold_crossing_index": first,
        "alarm_confirmation_index": None,
        "persistence_duration": 0,
        "rejection_reasons": "persistence/effect/follow-up criteria not satisfied",
    }
