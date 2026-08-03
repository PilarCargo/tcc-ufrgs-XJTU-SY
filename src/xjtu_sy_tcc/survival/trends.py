"""Causal trailing feature descriptors with fixed initial calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_trends(frame, features, baseline, windows, epsilon=1e-12):
    rows = []
    for _, group in frame.groupby("bearing_id", sort=True):
        g = group.sort_values("sequence_index").copy()
        for feature in features:
            values = g[feature].astype(float)
            base = values.iloc[:baseline]
            center = float(base.median())
            mad = float((base - center).abs().median())
            g[f"{feature}__baseline_difference"] = values - center
            g[f"{feature}__baseline_robust_ratio"] = (values - center) / max(mad, epsilon)
            g[f"{feature}__first_difference"] = values.diff().fillna(0)
            for window in windows:
                rolling = values.rolling(window, min_periods=1)
                prefix = f"{feature}__w{window}"
                g[f"{prefix}_mean"] = rolling.mean()
                g[f"{prefix}_median"] = rolling.median()
                g[f"{prefix}_std"] = rolling.std(ddof=0).fillna(0)
                g[f"{prefix}_ewma"] = values.ewm(span=window, adjust=False).mean()
                g[f"{prefix}_slope"] = [
                    _slope(values.iloc[max(0, i - window + 1) : i + 1].to_numpy())
                    for i in range(len(values))
                ]
        rows.append(g)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result.select_dtypes("number").to_numpy()).all():
        raise ValueError("Non-finite causal trend")
    return result


def _slope(values):
    return float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) >= 2 else 0.0
