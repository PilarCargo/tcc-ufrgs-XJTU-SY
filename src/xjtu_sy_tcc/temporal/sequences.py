"""Deterministic causal sequence construction within complete bearings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SequenceSet:
    features: np.ndarray
    context: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    metadata: pd.DataFrame


def build_sequences(frame: pd.DataFrame, feature_columns: list[str], length: int) -> SequenceSet:
    if length < 1:
        raise ValueError("Sequence length must be positive")
    xs = []
    cs = []
    ys = []
    meta = []
    for _bearing, group in frame.groupby("bearing_id", sort=True):
        g = group.sort_values("sequence_index")
        if not np.array_equal(np.diff(g.sequence_index.to_numpy()), np.ones(max(len(g) - 1, 0))):
            raise ValueError("Non-chronological or discontinuous bearing")
        values = g[feature_columns].to_numpy(float)
        context = g[["elapsed_minutes", "rotation_rpm", "radial_load_kn"]].to_numpy(float)
        for end in range(length - 1, len(g)):
            xs.append(values[end - length + 1 : end + 1])
            cs.append(context[end])
            ys.append(g.iloc[end].rul_minutes)
            meta.append(g.iloc[end])
    if not xs:
        raise ValueError("No eligible sequences")
    metadata = pd.DataFrame(meta).reset_index(drop=True)
    counts = metadata.bearing_id.value_counts()
    raw = metadata.bearing_id.map(1 / counts).to_numpy(float)
    weights = raw / raw.mean()
    return SequenceSet(
        np.asarray(xs, dtype=np.float32),
        np.asarray(cs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        weights.astype(np.float32),
        metadata,
    )


def sequence_coverage(frame: pd.DataFrame, length: int, fold: int, subset: str) -> pd.DataFrame:
    rows = []
    for bearing, g in frame.groupby("bearing_id", sort=True):
        n = len(g)
        eligible = max(0, n - length + 1)
        rows.append(
            {
                "fold": fold,
                "subset": subset,
                "bearing_id": bearing,
                "sequence_length": length,
                "original_acquisitions": n,
                "eligible_sequences": eligible,
                "dropped_prefix": min(length - 1, n),
                "coverage_proportion": eligible / n,
            }
        )
    return pd.DataFrame(rows)
