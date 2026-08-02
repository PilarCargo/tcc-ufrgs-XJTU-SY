"""Canonical, immutable harmonization and matched-support construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["condition_id", "bearing_id", "acquisition_number"]


def canonicalize(
    phase4: pd.DataFrame, phase5: pd.DataFrame, allowed: tuple[str, ...]
) -> pd.DataFrame:
    frames = []
    for phase, frame in ((4, phase4), (5, phase5)):
        required = set(
            KEY
            + [
                "fold",
                "experiment",
                "sequence_index",
                "rul_minutes",
                "prediction_raw_minutes",
                "prediction_non_negative_minutes",
            ]
        )
        if not required.issubset(frame):
            raise ValueError(f"Phase {phase} prediction schema is incomplete")
        scoped = frame[frame.experiment.isin(allowed)].copy()
        unknown = set(scoped.experiment) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        if scoped.duplicated(KEY + ["experiment"]).any():
            raise ValueError("Duplicate model prediction for canonical acquisition key")
        scoped["source_phase"] = phase
        scoped["canonical_key"] = (
            scoped.condition_id.astype(str)
            + "|"
            + scoped.bearing_id
            + "|"
            + scoped.acquisition_number.astype(str)
        )
        frames.append(scoped)
    result = pd.concat(frames, ignore_index=True)
    truth = result.groupby("canonical_key").rul_minutes.nunique()
    if (truth > 1).any():
        raise ValueError("True RUL mismatch across artifacts")
    fold = result.groupby("bearing_id").fold.nunique()
    if (fold != 1).any() or result.bearing_id.nunique() != 15:
        raise ValueError("Invalid frozen test-fold mapping")
    if not np.isfinite(
        result[
            ["rul_minutes", "prediction_raw_minutes", "prediction_non_negative_minutes"]
        ].to_numpy()
    ).all():
        raise ValueError("Non-finite prediction artifact")
    return result


def pairwise_support(data: pd.DataFrame, candidate: str, reference: str):
    left = data[data.experiment == candidate]
    right = data[data.experiment == reference]
    common = set(left.canonical_key) & set(right.canonical_key)
    if not common:
        raise ValueError("Empty pairwise support")
    return left[left.canonical_key.isin(common)].copy(), right[
        right.canonical_key.isin(common)
    ].copy()


def strict_common_support(data: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    keys = None
    for model in models:
        current = set(data.loc[data.experiment == model, "canonical_key"])
        keys = current if keys is None else keys & current
    result = data[data.canonical_key.isin(keys) & data.experiment.isin(models)].copy()
    if not keys or result.bearing_id.nunique() != 15:
        raise ValueError("Strict common support must cover all bearings")
    return result


def support_manifest(data, candidate, reference):
    left, right = pairwise_support(data, candidate, reference)
    rows = []
    for bearing in sorted(data.bearing_id.unique()):
        native_c = int((data.experiment.eq(candidate) & data.bearing_id.eq(bearing)).sum())
        native_r = int((data.experiment.eq(reference) & data.bearing_id.eq(bearing)).sum())
        matched = int((left.bearing_id == bearing).sum())
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "bearing_id": bearing,
                "candidate_native_count": native_c,
                "reference_native_count": native_r,
                "matched_count": matched,
                "candidate_retention": matched / native_c,
                "reference_retention": matched / native_r,
            }
        )
    return pd.DataFrame(rows)
