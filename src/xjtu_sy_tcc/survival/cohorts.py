"""Onset-dependent landmark cohorts, censoring, and bearing-balanced weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def landmarks(features, onsets, stride, minimum_followup):
    rows = []
    excluded = []
    for onset in onsets.itertuples():
        if onset.detection_status != "detected":
            excluded.append(
                {
                    "fold": onset.fold,
                    "subset": onset.subset,
                    "bearing_id": onset.bearing_id,
                    "reason": "causal onset not detected",
                }
            )
            continue
        group = features[
            (features.fold == onset.fold)
            & (features.subset == onset.subset)
            & (features.bearing_id == onset.bearing_id)
        ].sort_values("sequence_index")
        final = int(group.sequence_index.max())
        eligible = (
            group[
                (group.sequence_index >= onset.alarm_confirmation_index)
                & (group.sequence_index <= final - minimum_followup)
            ]
            .iloc[::stride]
            .copy()
        )
        for row in eligible.to_dict("records"):
            row.update(
                {
                    "fold": onset.fold,
                    "subset": onset.subset,
                    "landmark_id": (
                        f"{onset.fold}:{onset.subset}:{onset.bearing_id}:{row['sequence_index']}"
                    ),
                    "time_since_causal_onset": row["sequence_index"]
                    - onset.alarm_confirmation_index,
                    "true_time_to_failure_minutes": final - row["sequence_index"],
                    "duration_minutes": final - row["sequence_index"],
                    "event_observed": 1,
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        counts = result.groupby(["fold", "subset", "bearing_id"]).bearing_id.transform("size")
        result["landmark_weight"] = 1 / counts
        result["landmark_weight"] /= result.groupby(["fold", "subset"]).landmark_weight.transform(
            "mean"
        )
    return result, pd.DataFrame(excluded)


def fixed_horizon(full, horizons):
    rows = []
    for horizon in horizons:
        part = full.copy()
        part["scenario_id"] = f"fixed_{horizon:g}m"
        part["duration_minutes"] = np.minimum(part.true_time_to_failure_minutes, horizon)
        part["event_observed"] = (part.true_time_to_failure_minutes <= horizon).astype(int)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def rate_censor(full, rates):
    rows = []
    manifest = []
    for fold in sorted(full.fold.unique()):
        train = full[(full.fold == fold) & (full.subset == "train")]
        for rate in rates:
            horizon = float(train.true_time_to_failure_minutes.quantile(1 - rate))
            for subset in ("train", "validation", "test"):
                part = full[(full.fold == fold) & (full.subset == subset)].copy()
                part["scenario_id"] = f"target_censoring_{rate:.2f}"
                part["target_censoring_rate"] = rate
                part["administrative_horizon_minutes"] = horizon
                part["duration_minutes"] = np.minimum(part.true_time_to_failure_minutes, horizon)
                part["event_observed"] = (part.true_time_to_failure_minutes <= horizon).astype(int)
                rows.append(part)
                manifest.append(
                    {
                        "fold": fold,
                        "subset": subset,
                        "target_censoring_rate": rate,
                        "training_only_horizon_minutes": horizon,
                        "achieved_censoring_rate": float(1 - part.event_observed.mean())
                        if len(part)
                        else np.nan,
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(manifest)
