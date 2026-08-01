"""Retrospective life-stage and estimated-onset annotations for test evaluation only."""

from __future__ import annotations

import pandas as pd

from xjtu_sy_tcc.rul.metrics import assign_life_stage, grouped_metrics


def annotate_posthoc(
    predictions: pd.DataFrame,
    test_onsets: pd.DataFrame,
    thresholds: tuple[float, float],
) -> pd.DataFrame:
    """Annotate test rows after prediction; onset values never enter a model matrix."""
    if (
        len(test_onsets) != 15
        or test_onsets["bearing_id"].nunique() != 15
        or set(test_onsets["subset"]) != {"test"}
    ):
        raise ValueError("Exactly 15 unique test-onset rows are required")
    result = predictions.copy()
    result["life_stage"] = assign_life_stage(result, thresholds)
    onset_map = test_onsets.set_index(["fold", "bearing_id"])[
        ["estimated_onset_status", "estimated_onset_sequence_index"]
    ]
    regions = []
    for row in result[["fold", "bearing_id", "sequence_index"]].itertuples(index=False):
        onset = onset_map.loc[(row.fold, row.bearing_id)]
        if onset["estimated_onset_status"] != "detected":
            regions.append("estimated onset not detected")
        elif row.sequence_index < int(onset["estimated_onset_sequence_index"]):
            regions.append("before estimated onset")
        else:
            regions.append("after estimated onset")
    result["estimated_onset_region"] = regions
    return result


def life_stage_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return grouped_metrics(predictions, ["experiment", "model", "life_stage"])


def onset_region_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return grouped_metrics(predictions, ["experiment", "model", "estimated_onset_region"])
