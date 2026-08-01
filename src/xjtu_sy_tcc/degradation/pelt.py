"""Unsupervised PELT estimated-degradation-onset detection and sensitivity analysis."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import ruptures as rpt

from xjtu_sy_tcc.config.phase3 import Phase3Config
from xjtu_sy_tcc.degradation.health import smooth_indicator


@dataclass(frozen=True, slots=True)
class DetectionParameters:
    cost_model: str
    penalty: float
    minimum_segment_size: int
    jump: int
    smoothing_window: int
    persistence_window: int
    minimum_effect_size: float


def detect_estimated_onset(
    bearing: pd.DataFrame,
    parameters: DetectionParameters,
    baseline_count: int,
) -> dict[str, object]:
    """Return the earliest persistent positive level shift or explicit ``not_detected``.

    PELT candidates are segment boundaries. A candidate must follow the calibration interval,
    leave a full persistence window, have a positive robust median shift of at least the configured
    effect size, and keep at least 70% of the persistence window above the pre-change median.
    """
    ordered = bearing.sort_values("sequence_index")
    centered = ordered["baseline_centered_health_indicator"].reset_index(drop=True)
    smoothed = smooth_indicator(centered, "median", parameters.smoothing_window).to_numpy(float)
    size = len(smoothed)
    common = {
        **asdict(parameters),
        "estimated_onset_status": "not_detected",
        "estimated_onset_sequence_index": None,
        "estimated_onset_elapsed_minutes": None,
        "rul_at_estimated_onset": None,
        "life_fraction_at_estimated_onset": None,
        "pre_change_median": None,
        "post_change_median": None,
        "robust_effect_size": None,
        "persistence_duration": 0,
    }
    if size < max(
        2 * parameters.minimum_segment_size, baseline_count + parameters.persistence_window
    ):
        return {**common, "rejection_reasons": ["trajectory_too_short"]}
    signal = smoothed.reshape(-1, 1)
    try:
        boundaries = (
            rpt.Pelt(
                model=parameters.cost_model,
                min_size=parameters.minimum_segment_size,
                jump=parameters.jump,
            )
            .fit(signal)
            .predict(pen=parameters.penalty)
        )
    except Exception as exc:
        return {**common, "rejection_reasons": [f"pelt_error:{type(exc).__name__}"]}
    candidates = [int(item) for item in boundaries if item < size]
    rejections: list[str] = []
    for candidate in candidates:
        reasons = []
        if candidate < baseline_count:
            reasons.append("inside_initial_calibration_interval")
        if candidate < parameters.persistence_window:
            reasons.append("insufficient_pre_change_window")
        if size - candidate < parameters.persistence_window:
            reasons.append("insufficient_persistent_post_change_window")
        if reasons:
            rejections.extend(f"candidate_{candidate}:{reason}" for reason in reasons)
            continue
        pre = smoothed[candidate - parameters.persistence_window : candidate]
        post = smoothed[candidate : candidate + parameters.persistence_window]
        pre_median = float(np.median(pre))
        post_median = float(np.median(post))
        scale = _robust_scale(smoothed[:baseline_count])
        effect = (post_median - pre_median) / scale
        persistence = float(np.mean(post > pre_median))
        if effect < parameters.minimum_effect_size:
            rejections.append(f"candidate_{candidate}:minimum_effect_size")
            continue
        if persistence < 0.7:
            rejections.append(f"candidate_{candidate}:persistence")
            continue
        row = ordered.iloc[candidate]
        return {
            **common,
            "estimated_onset_status": "detected",
            "estimated_onset_sequence_index": int(row["sequence_index"]),
            "estimated_onset_elapsed_minutes": float(row["elapsed_minutes"]),
            "rul_at_estimated_onset": float(row["rul_minutes"]),
            "life_fraction_at_estimated_onset": candidate / (size - 1),
            "pre_change_median": pre_median,
            "post_change_median": post_median,
            "robust_effect_size": float(effect),
            "persistence_duration": parameters.persistence_window,
            "rejection_reasons": rejections,
        }
    if not candidates:
        rejections.append("pelt_returned_no_internal_breakpoint")
    return {**common, "rejection_reasons": rejections}


def sensitivity_grid(config: Phase3Config) -> tuple[DetectionParameters, ...]:
    pelt = config.pelt
    return tuple(
        DetectionParameters(
            cost,
            penalty,
            minimum,
            pelt.primary_jump,
            smoothing,
            persistence,
            pelt.minimum_effect_size,
        )
        for cost, penalty, smoothing, minimum, persistence in itertools.product(
            pelt.cost_models,
            pelt.penalties,
            pelt.smoothing_windows,
            pelt.minimum_segment_sizes,
            pelt.persistence_windows,
        )
    )


def run_sensitivity(health: pd.DataFrame, config: Phase3Config) -> pd.DataFrame:
    """Evaluate the grid on training and validation bearings only, never test bearings."""
    rows = []
    eligible = health[health["subset"].isin(["train", "validation"])]
    for (fold, subset, condition, bearing), group in eligible.groupby(
        ["fold", "subset", "condition_id", "bearing_id"], sort=True
    ):
        baseline = int(group["baseline_acquisitions"].iloc[0])
        for parameter_id, parameters in enumerate(sensitivity_grid(config), start=1):
            detection = detect_estimated_onset(group, parameters, baseline)
            rows.append(
                {
                    "fold": int(fold),
                    "subset": subset,
                    "condition_id": int(condition),
                    "bearing_id": bearing,
                    "parameter_id": parameter_id,
                    **detection,
                }
            )
    result = pd.DataFrame(rows)
    return add_stability_metrics(result, config)


def add_stability_metrics(sensitivity: pd.DataFrame, config: Phase3Config) -> pd.DataFrame:
    """Attach per-bearing median, IQR, normalized variability, detection rate, and agreement."""
    result = sensitivity.copy()
    metrics = []
    for keys, group in result.groupby(["fold", "subset", "condition_id", "bearing_id"], sort=True):
        detected = group[group["estimated_onset_status"] == "detected"]
        values = detected["estimated_onset_elapsed_minutes"].astype(float)
        life = detected["life_fraction_at_estimated_onset"].astype(float)
        median = float(values.median()) if len(values) else np.nan
        iqr = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
        life_iqr = float(life.quantile(0.75) - life.quantile(0.25)) if len(life) else np.nan
        agreement = (
            float((np.abs(values - median) <= config.pelt.agreement_minutes).mean())
            if len(values)
            else 0.0
        )
        life_agreement = (
            float((np.abs(life - life.median()) <= config.pelt.agreement_life_fraction).mean())
            if len(life)
            else 0.0
        )
        metrics.append(
            (*keys, median, iqr, life_iqr, len(detected) / len(group), agreement, life_agreement)
        )
    columns = [
        "fold",
        "subset",
        "condition_id",
        "bearing_id",
        "median_onset_minutes",
        "onset_iqr_minutes",
        "normalized_onset_iqr",
        "detection_proportion",
        "agreement_within_minutes",
        "agreement_within_life_fraction",
    ]
    return result.merge(pd.DataFrame(metrics, columns=columns), on=columns[:4], how="left")


def choose_parameters(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations on train+validation only.

    Score = 0.35 detection rate + 0.25 agreement with the bearing grid median +
    0.15 median clipped effect + 0.15 condition coverage + 0.10 avoidance of detections
    in the final 5% of life. Ties use parameter ID.
    """
    rows = []
    for (fold, parameter_id), group in sensitivity.groupby(["fold", "parameter_id"], sort=True):
        detected = group[group["estimated_onset_status"] == "detected"]
        detection_rate = len(detected) / len(group)
        effect = (
            float(np.clip(detected["robust_effect_size"].astype(float).median() / 5, 0, 1))
            if len(detected)
            else 0.0
        )
        coverage = detected["condition_id"].nunique() / 3
        agreement = (
            float(
                (
                    np.abs(
                        detected["estimated_onset_elapsed_minutes"].astype(float)
                        - detected["median_onset_minutes"].astype(float)
                    )
                    <= 10.0
                ).mean()
            )
            if len(detected)
            else 0.0
        )
        late_rate = (
            float((detected["life_fraction_at_estimated_onset"].astype(float) >= 0.95).mean())
            if len(detected)
            else 1.0
        )
        score = (
            0.35 * detection_rate
            + 0.25 * agreement
            + 0.15 * effect
            + 0.15 * coverage
            + 0.10 * (1 - late_rate)
        )
        first = group.iloc[0]
        rows.append(
            {
                "fold": int(fold),
                "parameter_id": int(parameter_id),
                "selection_score": score,
                "detection_rate": detection_rate,
                "median_grid_agreement": agreement,
                "median_clipped_effect": effect,
                "condition_coverage": coverage,
                "final_five_percent_rate": late_rate,
                **{
                    name: first[name]
                    for name in (
                        "cost_model",
                        "penalty",
                        "minimum_segment_size",
                        "jump",
                        "smoothing_window",
                        "persistence_window",
                        "minimum_effect_size",
                    )
                },
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["fold", "selection_score", "parameter_id"], ascending=[True, False, True]
    )
    ranking["configuration_rank"] = ranking.groupby("fold").cumcount() + 1
    ranking["selected_configuration"] = ranking["configuration_rank"] == 1
    return ranking.reset_index(drop=True)


def parameters_from_row(row: pd.Series) -> DetectionParameters:
    return DetectionParameters(
        str(row["cost_model"]),
        float(row["penalty"]),
        int(row["minimum_segment_size"]),
        int(row["jump"]),
        int(row["smoothing_window"]),
        int(row["persistence_window"]),
        float(row["minimum_effect_size"]),
    )


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale == 0:
        scale = float(np.std(values))
    return max(scale, np.finfo(float).eps)
