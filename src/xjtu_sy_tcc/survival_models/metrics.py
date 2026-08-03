"""Survival metrics aggregated with the bearing as inferential unit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw

from xjtu_sy_tcc.survival_models.features import structured_target
from xjtu_sy_tcc.survival_models.models import median_survival


def supported_grid(train: pd.DataFrame, test: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    upper = min(float(train.duration_minutes.max()), float(test.duration_minutes.max()))
    lower = max(
        0.0,
        float(train.duration_minutes.min()),
        float(test.duration_minutes.min()),
    )
    return grid[(grid > lower) & (grid < upper)]


def evaluate_curves(
    train: pd.DataFrame, test: pd.DataFrame, curves: np.ndarray, grid: np.ndarray
) -> tuple[dict, pd.DataFrame]:
    """Calculate library IPCW metrics, explicitly excluding unsupported times."""
    valid = supported_grid(train, test, grid)
    if len(valid) < 2:
        raise ValueError("Fewer than two supported survival evaluation times")
    positions = np.searchsorted(grid, valid)
    estimate = curves[:, positions]
    y_train, y_test = structured_target(train), structured_target(test)
    times, scores = brier_score(y_train, y_test, estimate, valid)
    ibs = float(np.trapezoid(scores, times) / (times[-1] - times[0]))
    risk = -np.trapezoid(estimate, times, axis=1)
    harrell = concordance_index_censored(y_test["event"], y_test["time"], risk)
    try:
        uno = concordance_index_ipcw(y_train, y_test, risk, tau=float(times[-1]))
        uno_value = float(uno[0])
    except ValueError:
        uno_value = np.nan
    curve = pd.DataFrame({"evaluation_time": times, "brier_score": scores})
    return {
        "integrated_brier_score": ibs,
        "harrell_c_index": float(harrell[0]),
        "harrell_comparable_pairs": int(harrell[2] + harrell[3]),
        "ipcw_c_index": uno_value,
        "supported_grid_points": len(times),
        "grid_lower": float(times[0]),
        "grid_upper": float(times[-1]),
    }, curve


def bearing_metrics(
    train: pd.DataFrame, test: pd.DataFrame, curves: np.ndarray, grid: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, brier_rows = [], []
    for bearing, group in test.groupby("bearing_id", sort=True):
        positions = test.index.get_indexer(group.index)
        try:
            metrics, brier = evaluate_curves(train, group, curves[positions], grid)
            metrics["valid"] = True
            for record in brier.to_dict("records"):
                record["bearing_id"] = bearing
                brier_rows.append(record)
        except ValueError as error:
            metrics = {
                "integrated_brier_score": np.nan,
                "harrell_c_index": np.nan,
                "ipcw_c_index": np.nan,
                "valid": False,
                "failure_reason": str(error),
            }
        medians = median_survival(curves[positions], grid)
        truth = group.duration_minutes.to_numpy(float)
        available = np.isfinite(medians)
        metrics.update(
            {
                "bearing_id": bearing,
                "condition_id": int(group.condition_id.iloc[0]),
                "landmark_count": len(group),
                "median_survival_coverage": float(available.mean()),
                "median_survival_mae": float(np.mean(np.abs(medians[available] - truth[available])))
                if available.any()
                else np.nan,
                "median_survival_rmse": float(
                    np.sqrt(np.mean((medians[available] - truth[available]) ** 2))
                )
                if available.any()
                else np.nan,
                "median_survival_signed_error": float(
                    np.mean(medians[available] - truth[available])
                )
                if available.any()
                else np.nan,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows), pd.DataFrame(brier_rows)


def horizon_predictions(
    curves: np.ndarray, grid: np.ndarray, horizons: tuple[float, ...]
) -> dict[float, np.ndarray]:
    result = {}
    for horizon in horizons:
        if horizon < grid[0] or horizon > grid[-1]:
            result[horizon] = np.full(len(curves), np.nan)
        else:
            result[horizon] = np.asarray([np.interp(horizon, grid, curve) for curve in curves])
    return result


def validate_survival_curves(curves: np.ndarray) -> None:
    if not np.isfinite(curves).all() or (curves < -1e-12).any() or (curves > 1 + 1e-12).any():
        raise ValueError("Invalid survival probability")
    if (np.diff(curves, axis=1) > 1e-10).any():
        raise ValueError("Survival functions must be non-increasing")


def rank_candidates(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fold",
        "scenario",
        "model",
        "validation_macro_ibs",
        "validation_main_horizon_brier",
        "validation_macro_ipcw_c_index",
        "selected_feature_count",
        "complexity_order",
        "candidate_id",
    ]
    if set(columns) - set(records):
        raise ValueError("Candidate selection columns are missing")
    ranked = records.sort_values(
        [
            "fold",
            "scenario",
            "model",
            "validation_macro_ibs",
            "validation_main_horizon_brier",
            "validation_macro_ipcw_c_index",
            "selected_feature_count",
            "complexity_order",
            "candidate_id",
        ],
        ascending=[True, True, True, True, True, False, True, True, True],
        kind="mergesort",
    ).copy()
    ranked["selection_rank"] = ranked.groupby(["fold", "scenario", "model"]).cumcount() + 1
    ranked["selected"] = ranked.selection_rank == 1
    ranked["tie_breaking_rule"] = (
        "macro_ibs,brier,ipcw_c_index,feature_count,complexity,candidate_order"
    )
    return ranked.reset_index(drop=True)


def calibration_table(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.dropna(subset=["failure_probability"]).groupby(
        ["fold", "scenario", "model", "horizon_minutes"]
    ):
        ordered = group.copy()
        unique = ordered.failure_probability.nunique()
        ordered["calibration_bin"] = (
            pd.qcut(ordered.failure_probability, min(bins, unique), duplicates="drop", labels=False)
            if unique > 1
            else 0
        )
        for bin_id, part in ordered.groupby("calibration_bin"):
            horizon = float(keys[3])
            known = (
                part.event_observed.eq(1) & part.duration_minutes.le(horizon)
            ) | part.duration_minutes.gt(horizon)
            scoped = part[known]
            observed = (
                (scoped.event_observed.eq(1) & scoped.duration_minutes.le(horizon)).mean()
                if len(scoped)
                else np.nan
            )
            rows.append(
                {
                    "fold": keys[0],
                    "scenario": keys[1],
                    "model": keys[2],
                    "horizon_minutes": horizon,
                    "calibration_bin": int(bin_id),
                    "mean_predicted_failure_probability": float(part.failure_probability.mean()),
                    "observed_event_probability": float(observed),
                    "calibration_difference": float(part.failure_probability.mean() - observed)
                    if np.isfinite(observed)
                    else np.nan,
                    "landmark_count": len(part),
                    "known_outcome_count": len(scoped),
                    "sparse_bin": len(scoped) < 10,
                }
            )
    return pd.DataFrame(rows)
