"""RUL prediction metrics at acquisition, bearing, fold, condition, and post-hoc scopes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or actual.size == 0:
        raise ValueError("Metrics require equal non-empty one-dimensional arrays")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Metrics require finite values")
    errors = predicted - actual
    r2 = float(r2_score(actual, predicted)) if actual.size >= 2 and np.ptp(actual) > 0 else math.nan
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "r2": r2,
        "signed_error": float(np.mean(errors)),
    }


def add_prediction_diagnostics(frame: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    values = np.asarray(raw, dtype=float)
    if len(values) != len(result) or not np.all(np.isfinite(values)):
        raise ValueError("Predictions must be finite and aligned")
    result["prediction_raw_minutes"] = values
    result["prediction_non_negative_minutes"] = np.maximum(values, 0.0)
    result["prediction_error_minutes"] = (
        result["prediction_non_negative_minutes"] - result["rul_minutes"]
    )
    result["absolute_error_minutes"] = result["prediction_error_minutes"].abs()
    result["squared_error_minutes2"] = result["prediction_error_minutes"].pow(2)
    return result


def monotonicity_violations(predicted: np.ndarray) -> dict[str, float]:
    differences = np.diff(np.asarray(predicted, dtype=float))
    increases = differences[differences > 0]
    return {
        "monotonicity_violation_count": int(len(increases)),
        "monotonicity_violation_magnitude": float(increases.sum()),
        "monotonicity_violation_rate": float(len(increases) / len(differences))
        if len(differences)
        else 0.0,
    }


def bearing_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["experiment", "model", "fold", "condition_id", "bearing_id"]
    for values, group in predictions.groupby(keys, sort=True):
        ordered = group.sort_values("sequence_index")
        metrics = regression_metrics(
            ordered["rul_minutes"].to_numpy(), ordered["prediction_non_negative_minutes"].to_numpy()
        )
        lifetime = float(ordered["rul_minutes"].max())
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                **metrics,
                "normalized_mae": metrics["mae"] / lifetime if lifetime > 0 else math.nan,
                "normalized_rmse": metrics["rmse"] / lifetime if lifetime > 0 else math.nan,
                "negative_raw_prediction_count": int((ordered["prediction_raw_minutes"] < 0).sum()),
                "acquisition_count": len(ordered),
                **monotonicity_violations(ordered["prediction_non_negative_minutes"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in predictions.groupby(group_columns, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        metrics = regression_metrics(
            group["rul_minutes"].to_numpy(), group["prediction_non_negative_minutes"].to_numpy()
        )
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                **metrics,
                "acquisition_count": len(group),
                "bearing_count": group["bearing_id"].nunique(),
                "negative_raw_prediction_count": int((group["prediction_raw_minutes"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def fold_metrics(bearing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in bearing.groupby(["experiment", "model", "fold"], sort=True):
        rows.append(
            {
                "experiment": values[0],
                "model": values[1],
                "fold": values[2],
                "macro_mae": float(group["mae"].mean()),
                "macro_rmse": float(group["rmse"].mean()),
                "macro_r2": float(group["r2"].mean()) if group["r2"].notna().any() else math.nan,
                "macro_normalized_mae": float(group["normalized_mae"].mean()),
                "test_bearing_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def assign_life_stage(frame: pd.DataFrame, thresholds: tuple[float, float]) -> pd.Series:
    consumed = frame.groupby("bearing_id")["sequence_index"].transform(
        lambda values: values / values.max() if values.max() > 0 else 0.0
    )
    return pd.cut(
        consumed,
        bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
        labels=["early", "intermediate", "late"],
        right=False,
    ).astype(str)
