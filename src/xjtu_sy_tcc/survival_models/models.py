"""Auditable Kaplan-Meier, regularized Cox, and random survival forest models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import RobustScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from xjtu_sy_tcc.survival_models.features import balanced_indices, structured_target


@dataclass(slots=True)
class SurvivalBundle:
    model_name: str
    model: object
    features: list[str]
    scaler: RobustScaler | None
    grid: np.ndarray
    parameters: dict[str, object]

    def curves(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_name == "landmark_km_baseline":
            upper = float(self.model.domain[1])  # type: ignore[union-attr]
            curve = self.model(np.minimum(self.grid, upper))  # type: ignore[operator]
            return np.tile(curve, (len(data), 1))
        x = data[self.features]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        functions = self.model.predict_survival_function(x)  # type: ignore[union-attr]
        return np.asarray(
            [function(np.minimum(self.grid, float(function.domain[1]))) for function in functions]
        )


def evaluation_grid(
    train: pd.DataFrame, scenario: str, quantiles: tuple[float, float], points: int
) -> np.ndarray:
    times = train.duration_minutes.to_numpy(float)
    if scenario.startswith("fixed_horizon_"):
        upper = float(scenario.removeprefix("fixed_horizon_"))
        lower = max(float(np.min(times[times > 0])), upper / points)
    else:
        lower, upper = np.quantile(times[train.event_observed.to_numpy(bool)], quantiles)
        upper = min(float(upper), float(np.nextafter(times.max(), 0)))
    if upper <= lower:
        raise ValueError("Training-only survival grid has no supported interval")
    return np.linspace(lower, upper, points)


def fit_km(train: pd.DataFrame, grid: np.ndarray, seed: int) -> SurvivalBundle:
    sampled = train.loc[balanced_indices(train, seed)]
    time, survival = kaplan_meier_estimator(
        sampled.event_observed.astype(bool), sampled.duration_minutes
    )
    function = StepFunction(time, survival, a=1.0, b=0.0)
    return SurvivalBundle(
        "landmark_km_baseline", function, [], None, grid, {"balancing": "deterministic_equal_count"}
    )


def fit_cox(
    train: pd.DataFrame, features: list[str], grid: np.ndarray, alpha: float, seed: int, name: str
) -> SurvivalBundle:
    sampled = train.loc[balanced_indices(train, seed)]
    scaler = RobustScaler().fit(sampled[features])
    x = scaler.transform(sampled[features])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = CoxPHSurvivalAnalysis(alpha=alpha).fit(x, structured_target(sampled))
    return SurvivalBundle(
        name,
        model,
        features,
        scaler,
        grid,
        {"alpha": alpha, "balancing": "deterministic_equal_count"},
    )


def fit_rsf(
    train: pd.DataFrame,
    features: list[str],
    grid: np.ndarray,
    parameters: dict[str, object],
    seed: int,
) -> SurvivalBundle:
    weights = train.landmark_weight.to_numpy(float)
    totals = train.assign(_w=weights).groupby("bearing_id")._w.sum()
    if not np.allclose(totals, totals.iloc[0]):
        raise ValueError("Landmark weights are not bearing-balanced")
    model = RandomSurvivalForest(**parameters, random_state=seed, n_jobs=-1).fit(
        train[features], structured_target(train), sample_weight=weights
    )
    return SurvivalBundle(
        "random_survival_forest",
        model,
        features,
        None,
        grid,
        {**parameters, "balancing": "inverse_landmark_frequency_weights"},
    )


def median_survival(curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    result = np.full(len(curves), np.nan)
    for index, curve in enumerate(curves):
        positions = np.flatnonzero(curve <= 0.5)
        if len(positions):
            result[index] = grid[positions[0]]
    return result
