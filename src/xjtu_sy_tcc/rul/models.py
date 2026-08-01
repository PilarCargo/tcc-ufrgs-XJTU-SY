"""Auditable model families and training-only preprocessing for Phase 4."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from xjtu_sy_tcc.config.phase4 import Phase4Config


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment: str
    model_family: str
    input_columns: tuple[str, ...]
    complexity_order: int


@dataclass(slots=True)
class FeatureTransformer:
    columns: tuple[str, ...]
    log1p_columns: tuple[str, ...]
    scaler: RobustScaler | None
    fitted_bearings: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.columns].to_numpy(dtype=float).copy()
        indices = [self.columns.index(name) for name in self.log1p_columns]
        if indices:
            if np.any(values[:, indices] < 0):
                raise ValueError("log1p predictive transformation received negative values")
            values[:, indices] = np.log1p(values[:, indices])
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("Predictive preprocessing generated non-finite values")
        return values


@dataclass(slots=True)
class ModelBundle:
    spec: ExperimentSpec
    parameters: dict[str, object]
    transformer: FeatureTransformer | None
    estimator: object
    preprocessing_fit_seconds: float
    training_seconds: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.model_family == "dummy":
            matrix = np.zeros((len(frame), 1), dtype=float)
        elif self.transformer is not None:
            matrix = self.transformer.transform(frame)
        else:
            raise ValueError("Non-dummy model has no fitted transformer")
        values = np.asarray(self.estimator.predict(matrix), dtype=float)  # type: ignore[union-attr]
        if not np.all(np.isfinite(values)):
            raise ValueError("Model generated non-finite predictions")
        return values


def experiment_specs(
    selected: tuple[str, ...], all_features: tuple[str, ...]
) -> tuple[ExperimentSpec, ...]:
    time_columns = ("elapsed_minutes", "rotation_rpm", "radial_load_kn")
    return (
        ExperimentSpec("dummy_median", "dummy", (), 0),
        ExperimentSpec("time_only_ridge", "ridge", time_columns, 1),
        ExperimentSpec("selected_features_ridge", "ridge", selected, 1),
        ExperimentSpec("selected_features_random_forest", "random_forest", selected, 3),
        ExperimentSpec(
            "selected_features_hist_gradient_boosting", "hist_gradient_boosting", selected, 2
        ),
        ExperimentSpec("all_features_ridge", "ridge", all_features, 1),
        ExperimentSpec("all_features_random_forest", "random_forest", all_features, 3),
        ExperimentSpec(
            "all_features_hist_gradient_boosting", "hist_gradient_boosting", all_features, 2
        ),
        ExperimentSpec("selected_features_plus_time_ridge", "ridge", selected + time_columns, 1),
    )


def candidates(spec: ExperimentSpec, config: Phase4Config) -> tuple[dict[str, object], ...]:
    if spec.model_family == "dummy":
        return ({"strategy": "median"},)
    if spec.model_family == "ridge":
        return tuple({"alpha": alpha} for alpha in config.ridge_alphas)
    if spec.model_family == "random_forest":
        return tuple(asdict(item) for item in config.random_forest_candidates)
    if spec.model_family == "hist_gradient_boosting":
        return tuple(asdict(item) for item in config.hist_gradient_boosting_candidates)
    raise ValueError(f"Unknown model family: {spec.model_family}")


def fit_model(
    spec: ExperimentSpec,
    parameters: dict[str, object],
    training: pd.DataFrame,
    sample_weight: np.ndarray,
    config: Phase4Config,
) -> ModelBundle:
    """Fit preprocessing and estimator using training rows and weights only."""
    preprocessing_started = time.perf_counter()
    transformer = None
    if spec.model_family != "dummy":
        transformer = fit_transformer(
            training,
            spec.input_columns,
            spec.model_family == "ridge",
            config.heavy_tail_feature_tokens,
        )
        matrix = transformer.transform(training)
    else:
        matrix = np.zeros((len(training), 1), dtype=float)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    estimator = _estimator(spec.model_family, parameters, config.random_seed)
    training_started = time.perf_counter()
    estimator.fit(
        matrix, training[config.target_column].to_numpy(float), sample_weight=sample_weight
    )
    training_seconds = time.perf_counter() - training_started
    return ModelBundle(
        spec,
        dict(parameters),
        transformer,
        estimator,
        preprocessing_seconds,
        training_seconds,
    )


def fit_transformer(
    training: pd.DataFrame,
    columns: tuple[str, ...],
    scale: bool,
    heavy_tail_tokens: tuple[str, ...],
) -> FeatureTransformer:
    bearings = tuple(sorted(training["bearing_id"].unique()))
    log_columns = tuple(
        name for name in columns if any(token in name for token in heavy_tail_tokens)
    )
    transformer = FeatureTransformer(columns, log_columns, None, bearings)
    if scale:
        balanced = _balanced_matrix(training, columns, 100)
        temporary = pd.DataFrame(balanced, columns=columns)
        transformed = FeatureTransformer(columns, log_columns, None, bearings).transform(temporary)
        transformer.scaler = RobustScaler().fit(transformed)
    return transformer


def _balanced_matrix(training: pd.DataFrame, columns: tuple[str, ...], points: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, points)
    matrices = []
    for _, group in training.groupby("bearing_id", sort=True):
        ordered = group.sort_values("sequence_index")
        source = np.linspace(0.0, 1.0, len(ordered))
        values = ordered.loc[:, columns].to_numpy(float)
        matrices.append(
            np.column_stack(
                [np.interp(grid, source, values[:, index]) for index in range(len(columns))]
            )
        )
    return np.vstack(matrices)


def _estimator(family: str, parameters: dict[str, object], seed: int):
    if family == "dummy":
        return DummyRegressor(strategy="median")
    if family == "ridge":
        return Ridge(alpha=float(parameters["alpha"]))
    if family == "random_forest":
        return RandomForestRegressor(
            **parameters,
            random_state=seed,
            n_jobs=-1,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**parameters, random_state=seed)
    raise ValueError(f"Unknown model family: {family}")
