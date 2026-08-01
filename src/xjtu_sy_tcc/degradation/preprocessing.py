"""Training-only balanced transformations, robust scaling, and PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, RobustScaler

from xjtu_sy_tcc.config.phase3 import Phase3Config


@dataclass(slots=True)
class FittedPreprocessor:
    selected_features: tuple[str, ...]
    transformed_features: tuple[str, ...]
    transformation_mode: str
    power_transformer: PowerTransformer | None
    scaler: RobustScaler
    pca: PCA
    orientation: int
    training_bearings: tuple[str, ...]
    balance_points_per_bearing: int

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply frozen training parameters without fitting on the supplied frame."""
        values = frame.loc[:, self.selected_features].to_numpy(dtype=np.float64)
        values = _apply_transformation(
            values,
            self.selected_features,
            self.transformed_features,
            self.transformation_mode,
            self.power_transformer,
            fit=False,
        )
        result = self.pca.transform(self.scaler.transform(values))
        result[:, 0] *= self.orientation
        if not np.all(np.isfinite(result)):
            raise ValueError("Preprocessing generated non-finite PCA scores")
        return result

    def artifact(self) -> dict[str, object]:
        """Return a complete JSON-serializable record of fitted training parameters."""
        transformer = self.power_transformer
        return {
            "selected_features": list(self.selected_features),
            "transformed_features": list(self.transformed_features),
            "transformation_mode": self.transformation_mode,
            "power_transformer_lambdas": transformer.lambdas_.tolist()
            if transformer is not None
            else None,
            "scaler_center": self.scaler.center_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "pca_components": self.pca.components_.tolist(),
            "pca_mean": self.pca.mean_.tolist(),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "orientation": self.orientation,
            "training_bearings": list(self.training_bearings),
            "balancing_strategy": "linear_interpolation_per_bearing_without_extrapolation",
            "balance_points_per_bearing": self.balance_points_per_bearing,
        }


def fit_preprocessor(
    training: pd.DataFrame,
    selected_features: tuple[str, ...],
    config: Phase3Config,
) -> FittedPreprocessor:
    """Fit every learned parameter on equal-size trajectories from training bearings only."""
    training_bearings = tuple(sorted(training["bearing_id"].unique()))
    if len(training_bearings) != 9:
        raise ValueError("Preprocessor fitting requires exactly nine training bearings")
    balanced, progression = balanced_training_matrix(
        training, selected_features, config.balance_points_per_bearing
    )
    transformed_features = _transformed_feature_names(selected_features, config)
    transformer = (
        PowerTransformer(method="yeo-johnson", standardize=False)
        if config.transformation.mode == "yeo_johnson"
        else None
    )
    transformed = _apply_transformation(
        balanced,
        selected_features,
        transformed_features,
        config.transformation.mode,
        transformer,
        fit=True,
    )
    scaler = RobustScaler().fit(transformed)
    scaled = scaler.transform(transformed)
    pca = PCA(n_components=config.pca_components, svd_solver="full").fit(scaled)
    component = pca.transform(scaled)[:, 0]
    correlations = []
    points = config.balance_points_per_bearing
    for index in range(len(training_bearings)):
        values = component[index * points : (index + 1) * points]
        rho = float(spearmanr(values, progression[:points]).statistic)
        if np.isfinite(rho):
            correlations.append(rho)
    aggregate = float(np.median(correlations)) if correlations else 0.0
    orientation = 1 if aggregate >= 0 else -1
    return FittedPreprocessor(
        selected_features,
        transformed_features,
        config.transformation.mode,
        transformer,
        scaler,
        pca,
        orientation,
        training_bearings,
        config.balance_points_per_bearing,
    )


def balanced_training_matrix(
    training: pd.DataFrame, selected_features: tuple[str, ...], points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate every training bearing to the same number of normalized-life positions."""
    if points < 3:
        raise ValueError("Balanced sampling requires at least three points")
    grid = np.linspace(0.0, 1.0, points)
    matrices = []
    for _, group in training.groupby("bearing_id", sort=True):
        ordered = group.sort_values("sequence_index")
        source = np.linspace(0.0, 1.0, len(ordered))
        values = ordered.loc[:, selected_features].to_numpy(dtype=float)
        matrices.append(
            np.column_stack([np.interp(grid, source, values[:, i]) for i in range(values.shape[1])])
        )
    matrix = np.vstack(matrices)
    progression = np.tile(grid, len(matrices))
    return matrix, progression


def _transformed_feature_names(features: tuple[str, ...], config: Phase3Config) -> tuple[str, ...]:
    if config.transformation.mode == "auto_log1p_nonnegative":
        return tuple(
            name
            for name in features
            if any(token in name for token in config.transformation.feature_tokens)
        )
    if config.transformation.mode == "yeo_johnson":
        return features
    return ()


def _apply_transformation(
    values: np.ndarray,
    features: tuple[str, ...],
    transformed_features: tuple[str, ...],
    mode: str,
    transformer: PowerTransformer | None,
    *,
    fit: bool,
) -> np.ndarray:
    result = values.copy()
    indices = [features.index(name) for name in transformed_features]
    if mode == "auto_log1p_nonnegative" and indices:
        if np.any(result[:, indices] < 0):
            raise ValueError("log1p transformation received a negative feature value")
        result[:, indices] = np.log1p(result[:, indices])
    elif mode == "yeo_johnson":
        if transformer is None:
            raise ValueError("Yeo-Johnson transformer is unavailable")
        result = transformer.fit_transform(result) if fit else transformer.transform(result)
    elif mode != "none":
        raise ValueError(f"Unsupported transformation mode: {mode}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Transformation generated non-finite values")
    return result
