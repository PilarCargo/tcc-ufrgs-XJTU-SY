"""Training-only feature, context, and weighted target preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import RobustScaler


@dataclass(slots=True)
class TemporalPreprocessor:
    feature_names: list[str]
    log_mask: np.ndarray
    feature_scaler: RobustScaler
    context_scaler: RobustScaler
    target_center: float
    target_scale: float
    training_bearings: tuple[str, ...]

    def transform_features(self, x):
        z = np.asarray(x, float).copy()
        z[..., self.log_mask] = np.log1p(z[..., self.log_mask])
        shape = z.shape
        return (
            self.feature_scaler.transform(z.reshape(-1, shape[-1])).reshape(shape).astype("float32")
        )

    def transform_context(self, x):
        return self.context_scaler.transform(x).astype("float32")

    def scale_target(self, y):
        return ((y - self.target_center) / self.target_scale).astype("float32")

    def inverse_target(self, y):
        return np.asarray(y) * self.target_scale + self.target_center


def fit_preprocessor(sequences, feature_names, tokens):
    x = sequences.features
    mask = np.array([any(t in f for t in tokens) for f in feature_names])
    if np.any(x[..., mask] < 0):
        raise ValueError("log1p feature has negative values")
    transformed = x.astype(float).copy()
    transformed[..., mask] = np.log1p(transformed[..., mask])
    balanced = []
    balanced_context = []
    bearings = sequences.metadata.bearing_id.to_numpy()
    for bearing in sorted(set(bearings)):
        indices = np.flatnonzero(bearings == bearing)
        selected = indices[
            np.linspace(0, len(indices) - 1, min(100, len(indices))).round().astype(int)
        ]
        balanced.append(transformed[selected].reshape(-1, len(feature_names)))
        balanced_context.append(sequences.context[selected])
    fs = RobustScaler().fit(np.concatenate(balanced))
    cs = RobustScaler().fit(np.concatenate(balanced_context))
    w = sequences.weights.astype(float)
    y = sequences.targets.astype(float)
    center = float(np.average(y, weights=w))
    scale = float(np.sqrt(np.average((y - center) ** 2, weights=w)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid target scale")
    return TemporalPreprocessor(
        feature_names,
        mask,
        fs,
        cs,
        center,
        scale,
        tuple(sorted(sequences.metadata.bearing_id.unique())),
    )
