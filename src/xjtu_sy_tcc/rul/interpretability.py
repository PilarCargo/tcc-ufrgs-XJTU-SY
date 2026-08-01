"""Non-causal Ridge coefficients and deterministic validation permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xjtu_sy_tcc.rul.metrics import bearing_metrics
from xjtu_sy_tcc.rul.models import ModelBundle


def ridge_coefficients(bundle: ModelBundle, fold: int) -> pd.DataFrame:
    if bundle.spec.model_family != "ridge":
        return pd.DataFrame()
    coefficients = np.asarray(bundle.estimator.coef_, dtype=float)  # type: ignore[union-attr]
    return pd.DataFrame(
        {
            "fold": fold,
            "experiment": bundle.spec.experiment,
            "feature": bundle.spec.input_columns,
            "coefficient": coefficients,
            "coefficient_direction": np.where(coefficients >= 0, "positive", "negative"),
            "absolute_coefficient": np.abs(coefficients),
            "interpretation_scope": "scaled coefficient; association not causality",
        }
    )


def permutation_importance(
    bundle: ModelBundle,
    validation: pd.DataFrame,
    base_predictions: pd.DataFrame,
    fold: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Measure validation macro-MAE increase after deterministic feature permutation."""
    if bundle.spec.model_family not in {"random_forest", "hist_gradient_boosting"}:
        return pd.DataFrame()
    base = float(bearing_metrics(base_predictions)["mae"].mean())
    rows = []
    for feature_index, feature in enumerate(bundle.spec.input_columns):
        values = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + fold * 10_000 + feature_index * 100 + repeat)
            permuted = validation.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            predicted = base_predictions.copy()
            raw = bundle.predict(permuted)
            predicted["prediction_raw_minutes"] = raw
            predicted["prediction_non_negative_minutes"] = np.maximum(raw, 0)
            values.append(float(bearing_metrics(predicted)["mae"].mean()) - base)
        rows.append(
            {
                "fold": fold,
                "experiment": bundle.spec.experiment,
                "model": bundle.spec.model_family,
                "feature": feature,
                "importance_mean_macro_mae_increase": float(np.mean(values)),
                "importance_std": float(np.std(values, ddof=0)),
                "repeats": repeats,
                "scope": "validation development analysis; not causality",
            }
        )
    return pd.DataFrame(rows)
