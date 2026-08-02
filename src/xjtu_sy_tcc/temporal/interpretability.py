"""Feature occlusion importance for frozen temporal models."""

import numpy as np
import pandas as pd

from xjtu_sy_tcc.rul.metrics import regression_metrics
from xjtu_sy_tcc.temporal.model import predict


def occlusion_importance(
    model, x, context, y, bearings, preprocessor, batch_size, device, fold, experiment
):
    base = preprocessor.inverse_target(predict(model, x, context, batch_size, device))
    base_score = _macro(y, base, bearings)
    rows = []
    for index, feature in enumerate(preprocessor.feature_names):
        changed = x.copy()
        changed[:, :, index] = 0
        values = preprocessor.inverse_target(predict(model, changed, context, batch_size, device))
        score = _macro(y, values, bearings)
        rows.append(
            {
                "fold": fold,
                "experiment": experiment,
                "feature": feature,
                "baseline_macro_mae": base_score,
                "occluded_macro_mae": score,
                "importance_macro_mae_increase": score - base_score,
                "scope": "validation development",
            }
        )
    return pd.DataFrame(rows)


def _macro(y, p, bearings):
    return float(
        np.mean(
            [
                regression_metrics(
                    y[np.asarray(bearings) == b], np.maximum(p[np.asarray(bearings) == b], 0)
                )["mae"]
                for b in sorted(set(bearings))
            ]
        )
    )
