import numpy as np
import pandas as pd

from xjtu_sy_tcc.survival_models.metrics import (
    bearing_metrics,
    horizon_predictions,
    rank_candidates,
    validate_survival_curves,
)
from xjtu_sy_tcc.survival_models.models import (
    evaluation_grid,
    fit_cox,
    fit_km,
    fit_rsf,
    median_survival,
)


def frame() -> pd.DataFrame:
    rows = []
    for bearing, shift in (("A", 0), ("B", 1), ("C", 2)):
        for time in range(1, 9):
            rows.append(
                {
                    "bearing_id": bearing,
                    "condition_id": shift + 1,
                    "duration_minutes": float(time),
                    "event_observed": int(time < 8),
                    "landmark_weight": 1.0,
                    "x": time + shift,
                    "z": np.log1p(time + shift),
                }
            )
    return pd.DataFrame(rows)


def test_training_grid_is_deterministic_and_fixed_horizon_bounded():
    data = frame()
    first = evaluation_grid(data, "fixed_horizon_6", (0.1, 0.9), 5)
    second = evaluation_grid(data, "fixed_horizon_6", (0.1, 0.9), 5)
    assert np.array_equal(first, second)
    assert first[-1] == 6


def test_km_cox_and_rsf_survival_shapes_and_bounds():
    data = frame()
    grid = np.linspace(1.1, 6.9, 6)
    bundles = [
        fit_km(data, grid, 42),
        fit_cox(data, ["x", "z"], grid, 1.0, 42, "cox_causal_features"),
        fit_rsf(
            data,
            ["x", "z"],
            grid,
            {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            42,
        ),
    ]
    for bundle in bundles:
        curves = bundle.curves(data)
        assert curves.shape == (len(data), len(grid))
        validate_survival_curves(curves)


def test_survival_curve_validation_rejects_increase():
    try:
        validate_survival_curves(np.array([[1.0, 0.8, 0.9]]))
    except ValueError as error:
        assert "non-increasing" in str(error)
    else:
        raise AssertionError("Increasing curve was accepted")


def test_median_survival_and_horizon_probabilities():
    grid = np.array([1.0, 2.0, 3.0])
    curves = np.array([[0.9, 0.5, 0.2], [0.9, 0.8, 0.7]])
    medians = median_survival(curves, grid)
    assert medians[0] == 2
    assert np.isnan(medians[1])
    values = horizon_predictions(curves, grid, (2.0, 4.0))
    assert values[2.0][0] == 0.5
    assert np.isnan(values[4.0]).all()


def test_bearing_macro_metrics_preserve_bearing_unit():
    data = frame().reset_index(drop=True)
    grid = np.linspace(1.1, 6.9, 6)
    bundle = fit_km(data, grid, 42)
    metrics, _ = bearing_metrics(data, data, bundle.curves(data), grid)
    assert metrics.bearing_id.nunique() == 3
    assert len(metrics) == 3


def test_candidate_ranking_tie_breaks_deterministically():
    records = pd.DataFrame(
        {
            "fold": [1, 1],
            "scenario": ["full_event"] * 2,
            "model": ["cox"] * 2,
            "validation_macro_ibs": [0.2, 0.2],
            "validation_main_horizon_brier": [0.1, 0.1],
            "validation_macro_ipcw_c_index": [0.6, 0.6],
            "selected_feature_count": [10, 5],
            "complexity_order": [1, 1],
            "candidate_id": [1, 2],
        }
    )
    ranked = rank_candidates(records)
    assert ranked.loc[ranked.selected, "candidate_id"].item() == 2
