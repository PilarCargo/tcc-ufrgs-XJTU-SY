"""Training-only model fitting and validation-only deterministic selection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.config.phase4 import load_phase4_config
from xjtu_sy_tcc.rul.models import ExperimentSpec, fit_model
from xjtu_sy_tcc.rul.selection import rank_candidates, validation_candidate_record
from xjtu_sy_tcc.rul.weights import bearing_balanced_weights


def _training() -> pd.DataFrame:
    rows = []
    for bearing, offset in (("A", 0.0), ("B", 1.0)):
        for index in range(12):
            rows.append(
                {
                    "bearing_id": bearing,
                    "sequence_index": index,
                    "horizontal_energy": float((index + 1) ** 2 + offset),
                    "rul_minutes": float(11 - index),
                }
            )
    return pd.DataFrame(rows)


def test_ridge_preprocessing_fits_training_bearings_only_and_is_deterministic() -> None:
    training = _training()
    config = load_phase4_config(Path("configs/phase4.yaml"))
    spec = ExperimentSpec("selected_features_ridge", "ridge", ("horizontal_energy",), 1)
    weights = bearing_balanced_weights(training)
    first = fit_model(spec, {"alpha": 1.0}, training, weights, config)
    second = fit_model(spec, {"alpha": 1.0}, training, weights, config)
    assert first.transformer is not None
    assert first.transformer.fitted_bearings == ("A", "B")
    assert np.allclose(first.predict(training), second.predict(training))
    assert first.transformer.log1p_columns == ("horizontal_energy",)


def test_candidate_selection_rejects_test_rows() -> None:
    training = _training()
    config = load_phase4_config(Path("configs/phase4.yaml"))
    spec = ExperimentSpec("selected_features_ridge", "ridge", ("horizontal_energy",), 1)
    bundle = fit_model(spec, {"alpha": 1.0}, training, bearing_balanced_weights(training), config)
    predictions = training.assign(
        experiment=spec.experiment,
        model="ridge",
        fold=1,
        condition_id=1,
        acquisition_number=np.arange(len(training)),
        elapsed_minutes=0.0,
        rotation_rpm=2100.0,
        radial_load_kn=12.0,
        subset="test",
        prediction_raw_minutes=1.0,
        prediction_non_negative_minutes=1.0,
    )
    with pytest.raises(ValueError, match="validation rows only"):
        validation_candidate_record(bundle, predictions, 1, 1)


def test_candidate_ranking_uses_mae_rmse_complexity_and_order() -> None:
    frame = pd.DataFrame(
        {
            "fold": [1, 1, 1],
            "experiment": ["e"] * 3,
            "candidate_id": [3, 2, 1],
            "validation_macro_mae": [2.0, 1.0, 1.0],
            "validation_macro_rmse": [2.0, 2.0, 2.0],
            "complexity_order": [1, 2, 1],
        }
    )
    ranked = rank_candidates(frame)
    assert ranked.iloc[0]["candidate_id"] == 1
    assert ranked["selected"].sum() == 1
