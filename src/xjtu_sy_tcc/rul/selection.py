"""Validation-bearing-only deterministic candidate selection."""

from __future__ import annotations

import json

import pandas as pd

from xjtu_sy_tcc.rul.metrics import bearing_metrics
from xjtu_sy_tcc.rul.models import ModelBundle


def validation_candidate_record(
    bundle: ModelBundle,
    validation_predictions: pd.DataFrame,
    fold: int,
    candidate_id: int,
) -> dict[str, object]:
    """Calculate validation macro metrics without accepting any test rows."""
    if set(validation_predictions["subset"].unique()) != {"validation"}:
        raise ValueError("Candidate selection accepts validation rows only")
    metrics = bearing_metrics(validation_predictions)
    return {
        "fold": fold,
        "experiment": bundle.spec.experiment,
        "model": bundle.spec.model_family,
        "candidate_id": candidate_id,
        "parameters_json": json.dumps(bundle.parameters, sort_keys=True),
        "validation_macro_mae": float(metrics["mae"].mean()),
        "validation_macro_rmse": float(metrics["rmse"].mean()),
        "validation_macro_r2": float(metrics["r2"].mean()),
        "complexity_order": bundle.spec.complexity_order,
        "preprocessing_fit_seconds": bundle.preprocessing_fit_seconds,
        "training_seconds": bundle.training_seconds,
        "input_feature_count": len(bundle.spec.input_columns),
    }


def rank_candidates(records: pd.DataFrame) -> pd.DataFrame:
    """Rank by macro MAE, RMSE, complexity, then deterministic candidate order."""
    required = {
        "fold",
        "experiment",
        "validation_macro_mae",
        "validation_macro_rmse",
        "complexity_order",
        "candidate_id",
    }
    if required - set(records):
        raise ValueError("Candidate records are missing selection columns")
    ranked = records.sort_values(
        [
            "fold",
            "experiment",
            "validation_macro_mae",
            "validation_macro_rmse",
            "complexity_order",
            "candidate_id",
        ],
        kind="mergesort",
    ).copy()
    ranked["selection_rank"] = ranked.groupby(["fold", "experiment"]).cumcount() + 1
    ranked["selected"] = ranked["selection_rank"] == 1
    ranked["tie_breaking_rule"] = "macro_mae,macro_rmse,complexity,candidate_order"
    return ranked.reset_index(drop=True)
