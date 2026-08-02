"""Deterministic validation-only temporal candidate ranking."""

import pandas as pd


def rank_candidates(table: pd.DataFrame) -> pd.DataFrame:
    result = table.sort_values(
        [
            "validation_macro_mae",
            "validation_macro_rmse",
            "trainable_parameters",
            "sequence_length",
            "candidate_order",
        ],
        kind="stable",
    ).reset_index(drop=True)
    result["selection_rank"] = range(1, len(result) + 1)
    result["selected"] = result.selection_rank.eq(1)
    return result
