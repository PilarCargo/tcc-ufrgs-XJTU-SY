import pandas as pd

from xjtu_sy_tcc.temporal.selection import rank_candidates


def test_candidate_ties_use_parameter_count_then_sequence_length():
    table = pd.DataFrame(
        [
            {
                "validation_macro_mae": 1.0,
                "validation_macro_rmse": 2.0,
                "trainable_parameters": 20,
                "sequence_length": 5,
                "candidate_order": 1,
            },
            {
                "validation_macro_mae": 1.0,
                "validation_macro_rmse": 2.0,
                "trainable_parameters": 10,
                "sequence_length": 20,
                "candidate_order": 2,
            },
            {
                "validation_macro_mae": 1.0,
                "validation_macro_rmse": 2.0,
                "trainable_parameters": 10,
                "sequence_length": 10,
                "candidate_order": 3,
            },
        ]
    )
    ranked = rank_candidates(table)
    assert ranked.iloc[0].candidate_order == 3
    assert ranked.selected.sum() == 1
