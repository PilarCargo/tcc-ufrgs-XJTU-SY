import numpy as np
import pandas as pd
import pytest
import torch

from xjtu_sy_tcc.temporal.model import RULLSTM
from xjtu_sy_tcc.temporal.preprocessing import fit_preprocessor
from xjtu_sy_tcc.temporal.sequences import SequenceSet


def test_lstm_shapes_context_and_finite_forward():
    x = torch.ones(4, 5, 3)
    model = RULLSTM(3, 8, 1, context_size=2)
    y = model(x, torch.ones(4, 2))
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert not model.lstm.bidirectional


def test_lstm_rejects_inactive_dropout():
    with pytest.raises(ValueError):
        RULLSTM(3, 8, 1, 0.2)


def test_training_only_weighted_target_and_scaling():
    metadata = pd.DataFrame({"bearing_id": ["A", "A", "B", "B"]})
    data = SequenceSet(
        np.ones((4, 2, 1), dtype="f"),
        np.ones((4, 3), dtype="f"),
        np.array([0, 2, 10, 12], dtype="f"),
        np.ones(4, dtype="f"),
        metadata,
    )
    prep = fit_preprocessor(data, ["rms"], ("rms",))
    assert prep.target_center == pytest.approx(6)
    assert prep.training_bearings == ("A", "B")
    assert np.isfinite(prep.transform_features(data.features)).all()


def test_incompatible_log_domain_rejected():
    data = SequenceSet(
        -np.ones((2, 1, 1), dtype="f"),
        np.ones((2, 3), dtype="f"),
        np.array([1, 2], dtype="f"),
        np.ones(2, dtype="f"),
        pd.DataFrame({"bearing_id": ["A", "B"]}),
    )
    with pytest.raises(ValueError):
        fit_preprocessor(data, ["rms"], ("rms",))
