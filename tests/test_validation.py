import numpy as np
import pytest

from unshrink import LccDebiaser, TweedieDebiaser


@pytest.mark.parametrize("Debiaser", [LccDebiaser, TweedieDebiaser])
def test_fit_rejects_non_1d_inputs(Debiaser):
    cal_preds = np.ones((5, 2))
    cal_targets = np.ones((5, 2))
    with pytest.raises(ValueError, match="1D array"):
        Debiaser().fit(cal_preds, cal_targets)


@pytest.mark.parametrize("Debiaser", [LccDebiaser, TweedieDebiaser])
def test_fit_rejects_mismatched_lengths(Debiaser):
    cal_preds = np.arange(10, dtype=float)
    cal_targets = np.arange(9, dtype=float)
    with pytest.raises(ValueError, match="same shape"):
        Debiaser().fit(cal_preds, cal_targets)


@pytest.mark.parametrize("Debiaser", [LccDebiaser, TweedieDebiaser])
def test_fit_rejects_non_finite_values(Debiaser):
    cal_preds = np.arange(10, dtype=float)
    cal_targets = np.arange(10, dtype=float)
    cal_targets[0] = np.inf
    with pytest.raises(ValueError, match="finite numeric values"):
        Debiaser().fit(cal_preds, cal_targets)


@pytest.mark.parametrize("Debiaser", [LccDebiaser, TweedieDebiaser])
def test_prediction_methods_reject_empty_inputs(Debiaser, no_noise_data):
    preds, targets = no_noise_data
    debiaser = Debiaser().fit(preds, targets)
    with pytest.raises(ValueError, match="must not be empty"):
        debiaser.debiased_predictions(np.array([]))
