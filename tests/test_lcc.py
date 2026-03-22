import numpy as np
import pandas as pd
import pytest

from unshrink import LCCDebiaser, LccDebiaser
from unshrink.reports import DebiasingWarning


def test_debiased_mean_before_fit():
    lcc = LccDebiaser()
    with pytest.raises(RuntimeError):
        lcc.debiased_mean(np.array([1.0, 2.0, 3.0]))


def test_debiased_predictions_before_fit():
    lcc = LccDebiaser()
    with pytest.raises(RuntimeError):
        lcc.debiased_predictions(np.array([1.0, 2.0, 3.0]))


def test_lcc_alias_points_to_same_class():
    assert LCCDebiaser is LccDebiaser


def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    lcc = LccDebiaser().fit(preds, targets)
    corrected = lcc.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())


def test_linear_bias_reduction(linear_shrinkage_data):
    cal_preds, cal_targets, preds, targets = linear_shrinkage_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = lcc.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)


def test_params_roundtrip():
    lcc = LccDebiaser()
    params = lcc.get_params()
    lcc.set_params(**params)
    assert lcc.get_params() == params


def test_debiased_predictions_mean_consistency(linear_shrinkage_data):
    cal_preds, cal_targets, preds, _ = linear_shrinkage_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)

    dp = lcc.debiased_predictions(preds)
    dm = lcc.debiased_mean(preds)

    assert np.isclose(dp.mean(), dm)


def test_works_with_pandas_inputs(linear_shrinkage_data):
    cal_preds, cal_targets, preds, _ = linear_shrinkage_data
    lcc = LccDebiaser().fit(pd.Series(cal_preds), pd.Series(cal_targets))
    corrected = lcc.debiased_predictions(pd.Series(preds))
    assert corrected.shape == preds.shape


def test_diagnostics_are_populated(linear_shrinkage_data):
    cal_preds, cal_targets, _, _ = linear_shrinkage_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)
    diagnostics = lcc.diagnostics_

    assert diagnostics.method == "lcc"
    assert diagnostics.n_calibration == cal_preds.size
    assert "slope" in diagnostics.details
    assert diagnostics.details["slope"] > 0


def test_rejects_constant_targets():
    cal_targets = np.ones(50)
    cal_preds = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="cal_targets must vary"):
        LccDebiaser().fit(cal_preds, cal_targets)


def test_rejects_constant_predictions():
    cal_targets = np.linspace(0, 1, 50)
    cal_preds = np.ones(50)
    with pytest.raises(ValueError, match="cal_predictions must vary"):
        LccDebiaser().fit(cal_preds, cal_targets)


def test_rejects_non_positive_slope():
    cal_targets = np.linspace(0, 1, 100)
    cal_preds = 1.0 - cal_targets
    with pytest.raises(ValueError, match="slope must be positive"):
        LccDebiaser().fit(cal_preds, cal_targets)


def test_warns_on_small_positive_slope():
    cal_targets = np.linspace(-1, 1, 300)
    cal_preds = 0.02 * cal_targets + np.random.default_rng(1).normal(0, 1e-3, 300)
    with pytest.warns(DebiasingWarning, match="slope is small"):
        debiaser = LccDebiaser().fit(cal_preds, cal_targets)
    assert "weak_slope" in debiaser.diagnostics_.warning_flags


def test_warns_when_predicting_outside_support(linear_shrinkage_data):
    cal_preds, cal_targets, _, _ = linear_shrinkage_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)
    outside = np.array([cal_preds.min() - 1.0, cal_preds.max() + 1.0])
    with pytest.warns(DebiasingWarning, match="calibration support"):
        lcc.debiased_predictions(outside)
