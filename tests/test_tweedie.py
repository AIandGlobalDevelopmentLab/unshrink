import numpy as np
import pandas as pd
import pytest

from unshrink import TweedieDebiaser
from unshrink.reports import DebiasingWarning


def test_debiased_mean_before_fit():
    tweedie = TweedieDebiaser()
    with pytest.raises(RuntimeError):
        tweedie.debiased_mean(np.array([1.0, 2.0, 3.0]))


def test_debiased_predictions_before_fit():
    tweedie = TweedieDebiaser()
    with pytest.raises(RuntimeError):
        tweedie.debiased_predictions(np.array([1.0, 2.0, 3.0]))


def test_no_noise_identity_behavior(no_noise_data):
    preds, targets = no_noise_data
    tweedie = TweedieDebiaser().fit(preds, targets)
    corrected = tweedie.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())
    assert "zero_sigma_identity" in tweedie.diagnostics_.warning_flags


def test_bias_reduction(nonlinear_shrinkage_data):
    cal_preds, cal_targets, preds, targets = nonlinear_shrinkage_data
    tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = tweedie.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)


def test_params_roundtrip():
    tweedie = TweedieDebiaser(delta=1e-4)
    params = tweedie.get_params()
    tweedie.set_params(**params)
    assert tweedie.get_params() == params


def test_sigma_from_separate_calibration():
    rng = np.random.default_rng(1)
    n_cal = 1000
    n_sigma = 800
    noise_cal = 0.05
    noise_sigma = 0.5

    cal_targets = rng.normal(0, 1, n_cal)
    cal_preds = cal_targets + rng.normal(0, noise_cal, n_cal)

    sigma_targets = rng.normal(0, 1, n_sigma)
    sigma_preds = sigma_targets + rng.normal(0, noise_sigma, n_sigma)

    tweedie = TweedieDebiaser().fit(
        cal_preds,
        cal_targets,
        cal_predictions_sigma=sigma_preds,
        cal_targets_sigma=sigma_targets,
    )

    expected_sigma = np.std(sigma_preds - sigma_targets)
    assert pytest.approx(expected_sigma, rel=1e-2) == tweedie.sigma_
    assert tweedie.diagnostics_.details["sigma_source"] == "separate"


def test_sigma_args_must_both_be_provided(noisy_data):
    cal_preds, cal_targets, _, _ = noisy_data
    with pytest.raises(ValueError):
        TweedieDebiaser().fit(cal_preds, cal_targets, cal_predictions_sigma=np.ones_like(cal_preds))
    with pytest.raises(ValueError):
        TweedieDebiaser().fit(cal_preds, cal_targets, cal_targets_sigma=np.ones_like(cal_targets))


def test_debiased_predictions_mean_consistency(nonlinear_shrinkage_data):
    cal_preds, cal_targets, preds, _ = nonlinear_shrinkage_data
    tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)

    dp = tweedie.debiased_predictions(preds)
    dm = tweedie.debiased_mean(preds)

    assert np.isclose(dp.mean(), dm)


def test_works_with_pandas_inputs(nonlinear_shrinkage_data):
    cal_preds, cal_targets, preds, _ = nonlinear_shrinkage_data
    tweedie = TweedieDebiaser().fit(pd.Series(cal_preds), pd.Series(cal_targets))
    corrected = tweedie.debiased_predictions(pd.Series(preds))
    assert corrected.shape == preds.shape


def test_invalid_delta_raises(nonlinear_shrinkage_data):
    cal_preds, cal_targets, _, _ = nonlinear_shrinkage_data
    with pytest.raises(ValueError, match="delta must be positive"):
        TweedieDebiaser(delta=0.0).fit(cal_preds, cal_targets)


def test_rejects_constant_predictions_with_noise():
    cal_targets = np.linspace(0, 1, 100)
    cal_preds = np.ones(100)
    with pytest.raises(ValueError, match="at least two unique values"):
        TweedieDebiaser().fit(cal_preds, cal_targets)


def test_warns_for_small_calibration_sample():
    rng = np.random.default_rng(2)
    cal_targets = rng.normal(0, 1, 80)
    cal_preds = cal_targets + rng.normal(0, 0.15, 80)
    with pytest.warns(DebiasingWarning, match="small calibration sample"):
        tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)
    assert "small_calibration_sample" in tweedie.diagnostics_.warning_flags


def test_warns_when_predicting_outside_support(nonlinear_shrinkage_data):
    cal_preds, cal_targets, _, _ = nonlinear_shrinkage_data
    tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)
    outside = np.array([cal_preds.min() - 1.0, cal_preds.max() + 1.0])
    with pytest.warns(DebiasingWarning, match="calibration support"):
        tweedie.debiased_predictions(outside)
