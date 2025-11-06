import numpy as np
import pandas as pd
from debiasor import LccDebiaser

def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    lcc = LccDebiaser().fit(preds, targets)
    corrected = lcc.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())

def test_bias_reduction(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
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


def test_debiased_predictions_mean_consistency(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)

    dp = lcc.debiased_predictions(preds)
    dm = lcc.debiased_mean(preds)

    # mean of the per-element debiased predictions should equal debiased_mean
    assert np.isclose(dp.mean(), dm)


def test_works_with_pandas_inputs_no_noise(no_noise_data):
    """Ensure methods accept pandas.Series when there is no noise."""
    preds, targets = no_noise_data
    preds_s = pd.Series(preds)
    targets_s = pd.Series(targets)

    lcc = LccDebiaser().fit(preds_s, targets_s)
    corrected = lcc.debiased_mean(preds_s)
    assert np.isclose(corrected, preds_s.mean())


def test_works_with_pandas_inputs_noisy(noisy_data):
    """Ensure methods accept pandas.Series for calibration and prediction sets."""
    cal_preds, cal_targets, preds, targets = noisy_data
    cal_preds_s = pd.Series(cal_preds)
    cal_targets_s = pd.Series(cal_targets)
    preds_s = pd.Series(preds)

    lcc = LccDebiaser().fit(cal_preds_s, cal_targets_s)

    dp = lcc.debiased_predictions(preds_s)
    dm = lcc.debiased_mean(preds_s)

    assert np.isclose(dp.mean(), dm)
