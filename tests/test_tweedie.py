import numpy as np
from debiasor import TweedieCorrection

def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    tweedie = TweedieCorrection().fit(preds, targets)
    corrected = tweedie.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())

def test_bias_reduction(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    tweedie = TweedieCorrection().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = tweedie.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)

def test_params_roundtrip():
    tweedie = TweedieCorrection(delta=1e-4)
    params = tweedie.get_params()
    tweedie.set_params(**params)
    assert tweedie.get_params() == params
