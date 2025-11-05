import numpy as np
from debiasor import LinearCalibrationCorrection

def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    lcc = LinearCalibrationCorrection().fit(preds, targets)
    corrected = lcc.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())

def test_bias_reduction(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    lcc = LinearCalibrationCorrection().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = lcc.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)

def test_params_roundtrip():
    lcc = LinearCalibrationCorrection()
    params = lcc.get_params()
    lcc.set_params(**params)
    assert lcc.get_params() == params
