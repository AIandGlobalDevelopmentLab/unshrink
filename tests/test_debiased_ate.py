import numpy as np
import pytest

from unshrink import TweedieDebiaser, LccDebiaser


@pytest.mark.parametrize("Debiaser", [TweedieDebiaser, LccDebiaser])
def test_debiased_ate_unweighted_consistency(Debiaser, no_noise_data):
    preds, targets = no_noise_data
    deb = Debiaser().fit(preds, targets)

    # split the prediction set into two groups (treated / control)
    treated = preds[:50]
    control = preds[50:100]

    ate = deb.debiased_ate(treated, control)

    # expected: mean of per-element debiased predictions difference
    expected = deb.debiased_predictions(treated).mean() - deb.debiased_predictions(control).mean()
    assert np.isclose(ate, expected)


@pytest.mark.parametrize("Debiaser", [TweedieDebiaser, LccDebiaser])
def test_debiased_ate_weighted_consistency(Debiaser, no_noise_data):
    preds, targets = no_noise_data
    deb = Debiaser().fit(preds, targets)

    treated = preds[:30]
    control = preds[30:80]

    # random positive weights
    rng = np.random.default_rng(1)
    w_t = rng.random(len(treated)) + 0.1
    w_c = rng.random(len(control)) + 0.1

    ate = deb.debiased_ate(treated, control, iptw_treated=w_t, iptw_control=w_c)

    deb_t = deb.debiased_predictions(treated)
    deb_c = deb.debiased_predictions(control)

    expected = np.average(deb_t, weights=w_t) - np.average(deb_c, weights=w_c)
    assert np.allclose(ate, expected)


@pytest.mark.parametrize("Debiaser", [TweedieDebiaser, LccDebiaser])
def test_debiased_ate_bad_weight_shapes(Debiaser, no_noise_data):
    preds, targets = no_noise_data
    deb = Debiaser().fit(preds, targets)

    treated = preds[:10]
    control = preds[10:20]

    # wrong-shaped weight for treated
    with pytest.raises(ValueError):
        deb.debiased_ate(treated, control, iptw_treated=np.ones(5))

    # wrong-shaped weight for control
    with pytest.raises(ValueError):
        deb.debiased_ate(treated, control, iptw_control=np.ones(5))
