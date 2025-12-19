import numpy as np
from unshrink import TweedieDebiaser, LccDebiaser
from unshrink.utils import evaluate_debiaser


def test_evaluate_debiaser_tweedie():
    """Test evaluate_debiaser returns expected keys and reduces bias."""
    rng = np.random.default_rng(42)
    n = 500
    noise = 0.1

    cal_preds = rng.random(n)
    cal_targets = cal_preds + rng.normal(0, noise, n)
    preds = rng.random(n)
    targets = preds + rng.normal(0, noise, n)

    result = evaluate_debiaser(
        TweedieDebiaser(),
        cal_preds, cal_targets,
        preds, targets
    )

    assert "true_mean" in result
    assert "naive_mean" in result
    assert "corrected_mean" in result
    assert "bias_before" in result
    assert "bias_after" in result

    assert np.isclose(result["true_mean"], np.mean(targets))
    assert np.isclose(result["naive_mean"], np.mean(preds))


def test_evaluate_debiaser_lcc():
    """Test evaluate_debiaser works with LccDebiaser."""
    rng = np.random.default_rng(42)
    n = 500
    noise = 0.1

    cal_preds = rng.random(n)
    cal_targets = cal_preds + rng.normal(0, noise, n)
    preds = rng.random(n)
    targets = preds + rng.normal(0, noise, n)

    result = evaluate_debiaser(
        LccDebiaser(),
        cal_preds, cal_targets,
        preds, targets
    )

    assert "true_mean" in result
    assert "corrected_mean" in result
    assert isinstance(result["bias_after"], float)
