import numpy as np
import pytest

from unshrink import (
    DebiaserComparisonReport,
    LccDebiaser,
    TweedieDebiaser,
    compare_debiasers,
    evaluate_debiaser,
)


def test_evaluate_debiaser_tweedie(nonlinear_shrinkage_data):
    cal_preds, cal_targets, preds, targets = nonlinear_shrinkage_data
    result = evaluate_debiaser(
        TweedieDebiaser(),
        cal_preds,
        cal_targets,
        preds,
        targets,
    )

    assert {"true_mean", "naive_mean", "corrected_mean", "bias_after", "bias_before"} <= result.keys()
    assert np.isclose(result["true_mean"], np.mean(targets))
    assert np.isclose(result["naive_mean"], np.mean(preds))


def test_evaluate_debiaser_lcc(linear_shrinkage_data):
    cal_preds, cal_targets, preds, targets = linear_shrinkage_data
    result = evaluate_debiaser(
        LccDebiaser(),
        cal_preds,
        cal_targets,
        preds,
        targets,
    )

    assert isinstance(result["bias_after"], float)
    assert result["bias_after"] < result["bias_before"]


def test_evaluate_debiaser_rejects_non_finite_targets(linear_shrinkage_data):
    cal_preds, cal_targets, preds, targets = linear_shrinkage_data
    bad_targets = targets.copy()
    bad_targets[0] = np.nan
    with pytest.raises(ValueError, match="targets must contain only finite"):
        evaluate_debiaser(LccDebiaser(), cal_preds, cal_targets, preds, bad_targets)


def test_compare_debiasers_prefers_lcc_for_linear_regime(linear_shrinkage_data):
    cal_preds, cal_targets, _, _ = linear_shrinkage_data
    report = compare_debiasers(cal_preds, cal_targets, n_splits=4, n_contrast_draws=40, random_state=3)

    assert isinstance(report, DebiaserComparisonReport)
    assert set(report.metrics) == {"naive", "lcc", "tweedie"}
    assert report.recommended_method == "lcc"
    assert report.metrics["lcc"].pseudo_ate_rmse < report.metrics["naive"].pseudo_ate_rmse


def test_compare_debiasers_prefers_tweedie_for_nonlinear_regime(nonlinear_shrinkage_data):
    cal_preds, cal_targets, _, _ = nonlinear_shrinkage_data
    report = compare_debiasers(cal_preds, cal_targets, n_splits=4, n_contrast_draws=50, random_state=4)

    assert report.recommended_method == "tweedie"
    assert abs(report.metrics["tweedie"].calibration_slope - 1.0) < abs(
        report.metrics["lcc"].calibration_slope - 1.0
    )


def test_compare_debiasers_rejects_bad_arguments(linear_shrinkage_data):
    cal_preds, cal_targets, _, _ = linear_shrinkage_data
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        compare_debiasers(cal_preds, cal_targets, n_splits=1)
    with pytest.raises(ValueError, match="n_contrast_draws must be at least 1"):
        compare_debiasers(cal_preds, cal_targets, n_contrast_draws=0)
