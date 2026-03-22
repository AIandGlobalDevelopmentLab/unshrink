from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import KFold

from .base import BaseDebiaser
from .lcc import LccDebiaser
from .reports import DebiaserComparisonReport, DebiaserMetrics
from .tweedie import TweedieDebiaser

FloatArray = NDArray[np.float64]


def _as_1d_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _validate_pair(predictions: ArrayLike, targets: ArrayLike) -> tuple[FloatArray, FloatArray]:
    prediction_array = _as_1d_float_array(predictions, name="cal_predictions")
    target_array = _as_1d_float_array(targets, name="cal_targets")
    if prediction_array.shape != target_array.shape:
        raise ValueError("cal_predictions and cal_targets must have the same shape.")
    return prediction_array, target_array


def _clone_debiaser(debiaser: BaseDebiaser) -> BaseDebiaser:
    return debiaser.__class__(**debiaser.get_params())


def _calibration_slope(truth: FloatArray, estimate: FloatArray) -> float:
    truth_centered = truth - np.mean(truth)
    denominator = float(np.dot(truth_centered, truth_centered))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    estimate_centered = estimate - np.mean(estimate)
    return float(np.dot(truth_centered, estimate_centered) / denominator)


def _summary_metrics(
    method: str,
    mean_truth: FloatArray,
    mean_estimate: FloatArray,
    contrast_truth: FloatArray,
    contrast_estimate: FloatArray,
    warning_flags: set[str],
) -> DebiaserMetrics:
    mean_errors = mean_estimate - mean_truth
    contrast_errors = contrast_estimate - contrast_truth
    return DebiaserMetrics(
        method=method,
        mean_bias=float(np.mean(mean_errors)),
        mean_absolute_error=float(np.mean(np.abs(mean_errors))),
        rmse=float(np.sqrt(np.mean(np.square(mean_errors)))),
        calibration_slope=_calibration_slope(contrast_truth, contrast_estimate),
        pseudo_ate_bias=float(np.mean(contrast_errors)),
        pseudo_ate_rmse=float(np.sqrt(np.mean(np.square(contrast_errors)))),
        warning_flags=tuple(sorted(warning_flags)),
    )


def _contrast_masks(
    predictions: FloatArray,
    rng: Generator,
    *,
    n_contrast_draws: int,
) -> list[tuple[FloatArray, FloatArray]]:
    masks: list[tuple[FloatArray, FloatArray]] = []
    for _ in range(n_contrast_draws):
        lower_quantile = float(rng.uniform(0.05, 0.35))
        upper_quantile = float(rng.uniform(0.65, 0.95))
        lower_cutoff, upper_cutoff = np.quantile(predictions, [lower_quantile, upper_quantile])
        control_mask = predictions <= lower_cutoff
        treated_mask = predictions >= upper_cutoff
        if control_mask.sum() == 0 or treated_mask.sum() == 0:
            continue
        masks.append((treated_mask, control_mask))
    return masks


def _contrast_metrics(
    predictions: FloatArray,
    targets: FloatArray,
    debiasers: dict[str, BaseDebiaser],
    rng: Generator,
    *,
    n_contrast_draws: int,
) -> tuple[dict[str, list[float]], list[float]]:
    truth_values: list[float] = []
    estimated_values: dict[str, list[float]] = defaultdict(list)
    for treated_mask, control_mask in _contrast_masks(
        predictions,
        rng,
        n_contrast_draws=n_contrast_draws,
    ):
        treated_predictions = predictions[treated_mask]
        control_predictions = predictions[control_mask]
        truth_values.append(float(np.mean(targets[treated_mask]) - np.mean(targets[control_mask])))
        estimated_values["naive"].append(
            float(np.mean(treated_predictions) - np.mean(control_predictions))
        )
        for name, debiaser in debiasers.items():
            estimated_values[name].append(
                float(debiaser.debiased_ate(treated_predictions, control_predictions))
            )
    return estimated_values, truth_values


def _method_score(metrics: DebiaserMetrics, *, target_scale: float) -> float:
    slope_penalty = 10.0 if not np.isfinite(metrics.calibration_slope) else abs(metrics.calibration_slope - 1.0)
    return (
        slope_penalty
        + metrics.pseudo_ate_rmse / target_scale
        + 0.5 * metrics.mean_absolute_error / target_scale
        + 0.05 * len(metrics.warning_flags)
    )


def evaluate_debiaser(
    debiaser: BaseDebiaser,
    cal_predictions: ArrayLike,
    cal_targets: ArrayLike,
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    """
    Fit debiaser and return naive vs corrected estimates.
    """
    cal_prediction_array, cal_target_array = _validate_pair(cal_predictions, cal_targets)
    prediction_array = _as_1d_float_array(predictions, name="predictions")
    target_array = _as_1d_float_array(targets, name="targets")
    if prediction_array.shape != target_array.shape:
        raise ValueError("predictions and targets must have the same shape.")

    debiaser.fit(cal_prediction_array, cal_target_array)
    corrected_mean = float(debiaser.debiased_mean(prediction_array))
    true_mean = float(np.mean(target_array))
    naive_mean = float(np.mean(prediction_array))

    return {
        "true_mean": true_mean,
        "naive_mean": naive_mean,
        "corrected_mean": corrected_mean,
        "bias_after": float(abs(corrected_mean - true_mean)),
        "bias_before": float(abs(naive_mean - true_mean)),
    }


def compare_debiasers(
    cal_predictions: ArrayLike,
    cal_targets: ArrayLike,
    *,
    lcc_debiaser: BaseDebiaser | None = None,
    tweedie_debiaser: BaseDebiaser | None = None,
    n_splits: int = 5,
    n_contrast_draws: int = 100,
    random_state: int = 0,
) -> DebiaserComparisonReport:
    """
    Compare naive, LCC, and Tweedie corrections on held-out calibration folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_contrast_draws < 1:
        raise ValueError("n_contrast_draws must be at least 1.")

    prediction_array, target_array = _validate_pair(cal_predictions, cal_targets)
    if prediction_array.size <= n_splits:
        raise ValueError("n_splits must be smaller than the number of calibration observations.")

    lcc_template = lcc_debiaser if lcc_debiaser is not None else LccDebiaser()
    tweedie_template = tweedie_debiaser if tweedie_debiaser is not None else TweedieDebiaser()
    master_rng = default_rng(random_state)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mean_truth: list[float] = []
    mean_estimates: dict[str, list[float]] = defaultdict(list)
    contrast_truth: list[float] = []
    contrast_estimates: dict[str, list[float]] = defaultdict(list)
    warning_flags: dict[str, set[str]] = {"naive": set(), "lcc": set(), "tweedie": set()}

    for train_index, test_index in splitter.split(prediction_array):
        train_predictions = prediction_array[train_index]
        train_targets = target_array[train_index]
        test_predictions = prediction_array[test_index]
        test_targets = target_array[test_index]

        lcc = _clone_debiaser(lcc_template)
        tweedie = _clone_debiaser(tweedie_template)
        lcc.fit(train_predictions, train_targets)
        tweedie.fit(train_predictions, train_targets)
        warning_flags["lcc"].update(lcc.diagnostics_.warning_flags)
        warning_flags["tweedie"].update(tweedie.diagnostics_.warning_flags)

        truth_mean = float(np.mean(test_targets))
        mean_truth.append(truth_mean)
        mean_estimates["naive"].append(float(np.mean(test_predictions)))
        mean_estimates["lcc"].append(float(lcc.debiased_mean(test_predictions)))
        mean_estimates["tweedie"].append(float(tweedie.debiased_mean(test_predictions)))

        contrast_rng = default_rng(int(master_rng.integers(0, 2**32 - 1)))
        fold_contrast_estimates, fold_truth = _contrast_metrics(
            test_predictions,
            test_targets,
            {"lcc": lcc, "tweedie": tweedie},
            contrast_rng,
            n_contrast_draws=n_contrast_draws,
        )
        contrast_truth.extend(fold_truth)
        for name, values in fold_contrast_estimates.items():
            contrast_estimates[name].extend(values)

    mean_truth_array = np.asarray(mean_truth, dtype=float)
    contrast_truth_array = np.asarray(contrast_truth, dtype=float)
    metrics = {
        name: _summary_metrics(
            name,
            mean_truth_array,
            np.asarray(mean_estimates[name], dtype=float),
            contrast_truth_array,
            np.asarray(contrast_estimates[name], dtype=float),
            warning_flags[name],
        )
        for name in ("naive", "lcc", "tweedie")
    }

    target_scale = max(float(np.std(target_array)), np.finfo(float).eps ** 0.5)
    candidate_scores = {
        name: _method_score(metrics[name], target_scale=target_scale)
        for name in ("lcc", "tweedie")
    }
    recommended_method = min(candidate_scores, key=candidate_scores.get)
    recommended_metrics = metrics[recommended_method]
    runner_up = "tweedie" if recommended_method == "lcc" else "lcc"
    runner_up_metrics = metrics[runner_up]
    rationale = (
        f"Recommended {recommended_method} because its pseudo-ATE RMSE "
        f"({recommended_metrics.pseudo_ate_rmse:.4f}) and calibration slope "
        f"({recommended_metrics.calibration_slope:.3f}) beat {runner_up}'s "
        f"({runner_up_metrics.pseudo_ate_rmse:.4f}, {runner_up_metrics.calibration_slope:.3f})."
    )
    aggregate_flags = tuple(sorted(set().union(*(metric.warning_flags for metric in metrics.values()))))
    return DebiaserComparisonReport(
        metrics=metrics,
        recommended_method=recommended_method,
        rationale=rationale,
        n_splits=n_splits,
        n_contrast_draws=n_contrast_draws,
        random_state=random_state,
        warning_flags=aggregate_flags,
    )
