from __future__ import annotations
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gaussian_kde

from .base import BaseDebiaser
from .reports import DebiasingWarning

FloatArray = NDArray[np.float64]


class TweedieDebiaser(BaseDebiaser):
    """
    Tweedie's correction
    """

    method_name = "tweedie"

    def __init__(self, delta: float | None = None):
        self.delta = delta

    def fit(
        self,
        cal_predictions: ArrayLike,
        cal_targets: ArrayLike,
        cal_predictions_sigma: ArrayLike | None = None,
        cal_targets_sigma: ArrayLike | None = None,
    ) -> "TweedieDebiaser":
        cal_predictions_array, cal_targets_array = self._validate_calibration_inputs(
            cal_predictions,
            cal_targets,
        )

        if cal_predictions_sigma is not None and cal_targets_sigma is not None:
            sigma_prediction_array, sigma_target_array = self._validate_calibration_inputs(
                cal_predictions_sigma,
                cal_targets_sigma,
            )
            sigma_source = "separate"
        elif cal_predictions_sigma is None and cal_targets_sigma is None:
            sigma_prediction_array = cal_predictions_array
            sigma_target_array = cal_targets_array
            sigma_source = "calibration"
        else:
            raise ValueError("Either both or neither of cal_predictions_sigma and cal_targets_sigma must be provided.")

        if self.delta is not None and self.delta <= 0:
            raise ValueError("delta must be positive when provided.")

        sigma_residuals = sigma_prediction_array - sigma_target_array
        self.sigma_ = float(np.std(sigma_residuals))
        if not np.isfinite(self.sigma_) or self.sigma_ < 0.0:
            raise ValueError("Estimated sigma must be finite and non-negative.")

        warning_flags: list[str] = []
        prediction_std = float(np.std(cal_predictions_array))
        unique_predictions = int(np.unique(cal_predictions_array).size)
        if self.sigma_ == 0.0:
            self.kde_ = None
            self.delta_ = 0.0
            warning_flags.append("zero_sigma_identity")
        else:
            if unique_predictions < 2 or np.isclose(prediction_std, 0.0):
                raise ValueError(
                    "cal_predictions must contain at least two unique values to fit TweedieDebiaser."
                )
            try:
                self.kde_ = gaussian_kde(cal_predictions_array)
            except (LinAlgError, ValueError) as exc:
                raise ValueError(
                    "Unable to fit the Tweedie KDE. Check for singular or near-constant calibration predictions."
                ) from exc
            self.delta_ = self._resolve_delta(cal_predictions_array)

        if cal_predictions_array.size < 100:
            warning_flags.append("small_calibration_sample")
            warnings.warn(
                "TweedieDebiaser is fitted on a small calibration sample; KDE-based corrections may be noisy.",
                DebiasingWarning,
                stacklevel=2,
            )

        self._set_diagnostics(
            cal_predictions=cal_predictions_array,
            residual_std=self.sigma_,
            warning_flags=warning_flags,
            details={
                "sigma": self.sigma_,
                "delta": self.delta_,
                "prediction_std": prediction_std,
                "unique_predictions": unique_predictions,
                "sigma_source": sigma_source,
            },
        )
        return self

    def _resolve_delta(self, cal_predictions: FloatArray) -> float:
        if self.delta is not None:
            return float(self.delta)
        scale = max(float(np.std(cal_predictions)), np.finfo(float).eps ** 0.5)
        return max(scale * 1e-3, 1e-8)

    def _score(self, predictions: FloatArray) -> FloatArray:
        if self.kde_ is None or self.sigma_ == 0.0:
            return np.zeros_like(predictions)
        log_p_plus = self.kde_.logpdf(predictions + self.delta_)
        log_p_minus = self.kde_.logpdf(predictions - self.delta_)
        return (log_p_plus - log_p_minus) / (2.0 * self.delta_)

    def debiased_predictions(self, predictions: ArrayLike) -> FloatArray:
        self._require_is_fit("sigma_", "diagnostics_", method_name="debiased_predictions")
        prediction_array = self._validate_prediction_inputs(predictions)
        self._warn_if_outside_support(prediction_array)
        scores = self._score(prediction_array)
        return prediction_array - self.sigma_**2 * scores
