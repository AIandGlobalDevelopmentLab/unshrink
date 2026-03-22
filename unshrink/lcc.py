from __future__ import annotations
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression

from .base import BaseDebiaser
from .reports import DebiasingWarning

FloatArray = NDArray[np.float64]


class LccDebiaser(BaseDebiaser):
    """
    Linear Calibration Correction (LCC)
    """

    method_name = "lcc"
    _IDENTIFIABLE_SLOPE = 1e-8
    _WEAK_SLOPE = 0.2

    def __init__(self):
        pass

    def fit(self, cal_predictions: ArrayLike, cal_targets: ArrayLike) -> "LccDebiaser":
        cal_predictions_array, cal_targets_array = self._validate_calibration_inputs(
            cal_predictions,
            cal_targets,
        )
        target_std = float(np.std(cal_targets_array))
        prediction_std = float(np.std(cal_predictions_array))
        if np.isclose(target_std, 0.0):
            raise ValueError("cal_targets must vary to fit LccDebiaser.")
        if np.isclose(prediction_std, 0.0):
            raise ValueError("cal_predictions must vary to fit LccDebiaser.")

        model = LinearRegression()
        model.fit(cal_targets_array.reshape(-1, 1), cal_predictions_array)

        self.intercept_ = float(model.intercept_)
        self.slope_ = float(model.coef_[0])

        if not np.isfinite(self.slope_) or self.slope_ <= self._IDENTIFIABLE_SLOPE:
            raise ValueError("LCC calibration slope must be positive and identifiable.")

        warning_flags: list[str] = []
        if self.slope_ < self._WEAK_SLOPE:
            warning_flags.append("weak_slope")
            warnings.warn(
                "LCC calibration slope is small; inverse calibration may be numerically unstable.",
                DebiasingWarning,
                stacklevel=2,
            )

        fitted_predictions = model.predict(cal_targets_array.reshape(-1, 1))
        residuals = cal_predictions_array - fitted_predictions
        self.residual_std_ = float(np.std(residuals))
        self.r_squared_ = float(model.score(cal_targets_array.reshape(-1, 1), cal_predictions_array))
        self._set_diagnostics(
            cal_predictions=cal_predictions_array,
            residual_std=self.residual_std_,
            warning_flags=warning_flags,
            details={
                "intercept": self.intercept_,
                "slope": self.slope_,
                "prediction_std": prediction_std,
                "target_std": target_std,
                "r_squared": self.r_squared_,
            },
        )
        return self

    def debiased_predictions(self, predictions: ArrayLike) -> FloatArray:
        self._require_is_fit("slope_", "intercept_", "diagnostics_", method_name="debiased_predictions")
        prediction_array = self._validate_prediction_inputs(predictions)
        self._warn_if_outside_support(prediction_array)
        return (prediction_array - self.intercept_) / self.slope_


LCCDebiaser = LccDebiaser
