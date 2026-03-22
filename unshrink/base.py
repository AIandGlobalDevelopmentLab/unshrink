from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reports import CalibrationDiagnostics, DebiasingWarning

FloatArray = NDArray[np.float64]


class BaseDebiaser(ABC):
    """
    Base class for debiasers with a sklearn-like fit/predict pattern.
    """

    method_name = "base"

    @abstractmethod
    def fit(self, cal_predictions: ArrayLike, cal_targets: ArrayLike) -> "BaseDebiaser":
        raise NotImplementedError

    @abstractmethod
    def debiased_predictions(self, predictions: ArrayLike) -> FloatArray:
        """Return per-observation debiased predictions."""
        raise NotImplementedError

    def debiased_mean(self, predictions: ArrayLike) -> float:
        self._require_is_fit("diagnostics_", method_name="debiased_mean")
        corrected = self.debiased_predictions(predictions)
        return float(np.mean(corrected))

    def _require_is_fit(self, *attributes: str, method_name: str) -> None:
        missing = [attribute for attribute in attributes if not hasattr(self, attribute)]
        if missing:
            raise RuntimeError(f"Call .fit() before .{method_name}().")

    def _as_1d_float_array(self, values: ArrayLike, *, name: str) -> FloatArray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite numeric values.")
        return array

    def _validate_calibration_inputs(
        self,
        cal_predictions: ArrayLike,
        cal_targets: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        cal_predictions_array = self._as_1d_float_array(cal_predictions, name="cal_predictions")
        cal_targets_array = self._as_1d_float_array(cal_targets, name="cal_targets")
        if cal_predictions_array.shape != cal_targets_array.shape:
            raise ValueError("cal_predictions and cal_targets must have the same shape.")
        return cal_predictions_array, cal_targets_array

    def _validate_prediction_inputs(self, predictions: ArrayLike) -> FloatArray:
        return self._as_1d_float_array(predictions, name="predictions")

    def _validate_weights(
        self,
        values: FloatArray,
        weights: Optional[Sequence[float]],
        *,
        name: str,
    ) -> Optional[FloatArray]:
        if weights is None:
            return None

        weight_array = self._as_1d_float_array(weights, name=name)
        if weight_array.shape != values.shape:
            raise ValueError(f"{name} must have the same shape as the corresponding predictions.")
        if np.any(weight_array < 0):
            raise ValueError(f"{name} must be non-negative.")
        total = float(weight_array.sum())
        if total <= 0.0:
            raise ValueError(f"{name} must sum to a positive value.")
        return weight_array

    def _weighted_mean(
        self,
        values: FloatArray,
        weights: Optional[Sequence[float]] = None,
        *,
        weight_name: str = "weights",
    ) -> float:
        if weights is None:
            return float(np.mean(values))

        weight_array = self._validate_weights(values, weights, name=weight_name)
        assert weight_array is not None
        return float(np.average(values, weights=weight_array))

    def _warn_if_outside_support(self, predictions: FloatArray) -> None:
        if not hasattr(self, "diagnostics_"):
            return
        diagnostics: CalibrationDiagnostics = self.diagnostics_
        if predictions.min() < diagnostics.prediction_min or predictions.max() > diagnostics.prediction_max:
            warnings.warn(
                "Prediction support extends beyond the calibration support; debiasing may be unstable.",
                DebiasingWarning,
                stacklevel=3,
            )

    def _set_diagnostics(
        self,
        *,
        cal_predictions: FloatArray,
        residual_std: Optional[float],
        warning_flags: Sequence[str],
        details: Dict[str, float | int | str | bool | None],
    ) -> None:
        self.diagnostics_ = CalibrationDiagnostics(
            method=self.method_name,
            n_calibration=int(cal_predictions.size),
            prediction_min=float(np.min(cal_predictions)),
            prediction_max=float(np.max(cal_predictions)),
            residual_std=None if residual_std is None else float(residual_std),
            warning_flags=tuple(warning_flags),
            details=dict(details),
        )

    def debiased_ate(
        self,
        treated_predictions: ArrayLike,
        control_predictions: ArrayLike,
        iptw_treated: Optional[Sequence[float]] = None,
        iptw_control: Optional[Sequence[float]] = None,
    ) -> float:
        """Compute a debiased average treatment effect between treated and control groups."""
        self._require_is_fit("diagnostics_", method_name="debiased_ate")

        treated_array = self._validate_prediction_inputs(treated_predictions)
        control_array = self._validate_prediction_inputs(control_predictions)

        debiased_treated = self.debiased_predictions(treated_array)
        debiased_control = self.debiased_predictions(control_array)

        mean_treated = self._weighted_mean(
            debiased_treated,
            iptw_treated,
            weight_name="iptw_treated",
        )
        mean_control = self._weighted_mean(
            debiased_control,
            iptw_control,
            weight_name="iptw_control",
        )
        return float(mean_treated - mean_control)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        del deep
        return {key: value for key, value in self.__dict__.items() if not key.endswith("_")}

    def set_params(self, **params: Any) -> "BaseDebiaser":
        for key, value in params.items():
            setattr(self, key, value)
        return self
