from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from .base import BaseDebiaser

class LccDebiaser(BaseDebiaser):
    """
    Linear Calibration Correction (LCC)
    """

    def __init__(self):
        pass

    def fit(self, cal_predictions: np.ndarray, cal_targets: np.ndarray) -> "LccDebiaser":
        cal_predictions_ = np.asarray(cal_predictions)
        cal_targets_ = np.asarray(cal_targets)

        model = LinearRegression()
        model.fit(cal_targets_.reshape(-1, 1), cal_predictions_)

        self.intercept_ = model.intercept_
        self.slope_ = model.coef_[0]
        return self

    def debiased_mean(self, predictions: np.ndarray) -> float:
        if not hasattr(self, "slope_"):
            raise RuntimeError("Call .fit() before .debiased_mean().")

        mean_pred = np.mean(predictions)
        return float((mean_pred - self.intercept_) / self.slope_)

    def debiased_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Return debiased predictions (vectorized) without taking the mean.
        """
        if not hasattr(self, "slope_"):
            raise RuntimeError("Call .fit() before .debiased_predictions().")

        preds = np.asarray(predictions)
        return (preds - self.intercept_) / self.slope_
