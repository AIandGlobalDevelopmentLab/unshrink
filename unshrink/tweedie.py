from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
from .base import BaseDebiaser

class TweedieDebiaser(BaseDebiaser):
    """
    Tweedie's correction
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta

    def fit(self, cal_predictions: np.ndarray, cal_targets: np.ndarray, 
            cal_predictions_sigma: np.ndarray = None, cal_targets_sigma: np.ndarray = None
            ) -> "TweedieDebiaser":
        cal_predictions_ = np.asarray(cal_predictions)
        cal_targets_ = np.asarray(cal_targets)

        # If a specific calibration set is provided for uncertainties, use it to estimate sigma
        # Otherwise, estimate sigma from the residuals on the calibration set
        if cal_predictions_sigma is not None and cal_targets_sigma is not None:
            cal_predictions_sigma_ = np.asarray(cal_predictions_sigma)
            cal_targets_sigma_ = np.asarray(cal_targets_sigma)
        elif cal_predictions_sigma is None and cal_targets_sigma is None:
            cal_predictions_sigma_ = cal_predictions_
            cal_targets_sigma_ = cal_targets_
        else:
            raise ValueError("Either both or neither of cal_predictions_sigma and cal_targets_sigma must be provided.")

        self.sigma_ = np.std(cal_predictions_sigma_ - cal_targets_sigma_)
        self.kde_ = gaussian_kde(cal_predictions_)
        return self

    def _score(self, y: np.ndarray) -> np.ndarray:
        log_p_plus = self.kde_.logpdf(y + self.delta)
        log_p_minus = self.kde_.logpdf(y - self.delta)
        return (log_p_plus - log_p_minus) / (2 * self.delta)

    def debiased_mean(self, predictions: np.ndarray) -> float:
        if not hasattr(self, "sigma_"):
            raise RuntimeError("Call .fit() before .debiased_mean().")

        mean_pred = np.mean(predictions)
        scores = self._score(predictions)
        return float(mean_pred - self.sigma_**2 * np.mean(scores))

    def debiased_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Return debiased predictions (vectorized) without taking the mean.
        """
        if not hasattr(self, "sigma_"):
            raise RuntimeError("Call .fit() before .debiased_predictions().")

        preds = np.asarray(predictions)
        scores = self._score(preds)
        return preds - self.sigma_**2 * scores
