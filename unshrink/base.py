from __future__ import annotations
from typing import Dict, Any, Optional, Sequence
import numpy as np

class BaseDebiaser:
    """
    Base class providing sklearn-like pattern.
    """

    def fit(self, cal_predictions: np.ndarray, cal_targets: np.ndarray):
        raise NotImplementedError

    def debiased_mean(self, predictions: np.ndarray) -> float:
        raise NotImplementedError

    def debiased_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Return debiased predictions for each input (no reduction to a scalar).

        Implementations should mirror the behavior of `debiased_mean` but return
        the full array of debiased values instead of the mean.
        """
        raise NotImplementedError

    def _weighted_mean(self, values: np.ndarray, weights: Optional[Sequence[float]] = None) -> float:
        """Compute (optionally) weighted mean. Accepts lists/arrays or None for weights.

        Parameters
        - values: array-like of values
        - weights: optional array-like of non-negative weights of same length as values

        Returns
        - scalar mean (float)
        """
        vals = np.asarray(values)
        if weights is None:
            return float(np.mean(vals))

        w = np.asarray(weights)
        if w.shape != vals.shape:
            raise ValueError("weights must have the same shape as values")
        total = w.sum()
        if total == 0:
            raise ValueError("sum of weights must be > 0")
        return float((vals * w).sum() / total)

    def debiased_ate(
        self,
        treated_predictions: np.ndarray,
        control_predictions: np.ndarray,
        iptw_treated: Optional[Sequence[float]] = None,
        iptw_control: Optional[Sequence[float]] = None,
    ) -> float:
        """Compute a debiased average treatment effect (ATE) between two groups.

        This accepts raw model predictions for a treated group and a control group, 
        applies the debiasing implemented by `debiased_predictions`, and returns the 
        (optionally weighted) difference in means: 
            mean(debiased_treated) - mean(debiased_control)

        Parameters
        - treated_predictions: array-like predictions for treated units
        - control_predictions: array-like predictions for control units
        - iptw_treated: optional array-like of importance/sample weights for treated units
        - iptw_control: optional array-like of importance/sample weights for control units

        Returns
        - debiased ATE as float
        """
        # get debiased per-unit predictions from the concrete debiaser
        deb_t = self.debiased_predictions(np.asarray(treated_predictions))
        deb_c = self.debiased_predictions(np.asarray(control_predictions))

        # compute (optionally) weighted means
        mean_t = self._weighted_mean(deb_t, iptw_treated)
        mean_c = self._weighted_mean(deb_c, iptw_control)

        return float(mean_t - mean_c)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def set_params(self, **params) -> "BaseDebiaser":
        for k, v in params.items():
            setattr(self, k, v)
        return self
