from __future__ import annotations
from typing import Dict, Any
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

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def set_params(self, **params) -> "BaseDebiaser":
        for k, v in params.items():
            setattr(self, k, v)
        return self
