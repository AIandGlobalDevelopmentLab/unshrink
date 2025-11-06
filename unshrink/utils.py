from __future__ import annotations
import numpy as np
from typing import Dict

def evaluate_debiaser(
    debiaser,
    cal_predictions: np.ndarray,
    cal_targets: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Fit debiaser and return naive vs corrected estimates.
    """

    debiaser.fit(cal_predictions, cal_targets)

    return {
        "true_mean": float(np.mean(targets)),
        "naive_mean": float(np.mean(predictions)),
        "corrected_mean": float(debiaser.predict_mean(predictions)),
        "bias_after": float(abs(debiaser.predict_mean(predictions) - np.mean(targets))),
        "bias_before": float(abs(np.mean(predictions) - np.mean(targets))),
    }
