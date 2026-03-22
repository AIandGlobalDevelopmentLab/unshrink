from __future__ import annotations

import numpy as np

from unshrink import LCCDebiaser, TweedieDebiaser, compare_debiasers


def _simulate_workflow(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 2500
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * x1 + 0.3 * x2)))
    tau = 0.45 + 0.2 * (x1 > 0.0)
    baseline = 0.8 * x1 - 0.4 * x2 + 0.5 * np.tanh(x1 * x2)
    targets = baseline + tau * treatment + rng.normal(0.0, 0.35, n)

    # The upstream predictor is intentionally shrunk and slightly nonlinear.
    predictions = 0.1 + 0.7 * targets - 0.18 * np.tanh(targets) + rng.normal(0.0, 0.12, n)
    return predictions, targets, treatment


def main() -> None:
    predictions, targets, treatment = _simulate_workflow()
    calibration_slice = slice(0, 1000)
    trial_slice = slice(1000, None)

    cal_predictions = predictions[calibration_slice]
    cal_targets = targets[calibration_slice]
    trial_predictions = predictions[trial_slice]
    trial_targets = targets[trial_slice]
    trial_treatment = treatment[trial_slice]

    report = compare_debiasers(
        cal_predictions,
        cal_targets,
        n_splits=5,
        n_contrast_draws=40,
        random_state=0,
    )
    debiaser = LCCDebiaser() if report.recommended_method == "lcc" else TweedieDebiaser()
    debiaser.fit(cal_predictions, cal_targets)

    treated_predictions = trial_predictions[trial_treatment == 1]
    control_predictions = trial_predictions[trial_treatment == 0]
    true_ate = float(np.mean(trial_targets[trial_treatment == 1]) - np.mean(trial_targets[trial_treatment == 0]))
    naive_ate = float(np.mean(treated_predictions) - np.mean(control_predictions))
    debiased_ate = float(debiaser.debiased_ate(treated_predictions, control_predictions))

    print("Calibration recommendation:", report.recommended_method)
    print("Naive ATE:", round(naive_ate, 4))
    print("Debiased ATE:", round(debiased_ate, 4))
    print("True ATE:", round(true_ate, 4))
    print("Chosen diagnostics:", debiaser.diagnostics_.to_dict())


if __name__ == "__main__":
    main()
