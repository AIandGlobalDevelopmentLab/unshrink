import numpy as np
import pytest


def _linear_shrinkage_sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    targets = rng.normal(loc=0.2, scale=1.1, size=n)
    predictions = 0.15 + 0.62 * targets + rng.normal(0.0, 0.12, size=n)
    return predictions, targets


def _nonlinear_shrinkage_sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([-2.2, -0.3, 1.9])
    weights = np.array([0.22, 0.43, 0.35])
    component_sd = 0.35
    components = rng.choice(means, size=n, p=weights)
    predictions = components + rng.normal(0.0, component_sd, size=n)

    centered = predictions[:, None] - means[None, :]
    kernels = weights[None, :] * np.exp(-0.5 * (centered / component_sd) ** 2)
    score = np.sum(kernels * (-centered / component_sd**2), axis=1) / np.sum(kernels, axis=1)
    coefficient = 1.0 / np.var(score)
    targets = predictions - coefficient * score
    return predictions, targets


@pytest.fixture
def no_noise_data():
    preds = np.linspace(0, 1, 1000)
    return preds, preds.copy()


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(0)
    n_cal = 5000
    n_inf = 3000
    noise = 0.2

    cal_preds = rng.random(n_cal)
    cal_targets = cal_preds + rng.normal(0, noise, n_cal)

    preds = rng.random(n_inf)
    targets = preds + rng.normal(0, noise, n_inf)

    mask = targets > 0.8
    return (cal_preds, cal_targets, preds[mask], targets[mask])


@pytest.fixture
def linear_shrinkage_data():
    rng = np.random.default_rng(42)
    cal_preds, cal_targets = _linear_shrinkage_sample(rng, 1500)
    preds, targets = _linear_shrinkage_sample(rng, 1000)
    return cal_preds, cal_targets, preds, targets


@pytest.fixture
def nonlinear_shrinkage_data():
    rng = np.random.default_rng(7)
    cal_preds, cal_targets = _nonlinear_shrinkage_sample(rng, 1800)
    preds, targets = _nonlinear_shrinkage_sample(rng, 1200)
    return cal_preds, cal_targets, preds, targets
