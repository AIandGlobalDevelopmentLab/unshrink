
<!---
<p align="center">
<a href="github.com/AIandGlobalDevelopmentLab/unshrink#gh-light-mode-only">
  <img src="https://github.com/AIandGlobalDevelopmentLab/unshrink/blob/main/inst/logo.webp?raw=true#gh-light-mode-only" alt="asa logo" width="400">
</a>

<a href="github.com/AIandGlobalDevelopmentLab/unshrink#gh-dark-mode-only">
  <img src="https://github.com/AIandGlobalDevelopmentLab/unshrink/blob/main/inst/logo_dark.webp?raw=true#gh-dark-mode-only" alt="asa logo" width="400">
</a>

</p>
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2508.01341" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-750014" /></a>
    <a style="text-decoration:none !important;" href="https://github.com/AIandGlobalDevelopmentLab/unshrink/actions/workflows/ci.yml" alt="CI"><img src="https://github.com/AIandGlobalDevelopmentLab/unshrink/actions/workflows/ci.yml/badge.svg" /></a>
</p>
-->


```
 ╔══════════════════════════════════════════════════════════════════╗
 ║  UNSHRINK v0.1.0                         STATUS: DEBIASED        ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  METHOD: Tweedie (Berkson / score-swap)                          ║
 ║  PSEUDO-OUTCOME:  y_tilde = z - sigma^2 * d/dz( log p_Y_hat(z) ) ║
 ║                                                                  ║
 ║  [INPUT: BIASED]               [OUTPUT: RESTORED]                ║
 ║  Variance: 0.42 (▼ 58%)        Variance: 1.00 (✔ OK)             ║
 ║                                                                  ║
 ║      SpikeAtMean                  TrueDistribution               ║
 ║          │                              │                        ║
 ║       ▄▄██▄▄                       ▂▄▆██▆▄▂                      ║
 ║      ▄██████▄                    ▄██████████▄                    ║
 ║     ██▀    ▀██                  ▄█▀          ▀█▄                 ║
 ║     ─┴────────┴─                ─┴────────────┴─                 ║
 ╚══════════════════════════════════════════════════════════════════╝

```

## Overview of unshrink 

**unshrink** is a Python package for correcting **attenuation bias** in machine learning predictions used for causal inference. When ML models predict outcomes (e.g., poverty levels from satellite imagery), their predictions systematically "shrink toward the mean"—overestimating low values and underestimating high values. This shrinkage causes treatment effects to appear smaller than they actually are.

The package provides two debiasing methods:
- **TweedieDebiaser**: Empirical Bayes correction using Tweedie's formula with KDE-based score estimation
- **LccDebiaser**: Linear Calibration Correction via inverse linear regression

Both methods require only a calibration dataset where you have both predictions and ground truth outcomes.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/AIandGlobalDevelopmentLab/unshrink.git
```

For development:
```bash
git clone https://github.com/AIandGlobalDevelopmentLab/unshrink.git
cd unshrink
pip install -e .[dev]
```

## Quick Start

### Using TweedieDebiaser

```python
from unshrink import TweedieDebiaser

# Fit on calibration data (where you have ground truth)
debiaser = TweedieDebiaser()
debiaser.fit(cal_predictions, cal_targets)

# Get debiased predictions for new data
debiased_preds = debiaser.debiased_predictions(test_predictions)

# Or get just the debiased mean
debiased_mean = debiaser.debiased_mean(test_predictions)
```

### Using LccDebiaser

```python
from unshrink import LccDebiaser

# Linear Calibration Correction - simpler, works well with linear shrinkage
debiaser = LccDebiaser()
debiaser.fit(cal_predictions, cal_targets)

debiased_preds = debiaser.debiased_predictions(test_predictions)
```

### Input Requirements

- `cal_predictions`: numpy array of shape `(n_cal,)` — model predictions for units with ground truth
- `cal_targets`: numpy array of shape `(n_cal,)` — true outcomes for the same units
- `test_predictions`: numpy array of shape `(n_test,)` — predictions to debias

**Important**: Calibration predictions should be out-of-sample (use cross-fitting if needed to avoid target leakage).

## When to Use Which Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **TweedieDebiaser** | Nonlinear shrinkage patterns | More flexible; handles complex bias | Requires more calibration data; sensitive to density estimation |
| **LccDebiaser** | Linear shrinkage patterns | Simple; robust with small samples | Assumes linear relationship between predictions and targets |

**Rule of thumb**: Start with `LccDebiaser` for simplicity. Use `TweedieDebiaser` if you have ample calibration data (n > 500) or suspect nonlinear shrinkage.

### Estimating Treatment Effects

```python
# Compute debiased Average Treatment Effect (ATE)
ate = debiaser.debiased_ate(
    treated_predictions,
    control_predictions,
    iptw_treated=treated_weights,  # optional inverse probability weights
    iptw_control=control_weights
)
```

## API Reference

### TweedieDebiaser

Uses Tweedie's formula with a KDE-based score function to correct prediction bias.

```python
TweedieDebiaser(delta=1e-5)
```

**Parameters:**
- `delta` (float): Step size for numerical differentiation of the log-density. Default: `1e-5`

**Methods:**
- `fit(cal_predictions, cal_targets)` — Calibrate using labeled data. Learns `sigma_` (residual std) and `kde_` (kernel density estimate).
- `debiased_predictions(predictions)` — Returns per-unit debiased predictions as an array.
- `debiased_mean(predictions)` — Returns the debiased mean as a scalar.
- `debiased_ate(treated, control, iptw_treated=None, iptw_control=None)` — Returns the debiased average treatment effect.

### LccDebiaser

Linear Calibration Correction applies an inverse linear transformation learned from calibration data.

```python
LccDebiaser()
```

**Methods:**
- `fit(cal_predictions, cal_targets)` — Fits a linear regression to learn `intercept_` and `slope_`.
- `debiased_predictions(predictions)` — Returns `(predictions - intercept) / slope`.
- `debiased_mean(predictions)` — Returns the debiased mean as a scalar.
- `debiased_ate(treated, control, iptw_treated=None, iptw_control=None)` — Returns the debiased average treatment effect.

## Citation

**Paper**: [arXiv:2508.01341](https://arxiv.org/abs/2508.01341) | [Project Page](https://aidevlab.org/tweedie/)

```bibtex
@inproceedings{pettersson2025debiasingmachinelearningpredictions,
  title        = {Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data: One Map, Many Trials in Satellite-Driven Poverty Analysis},
  author       = {Markus Pettersson and Connor T. Jerzak and Adel Daoud},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), Special Track on AI for Social Impact},
  year         = {2026},
  url          = {https://arxiv.org/abs/2508.01341}
}
```
