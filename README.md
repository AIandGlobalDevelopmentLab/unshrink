
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
 ╔══════════════════════════════════════════════════════════════╗
 ║  UNSHRINK v0.1.0                         STATUS: DEBIASING   ║
 ╠══════════════════════════════════════════════════════════════╣
 ║  METHOD: Tweedie                                             ║
 ║  FORMULA: E[μ|z] = z - σ² · d/dz(log p(z))                   ║
 ║                                                              ║
 ║  [INPUT: BIASED]             [OUTPUT: RESTORED]              ║
 ║  Variance: 0.42 (▼ 58%)      Variance: 1.01 (✔ OK)           ║
 ║                                                              ║
 ║      SpikeAtMean                 TrueDistribution            ║
 ║          │                              │                    ║
 ║       ▄▄██▄▄                       ▂▄▆██▆▄▂                  ║
 ║      ▄██████▄                    ▄██████████▄                ║
 ║     ██▀    ▀██                 ▄█▀          ▀█▄              ║
 ║    ─┴────────┴─               ─┴──────────────┴─             ║
 ╚══════════════════════════════════════════════════════════════╝
```

## Overview

**unshrink** is a Python package for debiasing machine learning predictions when there is a distribution shift between calibration data and test data. It corrects systematic biases that arise when model predictions are calibrated on one distribution but applied to another.

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

```bibtex
@inproceedings{pettersson2025debiasingmachinelearningpredictions,
  title        = {Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data: One Map, Many Trials in Satellite-Driven Poverty Analysis},
  author       = {Markus Pettersson and Connor T. Jerzak and Adel Daoud},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), Special Track on AI for Social Impact},
  year         = {2026},
}
```
