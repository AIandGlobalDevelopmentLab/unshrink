# `unshrink`

`unshrink` debiases shrunken ML predictions before you reuse them as outcomes in causal inference workflows.

The package keeps the user-facing API compact:

```python
from unshrink import LCCDebiaser, TweedieDebiaser

debiaser = TweedieDebiaser().fit(cal_predictions, cal_targets)
debiased_mean = debiaser.debiased_mean(test_predictions)
debiased_ate = debiaser.debiased_ate(treated_predictions, control_predictions)
```

New version updates:

- stronger validation of calibration data and weights
- fitted `diagnostics_` on each debiaser
- `compare_debiasers(...)` for explicit LCC vs. Tweedie checks
- clearer guidance on when each correction is appropriate
- deterministic ATE workflow examples and broader synthetic tests

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/AIandGlobalDevelopmentLab/unshrink.git
```

For development:

```bash
git clone https://github.com/AIandGlobalDevelopmentLab/unshrink.git
cd unshrink
pip install -e .[dev]
```

Runtime dependencies:

- `numpy`
- `scipy`
- `scikit-learn`

Supported Python versions:

- `>=3.9,<3.13`

## Calibration Assumptions

These assumptions matter more than the choice of debiaser.

1. `cal_predictions` must be out-of-sample for the units in `cal_targets`.
2. Calibration predictions must come from the same upstream model pipeline as the downstream predictions you want to debias.
3. Calibration and downstream predictions should live on broadly similar support. If downstream predictions extend far beyond calibration support, the debiaser will warn because correction becomes extrapolation.
4. The target definition must match between calibration and downstream analysis. If your calibration labels and downstream estimand are on different scales, neither LCC nor Tweedie can rescue that mismatch.

If you trained the upstream model on all labeled data, use cross-fitting or a held-out calibration split before fitting a debiaser.

## Which Debiaser When

### Use `LCCDebiaser` when

- the prediction-vs-truth relationship is roughly linear
- you want an interpretable global rescaling
- calibration sample size is modest
- tail behavior is not the main concern

`LCCDebiaser` fits `prediction = intercept + slope * truth` on the calibration set and inverts that relationship. It is the right baseline when shrinkage looks like a single slope less than one.

### Use `TweedieDebiaser` when

- shrinkage is clearly nonlinear, especially in the tails
- you want a local correction instead of one global slope
- calibration support is reasonably dense
- KDE-based score estimation is plausible for your calibration sample size

`TweedieDebiaser` uses a Gaussian-noise estimate plus a density-score correction. It is more flexible, but it relies more heavily on calibration support and smooth density estimation.

### Recommended workflow

Run `compare_debiasers(...)` on the calibration data first, then fit the preferred debiaser on the full calibration split.

```python
from unshrink import compare_debiasers

report = compare_debiasers(cal_predictions, cal_targets)
print(report.recommended_method)
print(report.metrics["lcc"])
print(report.metrics["tweedie"])
```

The comparison report uses held-out folds from the calibration sample and compares:

- mean recovery
- pseudo-ATE recovery from repeated subgroup contrasts
- calibration slope of estimated versus true subgroup contrasts

It does not silently auto-switch methods for you. The recommendation is explicit and inspectable.

## Quick Start

### Mean estimation

```python
from unshrink import LCCDebiaser

debiaser = LCCDebiaser().fit(cal_predictions, cal_targets)
corrected_mean = debiaser.debiased_mean(test_predictions)
print(debiaser.diagnostics_)
```

### ATE estimation

```python
from unshrink import TweedieDebiaser

debiaser = TweedieDebiaser().fit(cal_predictions, cal_targets)

ate = debiaser.debiased_ate(
    treated_predictions,
    control_predictions,
    iptw_treated=treated_weights,
    iptw_control=control_weights,
)
```

### Deterministic example

A runnable synthetic workflow lives in `examples/ate_workflow.py`. It:

1. simulates a treatment/outcome workflow
2. compares LCC and Tweedie on a calibration split
3. fits the recommended debiaser
4. reports naive, debiased, and true ATE values

Run it with:

```bash
python examples/ate_workflow.py
```

## Public API

### Debiasers

- `LCCDebiaser` and `LccDebiaser`
- `TweedieDebiaser`

Common methods:

- `fit(cal_predictions, cal_targets, ...)`
- `debiased_predictions(predictions)`
- `debiased_mean(predictions)`
- `debiased_ate(treated_predictions, control_predictions, iptw_treated=None, iptw_control=None)`

Fitted debiasers expose:

- `diagnostics_`: calibration support, residual scale, warnings, and method-specific details

### Utilities

- `evaluate_debiaser(...)`: returns a small dict with naive versus corrected mean bias
- `compare_debiasers(...)`: returns a typed `DebiaserComparisonReport`

### Report types

- `CalibrationDiagnostics`
- `DebiaserMetrics`
- `DebiaserComparisonReport`

## Edge Cases and Failure Modes

The package will raises errors on:

- empty, non-finite, or non-1D inputs
- mismatched calibration array lengths
- negative or zero-sum IPTW weights
- non-identifiable LCC fits such as constant calibration targets or non-positive slopes
- Tweedie fits with invalid `delta` or singular/no-support KDE inputs

The package warns on:

- very small positive LCC slopes
- small Tweedie calibration samples
- downstream predictions outside calibration support

Warnings indicate “computable but fragile,” not “safe to ignore.”

## Citation

Paper:

- [arXiv:2508.01341](https://arxiv.org/abs/2508.01341)
- [Project page](https://aidevlab.org/tweedie/)

```bibtex
@inproceedings{pettersson2025debiasingmachinelearningpredictions,
  title        = {Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data: One Map, Many Trials in Satellite-Driven Poverty Analysis},
  author       = {Markus Pettersson and Connor T. Jerzak and Adel Daoud},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), Special Track on AI for Social Impact},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i46.41258}
}
```
