<p align="center">
    <img width="400" height="400" alt="ca118993-b696-4bad-9116-ed5f6c293063" src="https://github.com/user-attachments/assets/f8610695-61f6-4789-a715-3ce8156bf35b" />
</p>
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2508.01341" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-750014" /></a>
    <a style="text-decoration:none !important;" href="https://github.com/AIandGlobalDevelopmentLab/debiasor/actions/workflows/ci.yml" alt="CI"><img src="https://github.com/AIandGlobalDevelopmentLab/debiasor/actions/workflows/ci.yml/badge.svg" /></a>
</p>

## Example
```
debiaser = TweedieDebiaser()
debiaser.fit(cal_predictions, cal_targets)
debiased_predictions = debiaser.debiased_predictions(predictions)
```
