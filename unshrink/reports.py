from __future__ import annotations

from dataclasses import asdict, dataclass, field


class DebiasingWarning(UserWarning):
    """Warning raised for statistically weak but still computable debiasing setups."""


@dataclass(frozen=True)
class CalibrationDiagnostics:
    method: str
    n_calibration: int
    prediction_min: float
    prediction_max: float
    residual_std: float | None
    warning_flags: tuple[str, ...] = ()
    details: dict[str, float | int | str | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DebiaserMetrics:
    method: str
    mean_bias: float
    mean_absolute_error: float
    rmse: float
    calibration_slope: float
    pseudo_ate_bias: float
    pseudo_ate_rmse: float
    warning_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DebiaserComparisonReport:
    metrics: dict[str, DebiaserMetrics]
    recommended_method: str
    rationale: str
    n_splits: int
    n_contrast_draws: int
    random_state: int
    warning_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "recommended_method": self.recommended_method,
            "rationale": self.rationale,
            "n_splits": self.n_splits,
            "n_contrast_draws": self.n_contrast_draws,
            "random_state": self.random_state,
            "warning_flags": self.warning_flags,
        }
