"""Statistical drift detection for production model monitoring.

Compares recent prediction distributions against a reference baseline
using Kolmogorov-Smirnov test and Population Stability Index (PSI).

Usage:
    from monitoring.drift import DriftDetector
    detector = DriftDetector(reference_distribution, class_names)
    report = detector.evaluate(recent_predictions)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable thresholds for drift alerts."""

    psi_warning: float = 0.1
    psi_critical: float = 0.25
    ks_p_value: float = 0.05
    confidence_mean_shift: float = 0.1


@dataclass
class PSIResult:
    """Population Stability Index result for one distribution comparison."""

    psi_value: float
    status: str  # "stable", "warning", "critical"
    bin_contributions: list[float]


@dataclass
class KSResult:
    """Kolmogorov-Smirnov test result."""

    statistic: float
    p_value: float
    drift_detected: bool


@dataclass
class ConfidenceShift:
    """Confidence distribution shift analysis."""

    reference_mean: float
    current_mean: float
    shift: float
    drift_detected: bool


@dataclass
class DriftReport:
    """Complete drift analysis report."""

    timestamp: str
    window_size: int
    class_distribution_psi: PSIResult
    confidence_ks: KSResult
    confidence_shift: ConfidenceShift
    per_class_drift: dict[str, float]
    overall_status: str  # "stable", "warning", "critical"
    recommendations: list[str]


def _compute_psi(
    reference: list[float],
    current: list[float],
    thresholds: DriftThresholds,
) -> PSIResult:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1  → stable
    PSI 0.1–0.25 → warning (moderate shift)
    PSI > 0.25 → critical (significant shift)
    """
    epsilon = 1e-6
    contributions = []
    psi_total = 0.0

    for ref_p, cur_p in zip(reference, current):
        ref_safe = max(ref_p, epsilon)
        cur_safe = max(cur_p, epsilon)
        contribution = (cur_safe - ref_safe) * math.log(cur_safe / ref_safe)
        contributions.append(round(contribution, 6))
        psi_total += contribution

    psi_total = round(psi_total, 6)

    if psi_total >= thresholds.psi_critical:
        status = "critical"
    elif psi_total >= thresholds.psi_warning:
        status = "warning"
    else:
        status = "stable"

    return PSIResult(psi_value=psi_total, status=status, bin_contributions=contributions)


def _compute_ks(
    reference_values: list[float],
    current_values: list[float],
    p_threshold: float,
) -> KSResult:
    """Compute two-sample Kolmogorov-Smirnov test.

    Uses a manual implementation to avoid scipy dependency in the gateway.
    """
    if not reference_values or not current_values:
        return KSResult(statistic=0.0, p_value=1.0, drift_detected=False)

    combined = sorted(set(reference_values + current_values))
    n_ref = len(reference_values)
    n_cur = len(current_values)

    ref_sorted = sorted(reference_values)
    cur_sorted = sorted(current_values)

    max_diff = 0.0
    ref_idx = 0
    cur_idx = 0

    for value in combined:
        while ref_idx < n_ref and ref_sorted[ref_idx] <= value:
            ref_idx += 1
        while cur_idx < n_cur and cur_sorted[cur_idx] <= value:
            cur_idx += 1
        diff = abs(ref_idx / n_ref - cur_idx / n_cur)
        max_diff = max(max_diff, diff)

    # Approximate p-value using asymptotic formula
    n_eff = (n_ref * n_cur) / (n_ref + n_cur)
    lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * max_diff

    # Kolmogorov distribution approximation
    p_value = _kolmogorov_p_value(lambda_val)

    return KSResult(
        statistic=round(max_diff, 6),
        p_value=round(p_value, 6),
        drift_detected=p_value < p_threshold,
    )


def _kolmogorov_p_value(lambda_val: float) -> float:
    """Approximate Kolmogorov distribution survival function."""
    if lambda_val <= 0:
        return 1.0
    if lambda_val >= 3.0:
        return 0.0

    # Series approximation: P(D > x) ≈ 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2k^2 * x^2)
    total = 0.0
    for k in range(1, 20):
        sign = (-1) ** (k - 1)
        total += sign * math.exp(-2.0 * k * k * lambda_val * lambda_val)
    return max(0.0, min(1.0, 2.0 * total))


def _compute_confidence_shift(
    reference_confidences: list[float],
    current_confidences: list[float],
    threshold: float,
) -> ConfidenceShift:
    """Detect shift in model confidence distribution."""
    ref_mean = sum(reference_confidences) / max(len(reference_confidences), 1)
    cur_mean = sum(current_confidences) / max(len(current_confidences), 1)
    shift = abs(cur_mean - ref_mean)

    return ConfidenceShift(
        reference_mean=round(ref_mean, 4),
        current_mean=round(cur_mean, 4),
        shift=round(shift, 4),
        drift_detected=shift > threshold,
    )


@dataclass
class DriftDetector:
    """Stateful drift detector comparing production predictions to a baseline.

    Reference is set once from validation/initial production data.
    Call ``evaluate()`` with recent prediction windows to check for drift.
    """

    class_names: list[str]
    reference_class_dist: list[float]
    reference_confidences: list[float]
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    def evaluate(
        self,
        recent_classes: list[str],
        recent_confidences: list[float],
    ) -> DriftReport:
        """Run full drift analysis on a window of recent predictions."""
        window_size = len(recent_classes)

        # Build current class distribution
        class_counts = {name: 0 for name in self.class_names}
        for cls in recent_classes:
            if cls in class_counts:
                class_counts[cls] += 1
        total = max(sum(class_counts.values()), 1)
        current_dist = [class_counts[name] / total for name in self.class_names]

        # PSI on class distribution
        psi_result = _compute_psi(self.reference_class_dist, current_dist, self.thresholds)

        # KS test on confidence values
        ks_result = _compute_ks(
            self.reference_confidences,
            recent_confidences,
            self.thresholds.ks_p_value,
        )

        # Confidence mean shift
        conf_shift = _compute_confidence_shift(
            self.reference_confidences,
            recent_confidences,
            self.thresholds.confidence_mean_shift,
        )

        # Per-class drift (absolute shift)
        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = round(abs(current_dist[i] - self.reference_class_dist[i]), 4)

        # Overall status
        recommendations: list[str] = []
        if psi_result.status == "critical":
            overall = "critical"
            recommendations.append(
                "Class distribution đã thay đổi đáng kể. Kiểm tra data pipeline."
            )
        elif psi_result.status == "warning" or ks_result.drift_detected:
            overall = "warning"
            recommendations.append("Phát hiện dấu hiệu drift. Theo dõi thêm hoặc trigger retrain.")
        elif conf_shift.drift_detected:
            overall = "warning"
            recommendations.append(
                "Confidence trung bình thay đổi. Model có thể cần recalibration."
            )
        else:
            overall = "stable"

        if ks_result.drift_detected:
            recommendations.append(
                f"KS test p-value={ks_result.p_value} < {self.thresholds.ks_p_value}. "
                "Confidence distribution đã thay đổi."
            )

        return DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            window_size=window_size,
            class_distribution_psi=psi_result,
            confidence_ks=ks_result,
            confidence_shift=conf_shift,
            per_class_drift=per_class,
            overall_status=overall,
            recommendations=recommendations,
        )


def save_report(report: DriftReport, path: Path) -> None:
    """Persist a drift report as JSON for audit and Grafana integration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference_from_predictions(predictions_csv: Path) -> tuple[list[str], list[float]]:
    """Load reference distribution from a validation predictions.csv file.

    Returns (class_names_list, confidence_list) for the reference window.
    """
    import csv

    classes: list[str] = []
    confidences: list[float] = []

    with open(predictions_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "predicted" in row:
                classes.append(row["predicted"])
            if "confidence" in row:
                confidences.append(float(row["confidence"]))
            elif "max_prob" in row:
                confidences.append(float(row["max_prob"]))

    return classes, confidences
