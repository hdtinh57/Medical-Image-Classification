"""Tests for monitoring.drift and monitoring.feedback modules."""

from __future__ import annotations

import pytest

from monitoring.drift import (
    DriftDetector,
    DriftThresholds,
    _compute_ks,
    _compute_psi,
)
from monitoring.feedback import FeedbackCollector

# ── Drift Detection Tests ────────────────────────────────────────────────


class TestPSI:
    """Population Stability Index computation."""

    def test_identical_distributions_zero_psi(self) -> None:
        dist = [0.2, 0.3, 0.5]
        thresholds = DriftThresholds()
        result = _compute_psi(dist, dist, thresholds)
        assert result.psi_value == pytest.approx(0.0, abs=1e-5)
        assert result.status == "stable"

    def test_small_shift_warning(self) -> None:
        reference = [0.2, 0.3, 0.5]
        current = [0.35, 0.25, 0.4]
        thresholds = DriftThresholds(psi_warning=0.05)
        result = _compute_psi(reference, current, thresholds)
        assert result.psi_value > 0.05
        assert result.status in ("warning", "critical")

    def test_large_shift_critical(self) -> None:
        reference = [0.1, 0.1, 0.8]
        current = [0.7, 0.2, 0.1]
        thresholds = DriftThresholds(psi_critical=0.2)
        result = _compute_psi(reference, current, thresholds)
        assert result.psi_value > 0.2
        assert result.status == "critical"

    def test_bin_contributions_length(self) -> None:
        dist = [0.25, 0.25, 0.25, 0.25]
        thresholds = DriftThresholds()
        result = _compute_psi(dist, dist, thresholds)
        assert len(result.bin_contributions) == 4


class TestKS:
    """Kolmogorov-Smirnov test."""

    def test_identical_samples_no_drift(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = _compute_ks(values, values, p_threshold=0.05)
        assert result.statistic == pytest.approx(0.0, abs=1e-5)
        assert not result.drift_detected

    def test_very_different_samples_drift(self) -> None:
        ref = [0.1, 0.15, 0.2, 0.25] * 25
        cur = [0.8, 0.85, 0.9, 0.95] * 25
        result = _compute_ks(ref, cur, p_threshold=0.05)
        assert result.statistic > 0.5
        assert result.drift_detected

    def test_empty_input_no_drift(self) -> None:
        result = _compute_ks([], [0.5], p_threshold=0.05)
        assert not result.drift_detected


class TestDriftDetector:
    """End-to-end drift detector tests."""

    def test_stable_report(self) -> None:
        class_names = ["a", "b", "c"]
        ref_dist = [0.33, 0.33, 0.34]
        ref_conf = [0.6, 0.7, 0.5, 0.8, 0.65] * 20

        detector = DriftDetector(
            class_names=class_names,
            reference_class_dist=ref_dist,
            reference_confidences=ref_conf,
        )

        recent_classes = ["a", "b", "c"] * 10
        recent_confidences = [0.65, 0.7, 0.55] * 10

        report = detector.evaluate(recent_classes, recent_confidences)
        assert report.overall_status == "stable"
        assert len(report.per_class_drift) == 3

    def test_shifted_report_warning(self) -> None:
        class_names = ["a", "b", "c"]
        ref_dist = [0.1, 0.1, 0.8]
        ref_conf = [0.9] * 100

        detector = DriftDetector(
            class_names=class_names,
            reference_class_dist=ref_dist,
            reference_confidences=ref_conf,
            thresholds=DriftThresholds(psi_warning=0.01),
        )

        # Completely different distribution
        recent_classes = ["a"] * 30
        recent_confidences = [0.3] * 30

        report = detector.evaluate(recent_classes, recent_confidences)
        assert report.overall_status in ("warning", "critical")
        assert len(report.recommendations) > 0


# ── Feedback Collector Tests ─────────────────────────────────────────────


class TestFeedbackCollector:
    """Ground-truth feedback collection and accuracy tracking."""

    @pytest.fixture()
    def collector(self) -> FeedbackCollector:
        return FeedbackCollector(class_names=["a", "b", "c"], window_size=50)

    def test_register_and_submit_correct(self, collector: FeedbackCollector) -> None:
        collector.register_prediction("pred-1", "a")
        record = collector.submit_feedback("pred-1", "a")
        assert record is not None
        assert record.correct is True

    def test_register_and_submit_incorrect(self, collector: FeedbackCollector) -> None:
        collector.register_prediction("pred-2", "a")
        record = collector.submit_feedback("pred-2", "b")
        assert record is not None
        assert record.correct is False

    def test_unknown_prediction_returns_none(self, collector: FeedbackCollector) -> None:
        result = collector.submit_feedback("nonexistent", "a")
        assert result is None

    def test_duplicate_feedback_returns_none(self, collector: FeedbackCollector) -> None:
        collector.register_prediction("pred-3", "a")
        collector.submit_feedback("pred-3", "a")
        # Second submission for same ID should fail
        result = collector.submit_feedback("pred-3", "a")
        assert result is None

    def test_snapshot_accuracy(self, collector: FeedbackCollector) -> None:
        # 3 correct, 1 incorrect = 75% accuracy
        for i, (pred, actual) in enumerate([("a", "a"), ("b", "b"), ("c", "c"), ("a", "b")]):
            collector.register_prediction(f"p-{i}", pred)
            collector.submit_feedback(f"p-{i}", actual)

        snapshot = collector.get_snapshot()
        assert snapshot.total_feedback == 4
        assert snapshot.accuracy == pytest.approx(0.75, abs=0.01)

    def test_snapshot_empty(self, collector: FeedbackCollector) -> None:
        snapshot = collector.get_snapshot()
        assert snapshot.total_feedback == 0
        assert snapshot.accuracy == 0.0

    def test_rolling_window(self) -> None:
        collector = FeedbackCollector(class_names=["a", "b"], window_size=3)
        # Fill beyond window size
        for i in range(5):
            collector.register_prediction(f"p-{i}", "a")
            collector.submit_feedback(f"p-{i}", "a" if i >= 2 else "b")

        snapshot = collector.get_snapshot()
        # Window keeps last 3: all correct (i=2,3,4 → a→a)
        assert snapshot.window_size == 3
        assert snapshot.accuracy == pytest.approx(1.0, abs=0.01)
