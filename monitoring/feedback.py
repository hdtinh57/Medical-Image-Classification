"""Ground-truth feedback collection and production accuracy tracking.

Provides a feedback endpoint for collecting actual diagnoses after prediction,
and computes rolling production accuracy metrics for monitoring.

Gateway endpoint POST /feedback accepts:
    { "prediction_id": str, "actual_class": str }
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Gauge

# Prometheus metrics for feedback tracking
FEEDBACK_TOTAL = Counter(
    "gateway_feedback_total",
    "Total feedback submissions received.",
)

FEEDBACK_CORRECT = Counter(
    "gateway_feedback_correct_total",
    "Feedback where prediction matched actual class.",
)

FEEDBACK_INCORRECT = Counter(
    "gateway_feedback_incorrect_total",
    "Feedback where prediction did not match actual class.",
    ["predicted_class", "actual_class"],
)

PRODUCTION_ACCURACY = Gauge(
    "gateway_production_accuracy",
    "Rolling production accuracy from feedback (last N samples).",
)

PRODUCTION_ACCURACY_PER_CLASS = Gauge(
    "gateway_production_accuracy_per_class",
    "Rolling production accuracy per class.",
    ["class_name"],
)

_ROLLING_WINDOW = 200


@dataclass
class FeedbackRecord:
    """One ground-truth feedback entry."""

    prediction_id: str
    predicted_class: str
    actual_class: str
    correct: bool
    timestamp: str


@dataclass
class PerformanceSnapshot:
    """Production performance at a point in time."""

    timestamp: str
    window_size: int
    total_feedback: int
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_pairs: list[dict[str, str | int]]


class FeedbackCollector:
    """Thread-safe feedback collector with rolling accuracy computation."""

    def __init__(self, class_names: list[str], window_size: int = _ROLLING_WINDOW) -> None:
        self._class_names = list(class_names)
        self._window_size = window_size
        self._records: deque[FeedbackRecord] = deque(maxlen=window_size)
        self._total_count = 0
        self._lock = threading.Lock()
        self._pending_predictions: dict[str, str] = {}
        self._pending_lock = threading.Lock()

    def register_prediction(self, prediction_id: str, predicted_class: str) -> None:
        """Register a prediction for later feedback matching."""
        with self._pending_lock:
            self._pending_predictions[prediction_id] = predicted_class
            _cleanup_old_predictions(self._pending_predictions)

    def submit_feedback(self, prediction_id: str, actual_class: str) -> FeedbackRecord | None:
        """Submit ground-truth feedback for a previous prediction."""
        with self._pending_lock:
            predicted = self._pending_predictions.pop(prediction_id, None)

        if predicted is None:
            return None

        correct = predicted == actual_class
        record = FeedbackRecord(
            prediction_id=prediction_id,
            predicted_class=predicted,
            actual_class=actual_class,
            correct=correct,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._records.append(record)
            self._total_count += 1

        # Update Prometheus metrics
        FEEDBACK_TOTAL.inc()
        if correct:
            FEEDBACK_CORRECT.inc()
        else:
            FEEDBACK_INCORRECT.labels(
                predicted_class=predicted,
                actual_class=actual_class,
            ).inc()

        self._update_rolling_accuracy()
        return record

    def _update_rolling_accuracy(self) -> None:
        """Recompute rolling accuracy and update Prometheus gauges."""
        with self._lock:
            records = list(self._records)

        if not records:
            return

        correct_count = sum(1 for r in records if r.correct)
        accuracy = correct_count / len(records)
        PRODUCTION_ACCURACY.set(round(accuracy, 4))

        # Per-class accuracy
        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}
        for r in records:
            class_total[r.actual_class] = class_total.get(r.actual_class, 0) + 1
            if r.correct:
                class_correct[r.actual_class] = class_correct.get(r.actual_class, 0) + 1

        for cls in self._class_names:
            total = class_total.get(cls, 0)
            correct = class_correct.get(cls, 0)
            cls_acc = correct / total if total > 0 else 0.0
            PRODUCTION_ACCURACY_PER_CLASS.labels(class_name=cls).set(round(cls_acc, 4))

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        with self._lock:
            records = list(self._records)

        if not records:
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                window_size=0,
                total_feedback=self._total_count,
                accuracy=0.0,
                per_class_accuracy={},
                confusion_pairs=[],
            )

        correct_count = sum(1 for r in records if r.correct)
        accuracy = correct_count / len(records)

        # Per-class
        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}
        for r in records:
            class_total[r.actual_class] = class_total.get(r.actual_class, 0) + 1
            if r.correct:
                class_correct[r.actual_class] = class_correct.get(r.actual_class, 0) + 1

        per_class = {}
        for cls in self._class_names:
            total = class_total.get(cls, 0)
            correct = class_correct.get(cls, 0)
            per_class[cls] = round(correct / total, 4) if total > 0 else 0.0

        # Top confusion pairs
        confusion: dict[tuple[str, str], int] = {}
        for r in records:
            if not r.correct:
                key = (r.predicted_class, r.actual_class)
                confusion[key] = confusion.get(key, 0) + 1

        sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]
        confusion_list = [{"predicted": p, "actual": a, "count": c} for (p, a), c in sorted_pairs]

        return PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            window_size=len(records),
            total_feedback=self._total_count,
            accuracy=round(accuracy, 4),
            per_class_accuracy=per_class,
            confusion_pairs=confusion_list,
        )

    def save_snapshot(self, path: Path) -> None:
        """Save current performance snapshot to disk."""
        snapshot = self.get_snapshot()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(snapshot), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _cleanup_old_predictions(
    pending: dict[str, str],
    max_size: int = 10000,
) -> None:
    """Prevent unbounded memory growth in pending predictions."""
    if len(pending) > max_size:
        keys = list(pending.keys())
        for key in keys[: len(keys) - max_size]:
            pending.pop(key, None)
