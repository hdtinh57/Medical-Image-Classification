"""Reusable classification metrics for training and evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from training.model import CLASS_NAMES, NUM_CLASSES


def compute_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute JSON-safe summary and per-class multiclass metrics."""
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    _validate_metric_inputs(target_array, probability_array)
    predictions = probability_array.argmax(axis=1)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(target_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target_array, predictions)),
        "macro_f1": float(f1_score(target_array, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(target_array, predictions, average="weighted", zero_division=0)
        ),
        "per_class": classification_report(
            target_array,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }
    roc_auc = _safe_macro_roc_auc(target_array, probability_array)
    if roc_auc is not None:
        metrics["macro_ovr_roc_auc"] = roc_auc
    return metrics


def summary_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Return only scalar metrics suitable for epoch logs and MLflow."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if key != "per_class" and isinstance(value, int | float)
    }


def _validate_metric_inputs(targets: np.ndarray, probabilities: np.ndarray) -> None:
    """Validate target and probability shapes before metric calculation."""
    if targets.ndim != 1:
        raise ValueError("targets phải là vector một chiều.")
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(f"probabilities phải có shape [N, {NUM_CLASSES}].")
    if len(targets) != len(probabilities):
        raise ValueError("targets và probabilities phải có cùng số records.")
    if not len(targets):
        raise ValueError("Không thể tính metrics trên tập rỗng.")


def _safe_macro_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Compute macro OVR ROC-AUC only when every class is represented."""
    if len(np.unique(targets)) != NUM_CLASSES:
        return None
    try:
        return float(
            roc_auc_score(
                targets,
                probabilities,
                labels=list(range(NUM_CLASSES)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None
