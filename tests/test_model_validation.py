"""Tests for the shared model, checkpoint, and metric contract."""

from __future__ import annotations

import numpy as np
import torch

from training.metrics import compute_classification_metrics
from training.model import (
    BASELINE_ARCH,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    build_checkpoint,
    create_model,
    load_model_from_checkpoint,
)


def test_model_outputs_nine_logits() -> None:
    model = create_model(BASELINE_ARCH, pretrained=False)

    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))

    assert logits.shape == (2, NUM_CLASSES)


def test_checkpoint_round_trip_preserves_contract(tmp_path) -> None:
    model = create_model(BASELINE_ARCH, pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint = build_checkpoint(
        model,
        architecture=BASELINE_ARCH,
        epoch=3,
        metrics={"macro_f1": 0.5},
    )
    torch.save(checkpoint, checkpoint_path)

    restored_model, restored_checkpoint = load_model_from_checkpoint(checkpoint_path)

    assert tuple(restored_checkpoint["class_names"]) == CLASS_NAMES
    assert restored_checkpoint["num_classes"] == NUM_CLASSES
    assert restored_checkpoint["epoch"] == 3
    with torch.inference_mode():
        logits = restored_model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (1, NUM_CLASSES)


def test_classification_metrics_include_macro_and_per_class_results() -> None:
    targets = np.arange(NUM_CLASSES)
    probabilities = np.eye(NUM_CLASSES, dtype=np.float32)

    metrics = compute_classification_metrics(targets, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_ovr_roc_auc"] == 1.0
    assert set(CLASS_NAMES).issubset(metrics["per_class"])
