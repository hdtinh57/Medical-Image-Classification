"""Shared model and class contract for training, evaluation, and serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

MODEL_NAME = "skin_classifier"
CLASS_NAMES = (
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
)
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
BASELINE_ARCH = "resnet18"
PRIMARY_ARCH = "convnextv2_tiny.fcmae_ft_in22k_in1k"
SUPPORTED_ARCHITECTURES = (BASELINE_ARCH, PRIMARY_ARCH)
CHECKPOINT_FORMAT_VERSION = 1


def create_model(
    architecture: str = BASELINE_ARCH,
    *,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Create a timm classifier that follows the shared output contract."""
    return timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the backbone while always keeping the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable

    classifier = model.get_classifier()
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def build_checkpoint(
    model: nn.Module,
    *,
    architecture: str,
    epoch: int,
    metrics: dict[str, float],
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-describing checkpoint payload."""
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
    return payload


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate that a checkpoint is compatible with the current serving contract."""
    required = {"architecture", "state_dict", "class_names", "num_classes", "image_size"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint thiếu field bắt buộc: {', '.join(missing)}")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError("Class order trong checkpoint không khớp contract 9 lớp hiện tại.")
    if int(checkpoint["num_classes"]) != NUM_CLASSES:
        raise ValueError(f"Checkpoint phải có {NUM_CLASSES} output classes.")
    if int(checkpoint["image_size"]) != IMAGE_SIZE:
        raise ValueError(f"Checkpoint phải dùng input size {IMAGE_SIZE}.")


def load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate one self-describing checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint phải là một dictionary.")
    validate_checkpoint(checkpoint)
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a model and its metadata from a self-describing checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    model = create_model(
        str(checkpoint["architecture"]),
        pretrained=False,
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def write_class_mapping(output_path: Path) -> None:
    """Write the canonical class mapping as a portable JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(CLASS_TO_INDEX, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
