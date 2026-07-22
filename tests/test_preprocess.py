"""Tests for deterministic image preprocessing."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from training.model import IMAGE_SIZE
from training.preprocess import build_eval_transform, build_train_transform


def _sample_image() -> Image.Image:
    pixels = np.linspace(0, 255, num=320 * 240 * 3, dtype=np.uint8).reshape(240, 320, 3)
    return Image.fromarray(pixels, mode="RGB")


def test_eval_transform_is_deterministic_and_has_expected_shape() -> None:
    transform = build_eval_transform()

    first = transform(_sample_image())
    second = transform(_sample_image())

    assert first.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_train_transform_returns_normalized_tensor() -> None:
    torch.manual_seed(42)

    transformed = build_train_transform()(_sample_image())

    assert transformed.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert transformed.dtype == torch.float32
    assert torch.isfinite(transformed).all()
