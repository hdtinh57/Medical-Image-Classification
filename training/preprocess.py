"""Image preprocessing for the 9-class skin-lesion classifier."""

from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode, v2

from training.model import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD


def build_train_transform(image_size: int = IMAGE_SIZE) -> v2.Compose:
    """Build moderate augmentations that preserve clinically relevant color cues."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(
                degrees=20,
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.02,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE) -> v2.Compose:
    """Build deterministic validation and test preprocessing."""
    resize_size = round(image_size / 0.875)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(
                size=resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.CenterCrop(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
