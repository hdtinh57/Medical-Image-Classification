"""Tests for grouped dataset manifests and leakage protection."""

from __future__ import annotations

import pandas as pd

from training.dataset import SplitConfig, build_training_manifest, summarize_manifest
from training.model import CLASS_NAMES


def _synthetic_index() -> pd.DataFrame:
    records = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for image_index in range(5):
            records.append(
                {
                    "relative_path": f"Train/{class_name}/image-{image_index}.jpg",
                    "split": "Train",
                    "class_name": class_name,
                    "sha256": f"train-{class_index}-{image_index}",
                }
            )
        records.append(
            {
                "relative_path": f"Test/{class_name}/image.jpg",
                "split": "Test",
                "class_name": class_name,
                "sha256": f"test-{class_index}",
            }
        )

    records.append(
        {
            "relative_path": f"Train/{CLASS_NAMES[0]}/duplicate.jpg",
            "split": "Train",
            "class_name": CLASS_NAMES[0],
            "sha256": "train-0-0",
        }
    )
    return pd.DataFrame.from_records(records)


def test_grouped_manifest_preserves_test_and_prevents_train_val_overlap() -> None:
    split_config = SplitConfig(n_splits=5, fold=0, seed=42)

    manifest = build_training_manifest(_synthetic_index(), split_config)
    summary = summarize_manifest(manifest, split_config)

    assert summary["class_count"] == 9
    assert summary["split_counts"]["test"] == 9
    assert summary["train_val_group_overlap"] == 0
    assert set(manifest["training_split"]) == {"train", "val", "test"}


def test_identical_content_stays_in_one_training_split() -> None:
    manifest = build_training_manifest(
        _synthetic_index(),
        SplitConfig(n_splits=5, fold=0, seed=7),
    )

    duplicate_rows = manifest[manifest["group_id"] == "train-0-0"]

    assert len(duplicate_rows) == 2
    assert duplicate_rows["training_split"].nunique() == 1


def test_manifest_assignment_is_deterministic() -> None:
    split_config = SplitConfig(n_splits=5, fold=2, seed=21)

    first = build_training_manifest(_synthetic_index(), split_config)
    second = build_training_manifest(_synthetic_index(), split_config)

    assert first.equals(second)
