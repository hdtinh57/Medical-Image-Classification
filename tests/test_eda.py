"""Tests for the reproducible EDA pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from training import eda_skin_cancer as eda

DATASET_FOLDER = "Skin cancer ISIC The International Skin Imaging Collaboration"


def _create_image(
    path: Path,
    color: tuple[int, int, int],
    size: tuple[int, int] = (12, 10),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _create_dataset(tmp_path: Path, images_per_class: int = 1) -> tuple[Path, Path]:
    outer = tmp_path / "skin-cancer9-classesisic"
    root = outer / DATASET_FOLDER
    classes = ("class-a", "class-b")

    for split_index, split in enumerate(eda.EXPECTED_SPLITS):
        for class_index, class_name in enumerate(classes):
            for image_index in range(images_per_class):
                color = (
                    20 + split_index * 30,
                    40 + class_index * 40,
                    60 + image_index * 20,
                )
                _create_image(
                    root / split / class_name / f"image-{image_index}.png",
                    color=color,
                )

    return outer, root


def test_resolve_dataset_root_accepts_outer_and_inner_directory(tmp_path: Path) -> None:
    outer, root = _create_dataset(tmp_path)

    assert eda.resolve_dataset_root(outer) == root.resolve()
    assert eda.resolve_dataset_root(root) == root.resolve()


def test_build_dataset_index_marks_corrupted_images(tmp_path: Path) -> None:
    _, root = _create_dataset(tmp_path)
    corrupted = root / "Train" / "class-a" / "corrupted.jpg"
    corrupted.write_bytes(b"not-an-image")

    index = eda.build_dataset_index(root)
    corrupted_row = index[index["relative_path"].str.endswith("corrupted.jpg")].iloc[0]

    assert len(index) == 5
    assert int(index["is_readable"].sum()) == 4
    assert bool(corrupted_row["is_supported"])
    assert not bool(corrupted_row["is_readable"])
    assert corrupted_row["error"]


def test_duplicate_report_detects_cross_split_leakage(tmp_path: Path) -> None:
    _, root = _create_dataset(tmp_path)
    source = root / "Train" / "class-a" / "image-0.png"
    duplicate = root / "Test" / "class-b" / "copied-image.png"
    shutil.copyfile(source, duplicate)

    index = eda.build_dataset_index(root)
    duplicates = eda.build_duplicate_report(index)
    cross_split = duplicates[duplicates["scope"] == "cross_split"]

    assert len(cross_split) == 1
    assert int(cross_split.iloc[0]["occurrence_count"]) == 2
    assert cross_split.iloc[0]["splits"] == "Test|Train"
    assert int(cross_split.iloc[0]["class_count"]) == 2
    assert bool(cross_split.iloc[0]["has_multiple_folder_labels"])
    assert any("nhiều thư mục nhãn" in warning for warning in eda.build_warnings(index, duplicates))


def test_class_sampling_is_deterministic(tmp_path: Path) -> None:
    _, root = _create_dataset(tmp_path, images_per_class=4)
    index = eda.build_dataset_index(root)

    first = eda.sample_images_by_class(index, count=2, seed=42)
    second = eda.sample_images_by_class(index, count=2, seed=42)

    assert first["relative_path"].tolist() == second["relative_path"].tolist()
    assert first["class_name"].value_counts().to_dict() == {"class-a": 2, "class-b": 2}


def test_run_eda_writes_required_artifacts(tmp_path: Path) -> None:
    outer, _ = _create_dataset(tmp_path, images_per_class=2)
    output_dir = tmp_path / "artifacts" / "eda"
    config = eda.EdaConfig(
        data_dir=outer,
        output_dir=output_dir,
        pixel_sample_per_class=1,
        seed=7,
    )

    summary = eda.run_eda(config)

    required_artifacts = {
        "dataset_index.csv",
        "summary.json",
        "class_distribution.png",
        "split_class_distribution.png",
        "sample_images.png",
        "image_size_distribution.png",
        "aspect_ratio_distribution.png",
        "color_brightness_by_class.png",
        "file_size_distribution.png",
    }
    generated = {path.name for path in output_dir.iterdir()}

    assert summary["readable_images"] == 8
    assert summary["class_count"] == 2
    assert summary["split_counts"] == {"Test": 4, "Train": 4}
    assert summary["duplicate_groups"] == 0
    assert summary["same_content_multi_folder_label_groups"] == 0
    assert required_artifacts <= generated
    assert all((output_dir / name).stat().st_size > 0 for name in required_artifacts)
