"""Run reproducible exploratory data analysis for the ISIC skin-cancer dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "skin-cancer9-classesisic"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"
EXPECTED_SPLITS = ("Train", "Test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_PIXEL_SAMPLE_PER_CLASS = 30
DEFAULT_SAMPLE_IMAGES_PER_CLASS = 3
DEFAULT_SEED = 42
MIN_RELIABLE_CLASS_COUNT = 10

INDEX_COLUMNS = [
    "filepath",
    "relative_path",
    "split",
    "class_name",
    "extension",
    "file_size_bytes",
    "is_supported",
    "is_readable",
    "error",
    "width",
    "height",
    "aspect_ratio",
    "image_mode",
    "image_format",
    "sha256",
]

DUPLICATE_COLUMNS = [
    "sha256",
    "occurrence_count",
    "scope",
    "splits",
    "classes",
    "class_count",
    "has_multiple_folder_labels",
    "paths",
]


class EdaError(RuntimeError):
    """Raised when the dataset cannot be analyzed safely."""


@dataclass(frozen=True)
class EdaConfig:
    """Runtime configuration for one EDA execution."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    pixel_sample_per_class: int = DEFAULT_PIXEL_SAMPLE_PER_CLASS
    seed: int = DEFAULT_SEED
    show: bool = False


def configure_console_encoding() -> None:
    """Use UTF-8 for Vietnamese console messages when the stream supports it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            continue


def parse_args() -> EdaConfig:
    """Parse command-line arguments into an immutable EDA configuration."""
    parser = argparse.ArgumentParser(
        description="Phân tích dữ liệu Skin Cancer ISIC 9 Classes và lưu artifacts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục dataset (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu kết quả EDA (mặc định: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pixel-sample-per-class",
        type=int,
        default=DEFAULT_PIXEL_SAMPLE_PER_CLASS,
        help="Số ảnh mỗi lớp dùng cho thống kê pixel (mặc định: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed cho sampling tái lập (mặc định: 42).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị biểu đồ tương tác sau khi lưu.",
    )
    args = parser.parse_args()

    if args.pixel_sample_per_class < 1:
        parser.error("--pixel-sample-per-class phải lớn hơn 0")

    return EdaConfig(
        data_dir=args.data_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        pixel_sample_per_class=args.pixel_sample_per_class,
        seed=args.seed,
        show=args.show,
    )


def _split_directories(directory: Path) -> dict[str, Path]:
    """Return expected split directories keyed by canonical split name."""
    if not directory.is_dir():
        return {}

    children = {path.name.casefold(): path for path in directory.iterdir() if path.is_dir()}
    return {
        split: children[split.casefold()]
        for split in EXPECTED_SPLITS
        if split.casefold() in children
    }


def resolve_dataset_root(data_dir: Path) -> Path:
    """Resolve either an outer download directory or the directory containing splits."""
    candidate = data_dir.expanduser().resolve()
    if len(_split_directories(candidate)) == len(EXPECTED_SPLITS):
        return candidate

    nested_candidates = (
        [
            child
            for child in sorted(candidate.iterdir())
            if child.is_dir() and len(_split_directories(child)) == len(EXPECTED_SPLITS)
        ]
        if candidate.is_dir()
        else []
    )

    if len(nested_candidates) == 1:
        return nested_candidates[0]

    expected = " và ".join(EXPECTED_SPLITS)
    if not candidate.exists():
        raise EdaError(f"Không tìm thấy thư mục dataset: {candidate}")
    if len(nested_candidates) > 1:
        raise EdaError(
            f"Tìm thấy nhiều dataset root trong {candidate}; hãy truyền --data-dir cụ thể."
        )
    raise EdaError(f"{candidate} không chứa đầy đủ thư mục {expected}.")


def _file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate an exact content hash without loading the whole file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_record(
    file_path: Path,
    dataset_root: Path,
    split: str,
    class_name: str,
) -> dict[str, Any]:
    """Build path and file metadata shared by all image inspection outcomes."""
    file_size = file_path.stat().st_size
    return {
        "filepath": str(file_path.resolve()),
        "relative_path": file_path.relative_to(dataset_root).as_posix(),
        "split": split,
        "class_name": class_name,
        "extension": file_path.suffix.lower(),
        "file_size_bytes": file_size,
        "is_supported": file_path.suffix.lower() in IMAGE_EXTENSIONS,
        "is_readable": False,
        "error": "",
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "image_mode": "",
        "image_format": "",
        "sha256": "",
    }


def inspect_image_file(
    file_path: Path,
    dataset_root: Path,
    split: str,
    class_name: str,
) -> dict[str, Any]:
    """Inspect one file for integrity, dimensions, color mode, and exact hash."""
    record = _base_record(file_path, dataset_root, split, class_name)
    if record["file_size_bytes"] == 0:
        record["error"] = "empty_file"
        return record
    if not record["is_supported"]:
        record["error"] = "unsupported_extension"
        return record

    try:
        record["sha256"] = _file_sha256(file_path)
        with Image.open(file_path) as image:
            width, height = image.size
            image_mode = image.mode
            image_format = image.format or ""
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record.update(
        {
            "is_readable": True,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else np.nan,
            "image_mode": image_mode,
            "image_format": image_format,
        }
    )
    return record


def _class_directories(split_dir: Path) -> list[Path]:
    """Return sorted class directories for one split."""
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def build_dataset_index(dataset_root: Path) -> pd.DataFrame:
    """Inspect all files under split/class directories and return one dataset index."""
    records: list[dict[str, Any]] = []
    split_dirs = _split_directories(dataset_root)

    for split in EXPECTED_SPLITS:
        split_dir = split_dirs.get(split)
        if split_dir is None:
            raise EdaError(f"Thiếu split bắt buộc: {split}")
        for class_dir in _class_directories(split_dir):
            files = sorted(path for path in class_dir.rglob("*") if path.is_file())
            records.extend(
                inspect_image_file(file_path, dataset_root, split, class_dir.name)
                for file_path in files
            )

    if not records:
        raise EdaError(f"Không tìm thấy file nào trong dataset: {dataset_root}")
    return pd.DataFrame.from_records(records, columns=INDEX_COLUMNS)


def build_duplicate_report(index: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact same-content groups and report source-split overlap."""
    hashed = index.loc[index["sha256"].astype(bool)].copy()
    duplicated = hashed[hashed.duplicated("sha256", keep=False)]
    records: list[dict[str, Any]] = []

    for sha256, group in duplicated.groupby("sha256", sort=True):
        splits = sorted(group["split"].unique().tolist())
        classes = sorted(group["class_name"].unique().tolist())
        records.append(
            {
                "sha256": sha256,
                "occurrence_count": int(len(group)),
                "scope": "cross_split" if len(splits) > 1 else "within_split",
                "splits": "|".join(splits),
                "classes": "|".join(classes),
                "class_count": len(classes),
                "has_multiple_folder_labels": len(classes) > 1,
                "paths": "|".join(sorted(group["relative_path"].tolist())),
            }
        )

    return pd.DataFrame.from_records(records, columns=DUPLICATE_COLUMNS)


def sample_images_by_class(index: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Return a deterministic, class-stratified sample of readable images."""
    readable = index[index["is_readable"]].copy()
    samples = [
        group.sample(n=min(count, len(group)), random_state=seed)
        for _, group in readable.groupby("class_name", sort=True)
    ]
    if not samples:
        return readable.head(0)
    return pd.concat(samples, ignore_index=True).sort_values(
        ["class_name", "relative_path"], ignore_index=True
    )


def compute_pixel_statistics(index: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Compute color and brightness statistics on a stratified sample."""
    records: list[dict[str, Any]] = []
    sampled = sample_images_by_class(index, count, seed)

    for row in sampled.itertuples(index=False):
        try:
            with Image.open(row.filepath) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError, ValueError):
            continue

        channel_means = pixels.mean(axis=(0, 1))
        records.append(
            {
                "relative_path": row.relative_path,
                "split": row.split,
                "class_name": row.class_name,
                "mean_r": float(channel_means[0]),
                "mean_g": float(channel_means[1]),
                "mean_b": float(channel_means[2]),
                "brightness": float(pixels.mean()),
                "contrast": float(pixels.std()),
            }
        )

    return pd.DataFrame.from_records(records)


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    """Return JSON-safe descriptive statistics for one numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
    }


def _class_counts(index: pd.DataFrame) -> dict[str, int]:
    """Return readable image counts by class."""
    counts = index[index["is_readable"]]["class_name"].value_counts().sort_index()
    return {str(name): int(count) for name, count in counts.items()}


def _split_class_counts(index: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return readable image counts by split and class."""
    readable = index[index["is_readable"]]
    result: dict[str, dict[str, int]] = {}
    for split, group in readable.groupby("split", sort=True):
        counts = group["class_name"].value_counts().sort_index()
        result[str(split)] = {str(name): int(count) for name, count in counts.items()}
    return result


def _class_sets_by_split(index: pd.DataFrame) -> dict[str, set[str]]:
    """Return discovered class names for each expected split."""
    return {
        split: set(index.loc[index["split"] == split, "class_name"].unique())
        for split in EXPECTED_SPLITS
    }


def build_warnings(index: pd.DataFrame, duplicates: pd.DataFrame) -> list[str]:
    """Build actionable warnings from data-quality and distribution checks."""
    warnings: list[str] = []
    unreadable_count = int((~index["is_readable"] & index["is_supported"]).sum())
    unsupported_count = int((~index["is_supported"]).sum())
    cross_split_count = int((duplicates["scope"] == "cross_split").sum())
    multi_folder_label_count = int(duplicates["has_multiple_folder_labels"].sum())

    if unreadable_count:
        warnings.append(f"Có {unreadable_count} ảnh hỗ trợ nhưng không đọc được.")
    if unsupported_count:
        warnings.append(f"Có {unsupported_count} file có extension không được hỗ trợ.")
    if cross_split_count:
        warnings.append(
            f"Có {cross_split_count} nhóm cùng nội dung xuất hiện ở cả raw Train và Test; "
            "benchmark được giữ nguyên."
        )
    if multi_folder_label_count:
        warnings.append(
            f"Có {multi_folder_label_count} nhóm cùng nội dung xuất hiện trong nhiều thư mục nhãn; "
            "EDA chỉ ghi nhận, không thay đổi raw benchmark."
        )

    class_sets = _class_sets_by_split(index)
    if class_sets["Train"] != class_sets["Test"]:
        warnings.append("Danh sách lớp giữa Train và Test không đồng nhất.")

    warnings.extend(_low_count_warnings(index))
    imbalance_ratio = _imbalance_ratio(index)
    if imbalance_ratio > 3:
        warnings.append(f"Dataset mất cân bằng lớp đáng kể ({imbalance_ratio:.2f}x).")
    warnings.append("Dataset chưa có validation split riêng.")
    return warnings


def _low_count_warnings(index: pd.DataFrame) -> list[str]:
    """Return warnings for classes with too few readable examples in a split."""
    readable = index[index["is_readable"]]
    counts = readable.groupby(["split", "class_name"]).size()
    return [
        f"{split}/{class_name} chỉ có {int(count)} ảnh; metric sẽ thiếu ổn định."
        for (split, class_name), count in counts.items()
        if count < MIN_RELIABLE_CLASS_COUNT
    ]


def _imbalance_ratio(index: pd.DataFrame) -> float:
    """Calculate the largest-to-smallest readable class count ratio."""
    counts = index[index["is_readable"]]["class_name"].value_counts()
    if counts.empty or counts.min() == 0:
        return 0.0
    return float(counts.max() / counts.min())


def build_summary(
    index: pd.DataFrame,
    pixel_statistics: pd.DataFrame,
    duplicates: pd.DataFrame,
    config: EdaConfig,
    dataset_root: Path,
) -> dict[str, Any]:
    """Build one machine-readable EDA summary."""
    readable = index[index["is_readable"]]
    split_counts = readable["split"].value_counts().sort_index()
    duplicate_scope_counts = duplicates["scope"].value_counts()
    duplicate_occurrences = int(duplicates["occurrence_count"].sum())

    return {
        "dataset_root": str(dataset_root),
        "total_files": int(len(index)),
        "supported_image_files": int(index["is_supported"].sum()),
        "readable_images": int(index["is_readable"].sum()),
        "unreadable_images": int((~index["is_readable"] & index["is_supported"]).sum()),
        "unsupported_files": int((~index["is_supported"]).sum()),
        "class_count": int(readable["class_name"].nunique()),
        "split_counts": {str(name): int(count) for name, count in split_counts.items()},
        "class_counts": _class_counts(index),
        "split_class_counts": _split_class_counts(index),
        "imbalance_ratio": _imbalance_ratio(index),
        "duplicate_groups": int(len(duplicates)),
        "duplicate_image_occurrences": duplicate_occurrences,
        "excess_duplicate_copies": duplicate_occurrences - int(len(duplicates)),
        "within_split_duplicate_groups": int(duplicate_scope_counts.get("within_split", 0)),
        "cross_split_duplicate_groups": int(duplicate_scope_counts.get("cross_split", 0)),
        "same_content_multi_folder_label_groups": int(
            duplicates["has_multiple_folder_labels"].sum()
        ),
        "image_statistics": {
            "width": _numeric_summary(readable["width"]),
            "height": _numeric_summary(readable["height"]),
            "aspect_ratio": _numeric_summary(readable["aspect_ratio"]),
            "file_size_bytes": _numeric_summary(readable["file_size_bytes"]),
        },
        "pixel_statistics": {
            "sample_count": int(len(pixel_statistics)),
            "sample_per_class": config.pixel_sample_per_class,
            "brightness": _numeric_summary(
                pixel_statistics.get("brightness", pd.Series(dtype=float))
            ),
            "contrast": _numeric_summary(pixel_statistics.get("contrast", pd.Series(dtype=float))),
        },
        "seed": config.seed,
        "warnings": build_warnings(index, duplicates),
    }


def _finalize_figure(fig: plt.Figure, output_path: Path, show: bool) -> None:
    """Save a figure and release memory unless interactive display was requested."""
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if not show:
        plt.close(fig)


def plot_class_distribution(index: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot overall readable image counts by class."""
    counts = (
        index[index["is_readable"]]["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=counts,
        x="count",
        y="class_name",
        hue="class_name",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set(title="Class distribution", xlabel="Image count", ylabel="Class")
    _finalize_figure(fig, output_dir / "class_distribution.png", show)


def plot_split_class_distribution(index: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot stacked readable image counts by split and class."""
    readable = index[index["is_readable"]]
    table = pd.crosstab(readable["class_name"], readable["split"])
    table = table.sort_values(table.columns.tolist(), ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    table.plot(kind="barh", stacked=True, ax=ax, colormap="Set2")
    ax.set(title="Train/Test distribution by class", xlabel="Image count", ylabel="Class")
    ax.legend(title="Split")
    _finalize_figure(fig, output_dir / "split_class_distribution.png", show)


def plot_sample_images(
    index: pd.DataFrame,
    output_dir: Path,
    seed: int,
    show: bool,
    images_per_class: int = DEFAULT_SAMPLE_IMAGES_PER_CLASS,
) -> None:
    """Plot a deterministic sample grid for every class."""
    sampled = sample_images_by_class(index, images_per_class, seed)
    classes = sorted(sampled["class_name"].unique())
    fig, axes = plt.subplots(
        len(classes),
        images_per_class,
        figsize=(images_per_class * 3, max(len(classes), 1) * 2.7),
        squeeze=False,
    )

    for row_index, class_name in enumerate(classes):
        class_sample = sampled[sampled["class_name"] == class_name]
        _draw_class_samples(axes[row_index], class_sample, class_name)

    fig.suptitle("Deterministic sample images by class", fontsize=14, y=1.01)
    _finalize_figure(fig, output_dir / "sample_images.png", show)


def _draw_class_samples(
    axes: np.ndarray,
    class_sample: pd.DataFrame,
    class_name: str,
) -> None:
    """Draw one class row in the sample-image grid."""
    rows = list(class_sample.itertuples(index=False))
    for column_index, axis in enumerate(axes):
        axis.axis("off")
        if column_index >= len(rows):
            continue
        with Image.open(rows[column_index].filepath) as image:
            axis.imshow(image.convert("RGB"))
        if column_index == 0:
            axis.set_title(class_name, loc="left", fontsize=10)


def plot_image_sizes(index: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot image width, height, and their relationship."""
    readable = index[index["is_readable"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(readable["width"], bins=30, color="steelblue", ax=axes[0])
    sns.histplot(readable["height"], bins=30, color="salmon", ax=axes[1])
    sns.scatterplot(
        data=readable,
        x="width",
        y="height",
        hue="class_name",
        legend=False,
        alpha=0.55,
        ax=axes[2],
    )
    axes[0].set_title("Width distribution")
    axes[1].set_title("Height distribution")
    axes[2].set_title("Width vs height")
    _finalize_figure(fig, output_dir / "image_size_distribution.png", show)


def plot_aspect_ratios(index: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot readable image aspect ratios by class."""
    readable = index[index["is_readable"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=readable, x="class_name", y="aspect_ratio", ax=ax)
    ax.set(title="Aspect ratio by class", xlabel="Class", ylabel="Width / height")
    ax.tick_params(axis="x", rotation=65)
    _finalize_figure(fig, output_dir / "aspect_ratio_distribution.png", show)


def plot_color_brightness(
    pixel_statistics: pd.DataFrame,
    output_dir: Path,
    show: bool,
) -> None:
    """Plot brightness and average RGB channel values by class."""
    if pixel_statistics.empty:
        return

    melted = pixel_statistics.melt(
        id_vars=["class_name"],
        value_vars=["mean_r", "mean_g", "mean_b"],
        var_name="channel",
        value_name="mean_value",
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.boxplot(data=pixel_statistics, x="class_name", y="brightness", ax=axes[0])
    sns.boxplot(
        data=melted,
        x="class_name",
        y="mean_value",
        hue="channel",
        ax=axes[1],
    )
    axes[0].set_title("Brightness by class")
    axes[1].set_title("Mean RGB channels by class")
    for axis in axes:
        axis.tick_params(axis="x", rotation=65)
    _finalize_figure(fig, output_dir / "color_brightness_by_class.png", show)


def plot_file_sizes(index: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot readable image file sizes in KiB."""
    readable = index[index["is_readable"]].copy()
    readable["file_size_kib"] = readable["file_size_bytes"] / 1024
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(readable["file_size_kib"], bins=40, color="teal", ax=ax)
    ax.set(title="Image file-size distribution", xlabel="File size (KiB)", ylabel="Count")
    _finalize_figure(fig, output_dir / "file_size_distribution.png", show)


def generate_visualizations(
    index: pd.DataFrame,
    pixel_statistics: pd.DataFrame,
    config: EdaConfig,
) -> None:
    """Generate all EDA figures in a deterministic order."""
    plot_class_distribution(index, config.output_dir, config.show)
    plot_split_class_distribution(index, config.output_dir, config.show)
    plot_sample_images(index, config.output_dir, config.seed, config.show)
    plot_image_sizes(index, config.output_dir, config.show)
    plot_aspect_ratios(index, config.output_dir, config.show)
    plot_color_brightness(pixel_statistics, config.output_dir, config.show)
    plot_file_sizes(index, config.output_dir, config.show)


def write_artifacts(
    index: pd.DataFrame,
    duplicates: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist the dataset index, summary, and optional duplicate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "dataset_index.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    duplicate_path = output_dir / "duplicate_report.csv"
    if duplicates.empty:
        duplicate_path.unlink(missing_ok=True)
        return
    duplicates.to_csv(duplicate_path, index=False)


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Print the concise result a CLI user needs after one EDA run."""
    print("\n" + "=" * 68)
    print("TÓM TẮT EDA — SKIN CANCER ISIC 9 CLASSES")
    print("=" * 68)
    print(f"Ảnh đọc được : {summary['readable_images']:,}")
    print(f"Số lớp       : {summary['class_count']}")
    print(f"Train / Test : {summary['split_counts']}")
    print(f"Imbalance    : {summary['imbalance_ratio']:.2f}x")
    print(f"Duplicate    : {summary['duplicate_groups']} nhóm")
    print(f"Cross-split  : {summary['cross_split_duplicate_groups']} nhóm")
    print(
        f"Same-content/nhiều thư mục nhãn: {summary['same_content_multi_folder_label_groups']} nhóm"
    )
    print(f"Output       : {output_dir}")

    warnings = summary["warnings"]
    if warnings:
        print("\nCảnh báo:")
        for warning in warnings:
            print(f"- {warning}")


def run_eda(config: EdaConfig) -> dict[str, Any]:
    """Run the complete EDA pipeline and return its summary."""
    dataset_root = resolve_dataset_root(config.data_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {dataset_root}")
    print("Đang kiểm tra toàn bộ file ảnh...")
    index = build_dataset_index(dataset_root)
    duplicates = build_duplicate_report(index)

    print("Đang tính thống kê pixel theo từng lớp...")
    pixel_statistics = compute_pixel_statistics(
        index,
        count=config.pixel_sample_per_class,
        seed=config.seed,
    )
    summary = build_summary(index, pixel_statistics, duplicates, config, dataset_root)

    write_artifacts(index, duplicates, summary, config.output_dir)
    print("Đang sinh biểu đồ...")
    generate_visualizations(index, pixel_statistics, config)

    if config.show:
        plt.show()
    plt.close("all")
    print_summary(summary, config.output_dir)
    return summary


def main() -> int:
    """CLI entrypoint."""
    configure_console_encoding()
    config = parse_args()
    try:
        run_eda(config)
    except (EdaError, OSError) as exc:
        print(f"Lỗi EDA: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
