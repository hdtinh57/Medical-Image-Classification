"""Manifest creation and PyTorch dataset utilities for model development."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from training.model import CLASS_NAMES, CLASS_TO_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "skin-cancer9-classesisic"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "artifacts" / "eda" / "dataset_index.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "training" / "data_manifest.csv"
EXPECTED_SOURCE_SPLITS = ("Train", "Test")
REQUIRED_INDEX_COLUMNS = {
    "relative_path",
    "split",
    "class_name",
    "is_readable",
    "sha256",
}
MANIFEST_COLUMNS = [
    "relative_path",
    "source_split",
    "training_split",
    "class_name",
    "class_index",
    "sha256",
    "group_id",
]


class DatasetContractError(RuntimeError):
    """Raised when EDA data cannot satisfy the model-development contract."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for one deterministic grouped validation fold."""

    n_splits: int = 5
    fold: int = 0
    seed: int = 42

    def validate(self) -> None:
        """Validate split parameters before invoking scikit-learn."""
        if self.n_splits < 2:
            raise ValueError("n_splits phải lớn hơn hoặc bằng 2.")
        if not 0 <= self.fold < self.n_splits:
            raise ValueError("fold phải nằm trong khoảng [0, n_splits).")


def resolve_dataset_root(data_dir: Path) -> Path:
    """Resolve the outer Kaggle directory or its direct Train/Test dataset root."""
    candidate = data_dir.expanduser().resolve()
    if _contains_source_splits(candidate):
        return candidate
    nested = [path for path in candidate.iterdir() if path.is_dir()] if candidate.is_dir() else []
    matches = [path for path in nested if _contains_source_splits(path)]
    if len(matches) == 1:
        return matches[0]
    if not candidate.exists():
        raise DatasetContractError(f"Không tìm thấy dataset directory: {candidate}")
    raise DatasetContractError(f"Không resolve được Train/Test dataset root từ: {candidate}")


def _contains_source_splits(directory: Path) -> bool:
    """Return whether a directory contains both source split directories."""
    if not directory.is_dir():
        return False
    children = {path.name.casefold() for path in directory.iterdir() if path.is_dir()}
    return all(split.casefold() in children for split in EXPECTED_SOURCE_SPLITS)


def _readable_mask(series: pd.Series) -> pd.Series:
    """Normalize CSV boolean values into one reliable mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.casefold().eq("true")


def load_eda_index(index_path: Path = DEFAULT_INDEX_PATH) -> pd.DataFrame:
    """Load and validate the machine-readable EDA dataset index."""
    if not index_path.is_file():
        raise DatasetContractError(
            f"Không tìm thấy EDA index: {index_path}. Hãy chạy training/eda_skin_cancer.py trước."
        )
    index = pd.read_csv(index_path)
    missing = sorted(REQUIRED_INDEX_COLUMNS.difference(index.columns))
    if missing:
        raise DatasetContractError(f"EDA index thiếu columns: {', '.join(missing)}")

    readable = index.loc[_readable_mask(index["is_readable"])].copy()
    discovered_classes = set(readable["class_name"].unique())
    if discovered_classes != set(CLASS_NAMES):
        raise DatasetContractError("Class names trong EDA index không khớp contract 9 lớp.")
    return readable


def build_training_manifest(
    index: pd.DataFrame,
    split_config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Build Train/Validation/Test assignments while grouping identical content."""
    split_config.validate()
    manifest = _base_manifest(index)
    source_train = manifest[manifest["source_split"] == "Train"].copy()
    source_test = manifest[manifest["source_split"] == "Test"].copy()
    if source_train.empty or source_test.empty:
        raise DatasetContractError("EDA index phải chứa cả Train và Test.")

    train_indices, validation_indices = _grouped_fold_indices(source_train, split_config)
    source_train["training_split"] = "train"
    source_train.loc[source_train.index[validation_indices], "training_split"] = "val"
    source_test["training_split"] = "test"

    result = pd.concat([source_train, source_test], ignore_index=True)
    result = result[MANIFEST_COLUMNS].sort_values(
        ["training_split", "class_index", "relative_path"],
        ignore_index=True,
    )
    validate_manifest(result)
    return result


def _base_manifest(index: pd.DataFrame) -> pd.DataFrame:
    """Convert an EDA index into the portable columns needed for splitting."""
    manifest = index[["relative_path", "split", "class_name", "sha256"]].copy()
    manifest = manifest.rename(columns={"split": "source_split"})
    manifest["class_index"] = manifest["class_name"].map(CLASS_TO_INDEX)
    hashes = manifest["sha256"].fillna("").astype(str)
    manifest["sha256"] = hashes
    manifest["group_id"] = hashes.where(hashes.astype(bool), manifest["relative_path"])
    return manifest


def _grouped_fold_indices(
    source_train: pd.DataFrame,
    split_config: SplitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation row indices for the selected fold."""
    splitter = StratifiedGroupKFold(
        n_splits=split_config.n_splits,
        shuffle=True,
        random_state=split_config.seed,
    )
    folds = splitter.split(
        source_train,
        y=source_train["class_index"],
        groups=source_train["group_id"],
    )
    for fold, indices in enumerate(folds):
        if fold == split_config.fold:
            return indices
    raise DatasetContractError(f"Không tạo được validation fold {split_config.fold}.")


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Validate class coverage and prevent exact-content leakage across Train/Val."""
    missing_columns = sorted(set(MANIFEST_COLUMNS).difference(manifest.columns))
    if missing_columns:
        raise DatasetContractError(f"Manifest thiếu columns: {', '.join(missing_columns)}")

    for split in ("train", "val", "test"):
        classes = set(manifest.loc[manifest["training_split"] == split, "class_name"].unique())
        if classes != set(CLASS_NAMES):
            raise DatasetContractError(f"Split {split} không chứa đủ contract 9 lớp.")

    train_groups = set(manifest.loc[manifest["training_split"] == "train", "group_id"])
    validation_groups = set(manifest.loc[manifest["training_split"] == "val", "group_id"])
    overlap = train_groups.intersection(validation_groups)
    if overlap:
        raise DatasetContractError(f"Train/Val còn {len(overlap)} SHA groups bị overlap.")


def summarize_manifest(manifest: pd.DataFrame, split_config: SplitConfig) -> dict[str, Any]:
    """Build a JSON-safe summary for one split manifest."""
    split_counts = manifest["training_split"].value_counts().sort_index()
    cross_label_groups = manifest.groupby("group_id")["class_name"].nunique()
    train_groups = set(manifest.loc[manifest["training_split"] == "train", "group_id"])
    validation_groups = set(manifest.loc[manifest["training_split"] == "val", "group_id"])
    test_groups = set(manifest.loc[manifest["training_split"] == "test", "group_id"])

    return {
        "total_records": int(len(manifest)),
        "class_count": int(manifest["class_name"].nunique()),
        "split_counts": {str(name): int(count) for name, count in split_counts.items()},
        "split_class_counts": _split_class_counts(manifest),
        "train_val_group_overlap": len(train_groups.intersection(validation_groups)),
        "test_train_group_overlap": len(test_groups.intersection(train_groups)),
        "test_val_group_overlap": len(test_groups.intersection(validation_groups)),
        "multi_label_content_groups": int((cross_label_groups > 1).sum()),
        "n_splits": split_config.n_splits,
        "fold": split_config.fold,
        "seed": split_config.seed,
    }


def _split_class_counts(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return class counts by training split."""
    result: dict[str, dict[str, int]] = {}
    for split, group in manifest.groupby("training_split", sort=True):
        counts = group["class_name"].value_counts().sort_index()
        result[str(split)] = {str(name): int(count) for name, count in counts.items()}
    return result


def write_manifest_artifacts(
    manifest: pd.DataFrame,
    output_path: Path,
    split_config: SplitConfig,
) -> dict[str, Any]:
    """Write the portable manifest and its summary next to each other."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    summary = summarize_manifest(manifest, split_config)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def create_manifest(
    index_path: Path = DEFAULT_INDEX_PATH,
    output_path: Path = DEFAULT_MANIFEST_PATH,
    split_config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create and persist one reproducible model-development manifest."""
    index = load_eda_index(index_path)
    manifest = build_training_manifest(index, split_config)
    summary = write_manifest_artifacts(manifest, output_path, split_config)
    return manifest, summary


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> pd.DataFrame:
    """Load and validate a persisted training manifest."""
    if not manifest_path.is_file():
        raise DatasetContractError(f"Không tìm thấy training manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    return manifest


class SkinCancerDataset(Dataset):
    """Image dataset backed by the portable training manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.records = manifest.loc[manifest["training_split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise DatasetContractError(f"Manifest không có records cho split: {split}")
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        image_path = self.dataset_root / Path(str(row["relative_path"]))
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "target": int(row["class_index"]),
            "relative_path": str(row["relative_path"]),
            "class_name": str(row["class_name"]),
        }


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside one DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    use_cuda: bool,
) -> DataLoader:
    """Create a deterministic DataLoader configured for the selected device."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )


def parse_args() -> argparse.Namespace:
    """Parse manifest-generation CLI arguments."""
    parser = argparse.ArgumentParser(description="Tạo grouped Train/Validation/Test manifest.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Create the manifest from the command line."""
    args = parse_args()
    split_config = SplitConfig(n_splits=args.n_splits, fold=args.fold, seed=args.seed)
    try:
        _, summary = create_manifest(
            index_path=args.index.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            split_config=split_config,
        )
    except (DatasetContractError, OSError, ValueError) as exc:
        print(f"Lỗi manifest: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
