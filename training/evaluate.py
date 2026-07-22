"""Evaluate trained checkpoints and persist reproducible classification reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from training.dataset import (
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    SkinCancerDataset,
    create_data_loader,
    load_manifest,
    resolve_dataset_root,
)
from training.metrics import compute_classification_metrics, summary_metrics
from training.model import CLASS_NAMES, NUM_CLASSES, load_model_from_checkpoint
from training.preprocess import build_eval_transform


@dataclass(frozen=True)
class EvaluateConfig:
    """Configuration for one checkpoint evaluation."""

    checkpoint: Path
    data_dir: Path = DEFAULT_DATA_DIR
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    output_dir: Path | None = None
    split: str = "test"
    batch_size: int = 32
    workers: int = 4
    device: str = "auto"
    amp: bool = True


def parse_args() -> EvaluateConfig:
    """Parse evaluation CLI options."""
    parser = argparse.ArgumentParser(description="Evaluate a 9-class skin classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    return EvaluateConfig(
        checkpoint=args.checkpoint.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        amp=not args.no_amp,
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve an evaluation device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    return device


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect targets, probabilities, and paths for one dataset split."""
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    relative_paths: list[str] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                logits = model(images)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(batch["target"].numpy())
            relative_paths.extend(batch["relative_path"])

    return np.concatenate(targets), np.concatenate(probabilities), relative_paths


def write_evaluation_artifacts(
    output_dir: Path,
    targets: np.ndarray,
    probabilities: np.ndarray,
    relative_paths: list[str],
    metrics: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> None:
    """Write JSON, CSV, and confusion-matrix artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": {
            "architecture": checkpoint_metadata["architecture"],
            "epoch": int(checkpoint_metadata.get("epoch", 0)),
            "validation_metrics": checkpoint_metadata.get("metrics", {}),
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _prediction_frame(targets, probabilities, relative_paths).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(metrics["per_class"]).transpose().to_csv(output_dir / "classification_report.csv")
    matrix = confusion_matrix(targets, probabilities.argmax(axis=1), labels=range(NUM_CLASSES))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")


def _prediction_frame(
    targets: np.ndarray,
    probabilities: np.ndarray,
    relative_paths: list[str],
) -> pd.DataFrame:
    """Build one row per image with ground truth and top prediction."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    frame = pd.DataFrame(
        {
            "relative_path": relative_paths,
            "target_index": targets,
            "target_class": [CLASS_NAMES[index] for index in targets],
            "predicted_index": predictions,
            "predicted_class": [CLASS_NAMES[index] for index in predictions],
            "confidence": confidence,
        }
    )
    for class_index, class_name in enumerate(CLASS_NAMES):
        frame[f"probability_{class_name}"] = probabilities[:, class_index]
    return frame


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Render a readable sequential-color confusion matrix for nine classes."""
    fig, axis = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Image count"},
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(
        title="Confusion matrix — Skin Cancer ISIC 9 Classes",
        xlabel="Predicted class",
        ylabel="True class",
    )
    axis.tick_params(axis="x", rotation=55)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(config: EvaluateConfig) -> tuple[dict[str, Any], Path]:
    """Evaluate one checkpoint and return its metrics and output directory."""
    device = resolve_device(config.device)
    manifest = load_manifest(config.manifest_path)
    dataset_root = resolve_dataset_root(config.data_dir)
    dataset = SkinCancerDataset(
        manifest,
        dataset_root,
        config.split,
        build_eval_transform(),
    )
    loader = create_data_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        workers=config.workers,
        seed=42,
        use_cuda=device.type == "cuda",
    )
    model, checkpoint = load_model_from_checkpoint(config.checkpoint, map_location="cpu")
    model = model.to(device)
    targets, probabilities, relative_paths = collect_predictions(
        model,
        loader,
        device,
        amp_enabled=config.amp and device.type == "cuda",
    )
    metrics = compute_classification_metrics(targets, probabilities)
    output_dir = config.output_dir or config.checkpoint.parent / f"evaluation-{config.split}"
    write_evaluation_artifacts(
        output_dir,
        targets,
        probabilities,
        relative_paths,
        metrics,
        checkpoint,
    )
    return metrics, output_dir


def main() -> int:
    """CLI entrypoint."""
    try:
        config = parse_args()
        metrics, output_dir = evaluate_checkpoint(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi evaluation: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary_metrics(metrics), ensure_ascii=False, indent=2))
    print(f"Evaluation artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
