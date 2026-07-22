"""Train reproducible 9-class skin-lesion classifiers with PyTorch and MLflow."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from training.dataset import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    SkinCancerDataset,
    SplitConfig,
    create_data_loader,
    create_manifest,
    resolve_dataset_root,
)
from training.metrics import compute_classification_metrics, summary_metrics
from training.model import (
    BASELINE_ARCH,
    NUM_CLASSES,
    build_checkpoint,
    create_model,
    set_backbone_trainable,
    write_class_mapping,
)
from training.preprocess import build_eval_transform, build_train_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "training" / "runs"
DEFAULT_MLFLOW_DIR = PROJECT_ROOT / "artifacts" / "mlflow"
DEFAULT_MLFLOW_DATABASE_PATH = DEFAULT_MLFLOW_DIR / "mlflow.db"
DEFAULT_MLFLOW_ARTIFACTS_DIR = DEFAULT_MLFLOW_DIR / "artifacts"
EXPERIMENT_NAME = "skin-cancer-isic-9-class"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one model-training run."""

    data_dir: Path = DEFAULT_DATA_DIR
    index_path: Path = DEFAULT_INDEX_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    runs_dir: Path = DEFAULT_RUNS_DIR
    architecture: str = BASELINE_ARCH
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    workers: int = 4
    seed: int = 42
    n_splits: int = 5
    fold: int = 0
    freeze_epochs: int = 2
    warmup_epochs: int = 1
    patience: int = 5
    gradient_clip: float = 1.0
    pretrained: bool = True
    device: str = "auto"
    amp: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    disable_mlflow: bool = False


@dataclass
class EpochResult:
    """Scalar epoch outputs plus arrays needed for downstream metrics."""

    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts and optional MLflow identity from one completed training run."""

    best_checkpoint: Path
    run_dir: Path
    mlflow_run_id: str | None


def parse_args() -> TrainConfig:
    """Parse command-line options into a validated training configuration."""
    parser = argparse.ArgumentParser(description="Train Skin Cancer ISIC 9-class model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--arch", default=BASELINE_ARCH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--disable-mlflow", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir.expanduser().resolve(),
        index_path=args.index.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        runs_dir=args.runs_dir.expanduser().resolve(),
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        workers=args.workers,
        seed=args.seed,
        n_splits=args.n_splits,
        fold=args.fold,
        freeze_epochs=args.freeze_epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        pretrained=not args.no_pretrained,
        device=args.device,
        amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        run_name=args.run_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        disable_mlflow=args.disable_mlflow,
    )
    validate_config(config, parser)
    return config


def validate_config(config: TrainConfig, parser: argparse.ArgumentParser | None = None) -> None:
    """Validate numeric training options."""
    conditions = {
        "epochs phải > 0": config.epochs > 0,
        "batch-size phải > 0": config.batch_size > 0,
        "learning-rate phải > 0": config.learning_rate > 0,
        "workers phải >= 0": config.workers >= 0,
        "freeze-epochs phải >= 0": config.freeze_epochs >= 0,
        "warmup-epochs phải >= 0": config.warmup_epochs >= 0,
        "patience phải > 0": config.patience > 0,
        "gradient-clip phải > 0": config.gradient_clip > 0,
        "label-smoothing phải trong [0, 1)": 0 <= config.label_smoothing < 1,
    }
    for message, is_valid in conditions.items():
        if is_valid:
            continue
        if parser is not None:
            parser.error(message)
        raise ValueError(message)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/cpu/cuda while failing clearly for unavailable devices."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng torch.cuda.is_available() là False.")
    return device


def prepare_data(
    config: TrainConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, torch.utils.data.DataLoader], dict[str, Any]]:
    """Create the grouped manifest, datasets, and data loaders."""
    split_config = SplitConfig(
        n_splits=config.n_splits,
        fold=config.fold,
        seed=config.seed,
    )
    manifest, manifest_summary = create_manifest(
        index_path=config.index_path,
        output_path=config.manifest_path,
        split_config=split_config,
    )
    dataset_root = resolve_dataset_root(config.data_dir)
    datasets = {
        "train": SkinCancerDataset(
            manifest,
            dataset_root,
            "train",
            build_train_transform(),
        ),
        "val": SkinCancerDataset(
            manifest,
            dataset_root,
            "val",
            build_eval_transform(),
        ),
    }
    loaders = {
        split: create_data_loader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            workers=config.workers,
            seed=config.seed,
            use_cuda=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    return manifest, loaders, manifest_summary


def class_weights(manifest: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Calculate balanced class weights from the training split."""
    counts = (
        manifest.loc[manifest["training_split"] == "train", "class_index"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .sort_index()
    )
    if (counts == 0).any():
        raise ValueError("Training split có class rỗng, không thể tính class weights.")
    training_count = len(manifest[manifest["training_split"] == "train"])
    weights = training_count / (NUM_CLASSES * counts.to_numpy(dtype=np.float32))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    """Create epoch-level linear warmup followed by cosine decay."""

    def learning_rate_multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        decay_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0) / decay_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_clip: float = 1.0,
    max_batches: int | None = None,
    backbone_frozen: bool = False,
) -> EpochResult:
    """Run one training or evaluation epoch and return scalar metrics."""
    is_training = optimizer is not None
    _set_epoch_mode(model, is_training, backbone_frozen)
    total_loss = 0.0
    total_samples = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_size, batch_loss, batch_targets, batch_probabilities = _process_batch(
            model,
            batch,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip=gradient_clip,
        )
        total_loss += batch_loss * batch_size
        total_samples += batch_size
        targets.append(batch_targets)
        probabilities.append(batch_probabilities)

    if not total_samples:
        raise RuntimeError("DataLoader không tạo được batch nào.")
    metric_payload = compute_classification_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
    )
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=summary_metrics(metric_payload),
    )


def _set_epoch_mode(model: nn.Module, is_training: bool, backbone_frozen: bool) -> None:
    """Set model mode without updating frozen backbone batch statistics."""
    if not is_training:
        model.eval()
        return
    model.train()
    if backbone_frozen:
        model.eval()
        model.get_classifier().train()


def _process_batch(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    gradient_clip: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Run forward/backward work for one batch."""
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    is_training = optimizer is not None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_training):
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if optimizer is not None:
            _optimizer_step(loss, model, optimizer, scaler, gradient_clip)

    batch_probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
    return (
        len(targets),
        float(loss.detach().item()),
        targets.detach().cpu().numpy(),
        batch_probabilities,
    )


def _optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
) -> None:
    """Apply one optimizer step with optional mixed-precision scaling."""
    if scaler is None or not scaler.is_enabled():
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        return
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()


def _run_directory(config: TrainConfig) -> Path:
    """Create a unique run directory without overwriting prior artifacts."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = config.run_name or f"{config.architecture.replace('.', '-')}-fold{config.fold}"
    run_dir = config.runs_dir / f"{timestamp}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _config_payload(config: TrainConfig, device: torch.device) -> dict[str, Any]:
    """Convert the dataclass into a JSON- and MLflow-safe payload."""
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif value is None:
            payload[key] = ""
    payload["resolved_device"] = str(device)
    payload["torch_version"] = torch.__version__
    payload["cuda_version"] = torch.version.cuda or ""
    return payload


def default_mlflow_tracking_uri() -> str:
    """Return the supported local SQLite tracking URI for MLflow."""
    return f"sqlite:///{DEFAULT_MLFLOW_DATABASE_PATH.resolve().as_posix()}"


def configure_local_mlflow_experiment() -> None:
    """Create the local experiment with an explicit local artifact location."""
    DEFAULT_MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=DEFAULT_MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def mlflow_run(
    config: TrainConfig,
    config_payload: dict[str, Any],
) -> Iterator[str | None]:
    """Start optional local or remote MLflow tracking and yield its run ID."""
    if config.disable_mlflow:
        yield None
        return
    tracking_uri = config.mlflow_tracking_uri or default_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    if config.mlflow_tracking_uri is None:
        configure_local_mlflow_experiment()
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=config.run_name) as active_run:
        mlflow.log_params(config_payload)
        yield active_run.info.run_id


def train_model(config: TrainConfig) -> Path:
    """Run training and return the best inference checkpoint for CLI callers."""
    return train_model_with_result(config).best_checkpoint


def train_model_with_result(config: TrainConfig) -> TrainingResult:
    """Run training and retain metadata needed by orchestration code."""
    validate_config(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = _run_directory(config)
    config_payload = _config_payload(config, device)
    _write_json(run_dir / "run_config.json", config_payload)
    write_class_mapping(run_dir / "class_to_idx.json")

    manifest, loaders, manifest_summary = prepare_data(config, device)
    _write_json(run_dir / "manifest_summary.json", manifest_summary)
    model = create_model(config.architecture, pretrained=config.pretrained).to(device)
    backbone_frozen = config.freeze_epochs > 0
    set_backbone_trainable(model, trainable=not backbone_frozen)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(manifest, device),
        label_smoothing=config.label_smoothing,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    with mlflow_run(config, config_payload) as run_id:
        best_checkpoint = _training_loop(
            config,
            run_dir,
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            tracking_enabled=run_id is not None,
        )
        if run_id is not None:
            mlflow.log_artifacts(str(run_dir), artifact_path="training_run")
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        run_dir=run_dir,
        mlflow_run_id=run_id,
    )


def _training_loop(
    config: TrainConfig,
    run_dir: Path,
    model: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    tracking_enabled: bool,
) -> Path:
    """Run epochs, early stopping, and checkpoint persistence."""
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_path = run_dir / "best_checkpoint.pth"
    backbone_frozen = config.freeze_epochs > 0

    for epoch in range(config.epochs):
        if backbone_frozen and epoch >= config.freeze_epochs:
            set_backbone_trainable(model, trainable=True)
            backbone_frozen = False
        train_result = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=scaler.is_enabled(),
            gradient_clip=config.gradient_clip,
            max_batches=config.max_train_batches,
            backbone_frozen=backbone_frozen,
        )
        validation_result = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            amp_enabled=scaler.is_enabled(),
            max_batches=config.max_val_batches,
        )
        scheduler.step()
        epoch_record = _epoch_record(epoch, optimizer, train_result, validation_result)
        history.append(epoch_record)
        _write_history(run_dir, history)
        _save_last_checkpoint(
            run_dir,
            model,
            config,
            epoch,
            validation_result.metrics,
            optimizer,
            scheduler,
        )
        _log_epoch(epoch_record, tracking_enabled)
        print(_epoch_message(epoch_record, config.epochs))

        current_macro_f1 = validation_result.metrics["macro_f1"]
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0
            _save_best_checkpoint(
                best_path,
                model,
                config,
                epoch,
                validation_result.metrics,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping tại epoch {epoch + 1}.")
            break

    return best_path


def _epoch_record(
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_result: EpochResult,
    validation_result: EpochResult,
) -> dict[str, Any]:
    """Flatten epoch results for CSV, JSON, and MLflow."""
    record: dict[str, Any] = {
        "epoch": epoch + 1,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": train_result.loss,
        "val_loss": validation_result.loss,
    }
    record.update({f"train_{key}": value for key, value in train_result.metrics.items()})
    record.update({f"val_{key}": value for key, value in validation_result.metrics.items()})
    return record


def _save_best_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Save the best inference checkpoint."""
    checkpoint = build_checkpoint(
        model,
        architecture=config.architecture,
        epoch=epoch + 1,
        metrics=metrics,
    )
    torch.save(checkpoint, checkpoint_path)


def _save_last_checkpoint(
    run_dir: Path,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
) -> None:
    """Save resumable state from the most recent epoch."""
    checkpoint = build_checkpoint(
        model,
        architecture=config.architecture,
        epoch=epoch + 1,
        metrics=metrics,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
    )
    torch.save(checkpoint, run_dir / "last_checkpoint.pth")


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    """Persist epoch history after every epoch."""
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    _write_json(run_dir / "history.json", history)


def _write_json(path: Path, payload: Any) -> None:
    """Write one UTF-8 JSON artifact."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _log_epoch(epoch_record: dict[str, Any], tracking_enabled: bool) -> None:
    """Log scalar epoch metrics when MLflow tracking is enabled."""
    if not tracking_enabled:
        return
    step = int(epoch_record["epoch"])
    metrics = {
        key: float(value)
        for key, value in epoch_record.items()
        if key != "epoch" and isinstance(value, int | float)
    }
    mlflow.log_metrics(metrics, step=step)


def _epoch_message(epoch_record: dict[str, Any], epochs: int) -> str:
    """Build one concise console progress line."""
    return (
        f"Epoch {epoch_record['epoch']:02d}/{epochs:02d} "
        f"train_loss={epoch_record['train_loss']:.4f} "
        f"val_loss={epoch_record['val_loss']:.4f} "
        f"val_macro_f1={epoch_record['val_macro_f1']:.4f}"
    )


def main() -> int:
    """CLI entrypoint."""
    try:
        config = parse_args()
        best_checkpoint = train_model(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi training: {exc}", file=sys.stderr)
        return 1
    print(f"Best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
