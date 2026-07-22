"""Run the reproducible training-to-candidate lifecycle on a trusted runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.dataset import DEFAULT_MANIFEST_PATH
from training.eda_skin_cancer import DEFAULT_OUTPUT_DIR as DEFAULT_EDA_OUTPUT_DIR
from training.eda_skin_cancer import EdaConfig, run_eda
from training.evaluate import EvaluateConfig, evaluate_checkpoint
from training.ingest import DEFAULT_OUTPUT_DIR as DEFAULT_DATA_DIR
from training.ingest import download_dataset
from training.quality_gate import (
    GateDecision,
    GateThresholds,
    evaluate_quality_gate,
    write_decision,
)
from training.train import (
    BASELINE_ARCH,
    DEFAULT_MLFLOW_DATABASE_PATH,
    DEFAULT_RUNS_DIR,
    TrainConfig,
    TrainingResult,
    train_model_with_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "pipeline" / "candidate"
BUNDLE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one candidate-training pipeline execution."""

    version: int
    data_dir: Path = DEFAULT_DATA_DIR
    eda_output_dir: Path = DEFAULT_EDA_OUTPUT_DIR
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR
    runs_dir: Path = DEFAULT_RUNS_DIR
    manifest_path: Path = DEFAULT_MANIFEST_PATH
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
    device: str = "auto"
    amp: bool = True
    pretrained: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    mlflow_tracking_uri: str | None = None
    disable_mlflow: bool = False
    skip_ingest: bool = False
    skip_eda: bool = False
    thresholds: GateThresholds = GateThresholds()

    def validate(self) -> None:
        """Validate pipeline-only invariants before mutating artifact directories."""
        if self.version < 1:
            raise ValueError("--version phải là số nguyên dương.")
        self.thresholds.validate()
        if self.skip_eda and not self.index_path.is_file():
            raise ValueError("--skip-eda yêu cầu dataset_index.csv đã tồn tại.")

    @property
    def index_path(self) -> Path:
        """Return the EDA index generated in the configured artifact directory."""
        return self.eda_output_dir / "dataset_index.csv"

    @property
    def bundle_dir(self) -> Path:
        """Return the immutable local artifact directory for this candidate version."""
        return self.candidate_dir / f"v{self.version}"


@dataclass(frozen=True)
class PipelineResult:
    """Outcome and artifacts from a candidate-training execution."""

    bundle_dir: Path
    training: TrainingResult
    decision: GateDecision
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Build a candidate bundle and reject it when Validation quality is insufficient."""
    config.validate()
    _prepare_bundle_directory(config.bundle_dir)
    if not config.skip_ingest:
        download_dataset(config.data_dir)
    eda_summary = _run_eda_if_needed(config)
    training = train_model_with_result(_train_config(config))
    validation_metrics, _ = _evaluate(training, config, "val")
    test_metrics, _ = _evaluate(training, config, "test")
    decision = evaluate_quality_gate(validation_metrics, config.thresholds)
    gate_path = config.bundle_dir / "quality_gate.json"
    write_decision(decision, gate_path)
    _write_bundle(
        config,
        training,
        decision,
        eda_summary,
    )
    _log_pipeline_artifacts(config, training, decision)
    return PipelineResult(
        bundle_dir=config.bundle_dir,
        training=training,
        decision=decision,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def _prepare_bundle_directory(bundle_dir: Path) -> None:
    """Create one empty versioned bundle directory without overwriting evidence."""
    if bundle_dir.exists():
        raise FileExistsError(f"Candidate bundle đã tồn tại: {bundle_dir}")
    bundle_dir.mkdir(parents=True)


def _run_eda_if_needed(config: PipelineConfig) -> dict[str, Any]:
    """Run EDA unless a caller deliberately reuses an existing index."""
    if config.skip_eda:
        return _read_json(config.eda_output_dir / "summary.json")
    return run_eda(EdaConfig(data_dir=config.data_dir, output_dir=config.eda_output_dir))


def _train_config(config: PipelineConfig) -> TrainConfig:
    """Translate pipeline options to the reusable training configuration."""
    return TrainConfig(
        data_dir=config.data_dir,
        index_path=config.index_path,
        manifest_path=config.manifest_path,
        runs_dir=config.runs_dir,
        architecture=config.architecture,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        workers=config.workers,
        seed=config.seed,
        n_splits=config.n_splits,
        fold=config.fold,
        freeze_epochs=config.freeze_epochs,
        warmup_epochs=config.warmup_epochs,
        patience=config.patience,
        gradient_clip=config.gradient_clip,
        pretrained=config.pretrained,
        device=config.device,
        amp=config.amp,
        max_train_batches=config.max_train_batches,
        max_val_batches=config.max_val_batches,
        run_name=f"candidate-v{config.version}",
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        disable_mlflow=config.disable_mlflow,
    )


def _evaluate(
    training: TrainingResult,
    config: PipelineConfig,
    split: str,
) -> tuple[dict[str, Any], Path]:
    """Evaluate the selected checkpoint without allowing Test to gate promotion."""
    output_dir = config.bundle_dir / "evaluations" / split
    return evaluate_checkpoint(
        EvaluateConfig(
            checkpoint=training.best_checkpoint,
            data_dir=config.data_dir,
            manifest_path=config.manifest_path,
            output_dir=output_dir,
            split=split,
            batch_size=config.batch_size,
            workers=config.workers,
            device=config.device,
            amp=config.amp,
        )
    )


def _write_bundle(
    config: PipelineConfig,
    training: TrainingResult,
    decision: GateDecision,
    eda_summary: dict[str, Any],
) -> None:
    """Copy only reproducible model evidence into the versioned candidate bundle."""
    checkpoint_path = config.bundle_dir / "best_checkpoint.pth"
    manifest_summary_path = config.bundle_dir / "manifest_summary.json"
    candidate_metadata_path = config.bundle_dir / "candidate.json"
    shutil.copy2(training.best_checkpoint, checkpoint_path)
    _copy_file(training.run_dir / "manifest_summary.json", manifest_summary_path)
    _copy_file(config.index_path, config.bundle_dir / "dataset_index.csv")
    _write_json(config.bundle_dir / "eda_summary.json", eda_summary)
    _write_json(candidate_metadata_path, _candidate_metadata(config, training, decision))
    _write_json(
        config.bundle_dir / "pipeline_summary.json",
        {
            "format_version": BUNDLE_FORMAT_VERSION,
            "status": "passed" if decision.passed else "rejected",
            "version": config.version,
            "validation_metrics_path": "evaluations/val/metrics.json",
            "test_metrics_path": "evaluations/test/metrics.json",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_json(config.bundle_dir / "checksums.json", _bundle_checksums(config.bundle_dir))


def _copy_file(source: Path, destination: Path) -> None:
    """Copy a required evidence artifact while failing with a clear missing path."""
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def _candidate_metadata(
    config: PipelineConfig,
    training: TrainingResult,
    decision: GateDecision,
) -> dict[str, Any]:
    """Build provenance metadata used by release verification and deployment."""
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_version": config.version,
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "mlflow_run_id": training.mlflow_run_id,
        "checkpoint_sha256": _sha256(training.best_checkpoint),
        "validation_metrics": decision.candidate_metrics,
        "quality_gate_passed": decision.passed,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _bundle_checksums(bundle_dir: Path) -> dict[str, str]:
    """Hash all persisted bundle files except the checksum manifest itself."""
    files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
    return {
        path.relative_to(bundle_dir).as_posix(): _sha256(path)
        for path in files
        if path.name != "checksums.json"
    }


def _sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 for a model or bundle artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one expected JSON evidence artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact không phải dictionary: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 machine-readable pipeline artifact."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _log_pipeline_artifacts(
    config: PipelineConfig,
    training: TrainingResult,
    decision: GateDecision,
) -> None:
    """Attach candidate evidence to the original MLflow run when tracking is enabled."""
    if training.mlflow_run_id is None:
        return
    import mlflow

    tracking_uri = config.mlflow_tracking_uri
    if tracking_uri is None:
        tracking_uri = f"sqlite:///{DEFAULT_MLFLOW_DATABASE_PATH.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=training.mlflow_run_id):
        mlflow.set_tag("candidate_status", "passed" if decision.passed else "rejected")
        mlflow.log_metrics(
            {f"candidate_{key}": value for key, value in decision.candidate_metrics.items()}
        )
        mlflow.log_artifacts(str(config.bundle_dir), artifact_path="candidate_bundle")


def parse_args() -> PipelineConfig:
    """Parse trusted-runner options for continuous training or local smoke runs."""
    parser = argparse.ArgumentParser(description="Train and gate one model candidate.")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eda-output-dir", type=Path, default=DEFAULT_EDA_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=GateThresholds().min_accuracy)
    parser.add_argument(
        "--min-balanced-accuracy",
        type=float,
        default=GateThresholds().min_balanced_accuracy,
    )
    parser.add_argument("--min-macro-f1", type=float, default=GateThresholds().min_macro_f1)
    parser.add_argument(
        "--max-macro-f1-regression",
        type=float,
        default=GateThresholds().max_macro_f1_regression,
    )
    args = parser.parse_args()
    thresholds = GateThresholds(
        min_accuracy=args.min_accuracy,
        min_balanced_accuracy=args.min_balanced_accuracy,
        min_macro_f1=args.min_macro_f1,
        max_macro_f1_regression=args.max_macro_f1_regression,
    )
    return PipelineConfig(
        version=args.version,
        data_dir=args.data_dir.expanduser().resolve(),
        eda_output_dir=args.eda_output_dir.expanduser().resolve(),
        candidate_dir=args.candidate_dir.expanduser().resolve(),
        runs_dir=args.runs_dir.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
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
        device=args.device,
        amp=not args.no_amp,
        pretrained=not args.no_pretrained,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        disable_mlflow=args.disable_mlflow,
        skip_ingest=args.skip_ingest,
        skip_eda=args.skip_eda,
        thresholds=thresholds,
    )


def main() -> int:
    """CLI entrypoint that preserves rejected-candidate evidence."""
    try:
        result = run_pipeline(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Lỗi pipeline: {exc}", file=sys.stderr)
        return 2
    print(f"Candidate bundle: {result.bundle_dir}")
    print(f"Quality gate: {'PASSED' if result.decision.passed else 'REJECTED'}")
    return 0 if result.decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
