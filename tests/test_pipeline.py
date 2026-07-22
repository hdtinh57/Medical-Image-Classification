"""Tests for candidate bundle assembly without running a real GPU training job."""

from __future__ import annotations

import json
from pathlib import Path

from training.pipeline import PipelineConfig, _bundle_checksums, run_pipeline
from training.quality_gate import GateThresholds
from training.train import TrainingResult


def _metrics(macro_f1: float) -> dict[str, float]:
    return {
        "accuracy": 0.80,
        "balanced_accuracy": 0.75,
        "macro_f1": macro_f1,
    }


def test_pipeline_keeps_rejected_candidate_diagnostics(monkeypatch, tmp_path) -> None:
    eda_dir = tmp_path / "eda"
    eda_dir.mkdir()
    (eda_dir / "dataset_index.csv").write_text("index", encoding="utf-8")
    (eda_dir / "summary.json").write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "best_checkpoint.pth").write_bytes(b"checkpoint")
    (run_dir / "manifest_summary.json").write_text("{}", encoding="utf-8")
    training = TrainingResult(
        best_checkpoint=run_dir / "best_checkpoint.pth",
        run_dir=run_dir,
        mlflow_run_id=None,
    )

    monkeypatch.setattr("training.pipeline.train_model_with_result", lambda _: training)
    monkeypatch.setattr("training.pipeline._log_pipeline_artifacts", lambda *_: None)
    monkeypatch.setattr("training.pipeline.evaluate_checkpoint", _fake_evaluate)

    config = PipelineConfig(
        version=7,
        data_dir=tmp_path / "data",
        eda_output_dir=eda_dir,
        candidate_dir=tmp_path / "candidate",
        manifest_path=tmp_path / "manifest.csv",
        skip_ingest=True,
        skip_eda=True,
        thresholds=GateThresholds(min_macro_f1=0.50),
    )

    result = run_pipeline(config)

    assert not result.decision.passed
    assert (result.bundle_dir / "quality_gate.json").is_file()
    assert (result.bundle_dir / "evaluations" / "test" / "metrics.json").is_file()
    checksums = json.loads((result.bundle_dir / "checksums.json").read_text(encoding="utf-8"))
    assert checksums == _bundle_checksums(result.bundle_dir)
    summary = json.loads((result.bundle_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["validation_metrics_path"] == "evaluations/val/metrics.json"
    assert summary["test_metrics_path"] == "evaluations/test/metrics.json"
    assert result.test_metrics["macro_f1"] > result.validation_metrics["macro_f1"]


def _fake_evaluate(config) -> tuple[dict[str, float], Path]:
    output_dir = config.output_dir
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(0.80 if config.split == "test" else 0.40)
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics}),
        encoding="utf-8",
    )
    return metrics, output_dir
