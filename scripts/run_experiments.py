"""Multi-experiment model lifecycle orchestrator.

Runs multiple training experiments with different configs, evaluates each
through the quality gate, selects the best candidate, and deploys it.

Usage:
    python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_BASE = PROJECT_ROOT / "artifacts" / "pipeline" / "candidate"
PYTHON = sys.executable


@dataclass
class ExperimentConfig:
    """One training experiment configuration."""

    name: str
    version: int
    arch: str
    epochs: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    freeze_epochs: int
    warmup_epochs: int
    patience: int
    batch_size: int = 32
    seed: int = 42


# ---------------------------------------------------------------------------
# Experiment matrix: vary architecture, LR, regularisation, schedule
# ---------------------------------------------------------------------------
EXPERIMENTS: list[ExperimentConfig] = [
    # Exp 1: ResNet18 baseline (conservative LR)
    ExperimentConfig(
        name="resnet18-baseline",
        version=4,
        arch="resnet18",
        epochs=10,
        learning_rate=3e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        freeze_epochs=2,
        warmup_epochs=1,
        patience=5,
    ),
    # Exp 2: ResNet18 aggressive (higher LR, more regularisation)
    ExperimentConfig(
        name="resnet18-aggressive",
        version=5,
        arch="resnet18",
        epochs=15,
        learning_rate=5e-4,
        weight_decay=5e-4,
        label_smoothing=0.15,
        freeze_epochs=3,
        warmup_epochs=2,
        patience=7,
    ),
    # Exp 3: ConvNeXtV2-Tiny (larger model, lower LR)
    ExperimentConfig(
        name="convnextv2-tiny",
        version=6,
        arch="convnextv2_tiny.fcmae_ft_in22k_in1k",
        epochs=12,
        learning_rate=1e-4,
        weight_decay=1e-3,
        label_smoothing=0.1,
        freeze_epochs=3,
        warmup_epochs=2,
        patience=5,
    ),
]


def run_pipeline(exp: ExperimentConfig) -> dict | None:
    """Run one pipeline experiment and return the quality gate result."""
    print(f"\n{'='*72}")
    print(f"EXPERIMENT: {exp.name} (v{exp.version})")
    print(f"  arch={exp.arch}, epochs={exp.epochs}, lr={exp.learning_rate}")
    print(f"  wd={exp.weight_decay}, ls={exp.label_smoothing}")
    print(f"  freeze={exp.freeze_epochs}, warmup={exp.warmup_epochs}")
    print(f"{'='*72}\n")

    cmd = [
        PYTHON,
        "-m",
        "training.pipeline",
        "--version",
        str(exp.version),
        "--arch",
        exp.arch,
        "--epochs",
        str(exp.epochs),
        "--learning-rate",
        str(exp.learning_rate),
        "--weight-decay",
        str(exp.weight_decay),
        "--label-smoothing",
        str(exp.label_smoothing),
        "--freeze-epochs",
        str(exp.freeze_epochs),
        "--warmup-epochs",
        str(exp.warmup_epochs),
        "--patience",
        str(exp.patience),
        "--batch-size",
        str(exp.batch_size),
        "--seed",
        str(exp.seed),
        "--skip-ingest",
        "--skip-eda",
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)

    candidate_dir = CANDIDATE_BASE / f"v{exp.version}"
    gate_path = candidate_dir / "quality_gate.json"
    candidate_path = candidate_dir / "candidate.json"

    if result.returncode != 0:
        print(f"  ❌ Pipeline FAILED (exit code {result.returncode})")
        return None

    if not gate_path.is_file():
        print("  ❌ No quality_gate.json found")
        return None

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    passed = gate.get("passed", False)
    metrics = candidate.get("validation_metrics", {})
    print(f"\n  Quality Gate: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(
        f"  Metrics: acc={metrics.get('accuracy', 0):.4f}, "
        f"bal_acc={metrics.get('balanced_accuracy', 0):.4f}, "
        f"macro_f1={metrics.get('macro_f1', 0):.4f}"
    )

    return {
        "name": exp.name,
        "version": exp.version,
        "config": asdict(exp),
        "passed": passed,
        "metrics": metrics,
        "candidate_dir": str(candidate_dir),
    }


def select_best(results: list[dict]) -> dict | None:
    """Select the best passing candidate by macro_f1."""
    passed = [r for r in results if r["passed"]]
    if not passed:
        return None
    return max(passed, key=lambda r: r["metrics"].get("macro_f1", 0))


def deploy_candidate(candidate_dir: str) -> bool:
    """Deploy a quality-gated candidate bundle."""
    print(f"\n{'='*72}")
    print(f"DEPLOYING: {candidate_dir}")
    print(f"{'='*72}\n")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MINIO_ENDPOINT"] = "http://localhost:9000"
    env["MINIO_ACCESS_KEY"] = "minioadmin"
    env["MINIO_SECRET_KEY"] = "minioadmin"
    env["TRITON_HTTP_URL"] = "http://localhost:8000"

    cmd = [
        PYTHON,
        "-m",
        "training.deploy",
        "--bundle-dir",
        candidate_dir,
        "--timeout-seconds",
        "180",
        "--poll-seconds",
        "5",
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=False)
    return result.returncode == 0


def main() -> int:
    """Run all experiments, compare, and deploy best candidate."""
    start = datetime.now(timezone.utc)
    print(f"Multi-experiment lifecycle started at {start.isoformat()}")
    print(f"Experiments: {len(EXPERIMENTS)}")

    results: list[dict] = []

    for exp in EXPERIMENTS:
        result = run_pipeline(exp)
        if result:
            results.append(result)

    # Summary
    print(f"\n\n{'='*72}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*72}")
    print(f"{'Name':<25} {'Arch':<15} {'Gate':<8} {'Acc':<8} {'Bal_Acc':<8} {'F1':<8}")
    print("-" * 72)
    for r in results:
        m = r["metrics"]
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['name']:<25} {r['config']['arch'][:15]:<15} {status:<8} "
            f"{m.get('accuracy', 0):<8.4f} {m.get('balanced_accuracy', 0):<8.4f} "
            f"{m.get('macro_f1', 0):<8.4f}"
        )

    # Select best
    best = select_best(results)
    if not best:
        print("\n❌ No candidate passed quality gate. Deployment skipped.")
        return 1

    print(f"\n🏆 Best candidate: {best['name']} (v{best['version']})")
    print(f"   macro_f1 = {best['metrics']['macro_f1']:.4f}")

    # Deploy
    success = deploy_candidate(best["candidate_dir"])

    # Save comparison report
    report = {
        "started_at": start.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "experiments": results,
        "best_candidate": best["name"],
        "best_version": best["version"],
        "deployed": success,
    }
    report_path = CANDIDATE_BASE / "experiment_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nComparison report: {report_path}")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    if success:
        print("✅ Best candidate deployed successfully!")
        return 0
    else:
        print("⚠️ Deployment failed — check logs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
