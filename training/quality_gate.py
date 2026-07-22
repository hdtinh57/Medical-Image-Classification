"""Reusable validation-metric quality gates for model promotion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GATED_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
DEFAULT_MIN_ACCURACY = 0.50
DEFAULT_MIN_BALANCED_ACCURACY = 0.50
DEFAULT_MIN_MACRO_F1 = 0.50
DEFAULT_MAX_MACRO_F1_REGRESSION = 0.02


@dataclass(frozen=True)
class GateThresholds:
    """Absolute and champion-relative requirements for one candidate."""

    min_accuracy: float = DEFAULT_MIN_ACCURACY
    min_balanced_accuracy: float = DEFAULT_MIN_BALANCED_ACCURACY
    min_macro_f1: float = DEFAULT_MIN_MACRO_F1
    max_macro_f1_regression: float = DEFAULT_MAX_MACRO_F1_REGRESSION

    def validate(self) -> None:
        """Reject impossible metric thresholds before evaluating a candidate."""
        values = {
            "min_accuracy": self.min_accuracy,
            "min_balanced_accuracy": self.min_balanced_accuracy,
            "min_macro_f1": self.min_macro_f1,
            "max_macro_f1_regression": self.max_macro_f1_regression,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong [0, 1].")


@dataclass(frozen=True)
class GateCheck:
    """One machine-readable promotion check."""

    name: str
    metric: str
    actual: float
    operator: str
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateDecision:
    """Complete result persisted with each model candidate."""

    passed: bool
    candidate_metrics: dict[str, float]
    reference_metrics: dict[str, float] | None
    thresholds: GateThresholds
    checks: tuple[GateCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the decision to a JSON-safe artifact payload."""
        return {
            "format_version": 1,
            "passed": self.passed,
            "candidate_metrics": self.candidate_metrics,
            "reference_metrics": self.reference_metrics,
            "thresholds": asdict(self.thresholds),
            "checks": [asdict(check) for check in self.checks],
        }


def load_metrics(path: Path) -> dict[str, float]:
    """Load scalar classification metrics from evaluation or deployment JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return extract_metrics(payload)


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract the shared scalar metrics from supported artifact schemas."""
    source = _metric_source(payload)
    metrics: dict[str, float] = {}
    for name in GATED_METRICS:
        if name not in source:
            raise ValueError(f"Metrics artifact thiếu '{name}'.")
        value = float(source[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Metric '{name}' phải nằm trong [0, 1].")
        metrics[name] = value
    return metrics


def _metric_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve evaluation, production-pointer, or direct metric payloads."""
    for key in ("metrics", "validation_metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def evaluate_quality_gate(
    candidate_metrics: dict[str, float],
    thresholds: GateThresholds = GateThresholds(),
    reference_metrics: dict[str, float] | None = None,
) -> GateDecision:
    """Evaluate absolute requirements and an optional champion regression limit."""
    thresholds.validate()
    candidate = extract_metrics(candidate_metrics)
    reference = extract_metrics(reference_metrics) if reference_metrics is not None else None
    checks = _absolute_checks(candidate, thresholds)
    if reference is not None:
        checks.append(_regression_check(candidate, reference, thresholds))
    return GateDecision(
        passed=all(check.passed for check in checks),
        candidate_metrics=candidate,
        reference_metrics=reference,
        thresholds=thresholds,
        checks=tuple(checks),
    )


def _absolute_checks(
    metrics: dict[str, float],
    thresholds: GateThresholds,
) -> list[GateCheck]:
    """Build the three absolute model-quality checks."""
    requirements = {
        "accuracy": thresholds.min_accuracy,
        "balanced_accuracy": thresholds.min_balanced_accuracy,
        "macro_f1": thresholds.min_macro_f1,
    }
    return [
        GateCheck(
            name=f"minimum_{metric}",
            metric=metric,
            actual=metrics[metric],
            operator=">=",
            threshold=minimum,
            passed=metrics[metric] >= minimum,
        )
        for metric, minimum in requirements.items()
    ]


def _regression_check(
    candidate: dict[str, float],
    reference: dict[str, float],
    thresholds: GateThresholds,
) -> GateCheck:
    """Limit macro-F1 regression relative to the deployed champion."""
    minimum = reference["macro_f1"] - thresholds.max_macro_f1_regression
    return GateCheck(
        name="champion_macro_f1_regression",
        metric="macro_f1",
        actual=candidate["macro_f1"],
        operator=">=",
        threshold=minimum,
        passed=candidate["macro_f1"] >= minimum,
    )


def write_decision(decision: GateDecision, output_path: Path) -> None:
    """Persist a quality-gate decision even when promotion is rejected."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(decision.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse standalone quality-gate CLI options."""
    parser = argparse.ArgumentParser(description="Gate a model using Validation metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument(
        "--min-balanced-accuracy",
        type=float,
        default=DEFAULT_MIN_BALANCED_ACCURACY,
    )
    parser.add_argument("--min-macro-f1", type=float, default=DEFAULT_MIN_MACRO_F1)
    parser.add_argument(
        "--max-macro-f1-regression",
        type=float,
        default=DEFAULT_MAX_MACRO_F1_REGRESSION,
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint used by local and CI promotion checks."""
    args = parse_args()
    thresholds = GateThresholds(
        min_accuracy=args.min_accuracy,
        min_balanced_accuracy=args.min_balanced_accuracy,
        min_macro_f1=args.min_macro_f1,
        max_macro_f1_regression=args.max_macro_f1_regression,
    )
    try:
        reference = load_metrics(args.reference) if args.reference else None
        decision = evaluate_quality_gate(load_metrics(args.metrics), thresholds, reference)
        write_decision(decision, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Lỗi quality gate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
