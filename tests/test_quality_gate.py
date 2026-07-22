"""Tests for Validation-only model quality promotion gates."""

from __future__ import annotations

import json

import pytest

from training.quality_gate import (
    GateThresholds,
    evaluate_quality_gate,
    extract_metrics,
    load_metrics,
    write_decision,
)


def _metrics(macro_f1: float = 0.60) -> dict[str, float]:
    return {
        "accuracy": 0.70,
        "balanced_accuracy": 0.65,
        "macro_f1": macro_f1,
    }


def test_quality_gate_passes_absolute_and_champion_requirements() -> None:
    decision = evaluate_quality_gate(
        _metrics(),
        GateThresholds(
            min_accuracy=0.60,
            min_balanced_accuracy=0.60,
            min_macro_f1=0.55,
            max_macro_f1_regression=0.02,
        ),
        _metrics(0.61),
    )

    assert decision.passed
    assert len(decision.checks) == 4
    assert decision.checks[-1].name == "champion_macro_f1_regression"


def test_quality_gate_rejects_macro_f1_regression_and_writes_decision(tmp_path) -> None:
    decision = evaluate_quality_gate(
        _metrics(0.55),
        GateThresholds(max_macro_f1_regression=0.02),
        _metrics(0.60),
    )
    output = tmp_path / "quality_gate.json"

    write_decision(decision, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not decision.passed
    assert not payload["passed"]
    assert payload["checks"][-1]["threshold"] == pytest.approx(0.58)


def test_extract_metrics_accepts_evaluation_artifact_and_rejects_missing_value(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"metrics": _metrics()}), encoding="utf-8")

    assert load_metrics(metrics_path) == _metrics()
    with pytest.raises(ValueError, match="balanced_accuracy"):
        extract_metrics({"accuracy": 0.8, "macro_f1": 0.8})
