"""Per-class fairness and bias analysis for the 9-class skin-lesion classifier.

Reads evaluation ``predictions.csv`` and ``metrics.json`` produced by
``training.evaluate`` and generates a bias report highlighting classes
with disproportionately low performance — a proxy for model-level
fairness across diagnostic categories.

Intended usage:
    python -m responsible_ai.fairness \
        --predictions artifacts/training/runs/<run>/evaluation-test/predictions.csv \
        --metrics artifacts/training/runs/<run>/evaluation-test/metrics.json \
        --output artifacts/responsible_ai/
"""

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
from sklearn.metrics import f1_score, precision_score, recall_score

# Canonical contract (mirror ``training/model.py``).
CLASS_NAMES: tuple[str, ...] = (
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
)
NUM_CLASSES = len(CLASS_NAMES)

# Clinical-risk severity tiers (higher → more dangerous misclassification).
CLINICAL_RISK: dict[str, str] = {
    "melanoma": "high",
    "basal cell carcinoma": "high",
    "squamous cell carcinoma": "high",
    "actinic keratosis": "medium",
    "pigmented benign keratosis": "low",
    "seborrheic keratosis": "low",
    "dermatofibroma": "low",
    "nevus": "low",
    "vascular lesion": "low",
}


@dataclass(frozen=True)
class FairnessReport:
    """Complete fairness analysis result."""

    per_class_metrics: dict[str, dict[str, float]]
    disparity_metrics: dict[str, float]
    high_risk_analysis: dict[str, Any]
    confusion_pairs: list[dict[str, Any]]
    recommendations: list[str]


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load and validate evaluation predictions."""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    required = {"target_class", "predicted_class", "target_index", "predicted_index", "confidence"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions thiếu columns: {', '.join(sorted(missing))}")
    return predictions


def compute_per_class_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1, and support for each class."""
    targets = predictions["target_index"].to_numpy()
    preds = predictions["predicted_index"].to_numpy()

    precision = precision_score(
        targets, preds, labels=range(NUM_CLASSES), average=None, zero_division=0
    )
    recall = recall_score(targets, preds, labels=range(NUM_CLASSES), average=None, zero_division=0)
    f1 = f1_score(targets, preds, labels=range(NUM_CLASSES), average=None, zero_division=0)

    result: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(CLASS_NAMES):
        support = int((targets == idx).sum())
        result[name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1_score": round(float(f1[idx]), 4),
            "support": support,
            "clinical_risk": CLINICAL_RISK[name],
        }
    return result


def compute_disparity_metrics(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute disparity measures across all classes."""
    f1_values = [v["f1_score"] for v in per_class.values()]
    recall_values = [v["recall"] for v in per_class.values()]
    precision_values = [v["precision"] for v in per_class.values()]

    return {
        "f1_max": round(max(f1_values), 4),
        "f1_min": round(min(f1_values), 4),
        "f1_range": round(max(f1_values) - min(f1_values), 4),
        "f1_std": round(float(np.std(f1_values)), 4),
        "recall_range": round(max(recall_values) - min(recall_values), 4),
        "precision_range": round(max(precision_values) - min(precision_values), 4),
        "worst_class": min(per_class, key=lambda k: per_class[k]["f1_score"]),
        "best_class": max(per_class, key=lambda k: per_class[k]["f1_score"]),
    }


def analyze_high_risk_classes(per_class: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Evaluate performance on clinically high-risk classes."""
    high_risk = {k: v for k, v in per_class.items() if CLINICAL_RISK[k] == "high"}
    low_risk = {k: v for k, v in per_class.items() if CLINICAL_RISK[k] == "low"}

    high_risk_f1 = [v["f1_score"] for v in high_risk.values()]
    low_risk_f1 = [v["f1_score"] for v in low_risk.values()]

    return {
        "high_risk_classes": list(high_risk.keys()),
        "high_risk_avg_f1": round(float(np.mean(high_risk_f1)), 4) if high_risk_f1 else 0.0,
        "low_risk_avg_f1": round(float(np.mean(low_risk_f1)), 4) if low_risk_f1 else 0.0,
        "high_risk_min_recall": round(
            min((v["recall"] for v in high_risk.values()), default=0.0), 4
        ),
        "concern": any(v["recall"] < 0.5 for v in high_risk.values()),
        "detail": {
            name: {"recall": v["recall"], "f1_score": v["f1_score"]}
            for name, v in high_risk.items()
        },
    }


def find_confusion_pairs(predictions: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Identify the most frequent misclassification pairs."""
    misclassified = predictions[predictions["target_class"] != predictions["predicted_class"]]
    if misclassified.empty:
        return []

    pair_counts = (
        misclassified.groupby(["target_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return [
        {
            "true_class": row["target_class"],
            "predicted_as": row["predicted_class"],
            "count": int(row["count"]),
            "true_risk": CLINICAL_RISK.get(row["target_class"], "unknown"),
            "predicted_risk": CLINICAL_RISK.get(row["predicted_class"], "unknown"),
            "risk_direction": _risk_direction(row["target_class"], row["predicted_class"]),
        }
        for _, row in pair_counts.iterrows()
    ]


def _risk_direction(true_class: str, predicted_class: str) -> str:
    """Classify whether a misclassification is a dangerous downgrade."""
    risk_order = {"high": 3, "medium": 2, "low": 1}
    true_risk = risk_order.get(CLINICAL_RISK.get(true_class, "low"), 1)
    pred_risk = risk_order.get(CLINICAL_RISK.get(predicted_class, "low"), 1)
    if true_risk > pred_risk:
        return "dangerous_downgrade"
    if true_risk < pred_risk:
        return "safe_upgrade"
    return "same_tier"


def generate_recommendations(
    disparity: dict[str, float],
    high_risk: dict[str, Any],
    confusion_pairs: list[dict[str, Any]],
) -> list[str]:
    """Generate actionable fairness recommendations."""
    recs: list[str] = []

    if disparity["f1_range"] > 0.3:
        recs.append(
            f"F1 disparity cao ({disparity['f1_range']:.2f}): class "
            f"'{disparity['worst_class']}' yếu nhất. Cân nhắc oversampling, "
            f"class-specific augmentation hoặc focal loss."
        )

    if high_risk.get("concern"):
        recs.append(
            "Cảnh báo: một hoặc nhiều class high-risk (melanoma, BCC, SCC) có recall < 0.5. "
            "False negative cho các class này có thể trì hoãn điều trị ung thư."
        )

    dangerous = [p for p in confusion_pairs if p["risk_direction"] == "dangerous_downgrade"]
    if dangerous:
        worst = dangerous[0]
        recs.append(
            f"Nhầm lẫn nguy hiểm nhất: '{worst['true_class']}' (high-risk) bị classify "
            f"thành '{worst['predicted_as']}' (low-risk) — {worst['count']} trường hợp."
        )

    recs.append(
        "Model chỉ nên dùng như công cụ hỗ trợ sàng lọc (decision-support). "
        "Bác sĩ da liễu phải xác nhận mọi kết quả trước khi đưa ra quyết định lâm sàng."
    )
    recs.append(
        "Dataset ISIC không đại diện đầy đủ các tông da (skin tone). "
        "Cần kiểm tra performance trên nhóm bệnh nhân đa dạng trước khi deploy."
    )
    return recs


def plot_per_class_fairness(
    per_class: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot a grouped bar chart of precision, recall, F1 per class."""
    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1_score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    fig, axis = plt.subplots(figsize=(14, 7))
    axis.bar(x - width, precision, width, label="Precision", color="#4285F4")
    axis.bar(x, recall, width, label="Recall", color="#EA4335")
    axis.bar(x + width, f1, width, label="F1-Score", color="#34A853")

    axis.set_ylabel("Score")
    axis.set_title("Per-Class Fairness Analysis — Skin Cancer ISIC 9 Classes")
    axis.set_xticks(x)
    axis.set_xticklabels(classes, rotation=45, ha="right")
    axis.legend()
    axis.set_ylim(0, 1.05)
    axis.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")

    for idx, cls in enumerate(classes):
        if CLINICAL_RISK[cls] == "high":
            axis.get_xticklabels()[idx].set_color("red")
            axis.get_xticklabels()[idx].set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_fairness_analysis(
    predictions_path: Path,
    output_dir: Path,
) -> FairnessReport:
    """Run complete fairness analysis and persist artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(predictions_path)
    per_class = compute_per_class_metrics(predictions)
    disparity = compute_disparity_metrics(per_class)
    high_risk = analyze_high_risk_classes(per_class)
    confusion_pairs = find_confusion_pairs(predictions)
    recommendations = generate_recommendations(disparity, high_risk, confusion_pairs)

    report = FairnessReport(
        per_class_metrics=per_class,
        disparity_metrics=disparity,
        high_risk_analysis=high_risk,
        confusion_pairs=confusion_pairs,
        recommendations=recommendations,
    )

    report_payload = {
        "per_class_metrics": per_class,
        "disparity_metrics": disparity,
        "high_risk_analysis": high_risk,
        "top_confusion_pairs": confusion_pairs,
        "recommendations": recommendations,
    }
    (output_dir / "fairness_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_per_class_fairness(per_class, output_dir / "per_class_fairness.png")

    return report


def parse_args() -> argparse.Namespace:
    """Parse fairness analysis CLI arguments."""
    parser = argparse.ArgumentParser(description="Per-class fairness and bias analysis.")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Path to evaluation predictions.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/responsible_ai"),
        help="Output directory for fairness artifacts",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        report = run_fairness_analysis(args.predictions, args.output)
    except (OSError, ValueError) as exc:
        print(f"Fairness analysis error: {exc}", file=sys.stderr)
        return 1

    print(f"Fairness report: {args.output / 'fairness_report.json'}")
    print(f"Classes analyzed: {NUM_CLASSES}")
    print(f"F1 range: {report.disparity_metrics['f1_range']:.4f}")
    print(f"Worst class: {report.disparity_metrics['worst_class']}")
    if report.recommendations:
        print("\nRecommendations:")
        for idx, rec in enumerate(report.recommendations, 1):
            print(f"  {idx}. {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
