"""Custom Prometheus metrics for the FastAPI gateway.

Phơi bày ở GET /metrics; Prometheus scrape job "gateway" (xem
monitoring/prometheus/prometheus.yml). Bổ sung cho metric gốc của Triton:
đo latency, phân phối class dự đoán và phân phối confidence phía gateway.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Đếm số dự đoán theo class -> theo dõi phân phối lớp phục vụ (data drift).
PREDICTIONS = Counter(
    "gateway_predictions_total",
    "Số dự đoán thành công, chia theo class.",
    ["predicted_class"],
)

# Đếm lỗi phân theo loại (bad_image, triton_unavailable, internal...).
PREDICTION_ERRORS = Counter(
    "gateway_prediction_errors_total",
    "Số request /predict lỗi, chia theo loại.",
    ["reason"],
)

# Latency end-to-end của /predict (preprocess + gọi Triton + hậu xử lý).
REQUEST_LATENCY = Histogram(
    "gateway_predict_latency_seconds",
    "Latency end-to-end của endpoint /predict.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Phân phối confidence của top-1 -> phát hiện model 'lưỡng lự' hàng loạt.
CONFIDENCE = Histogram(
    "gateway_prediction_confidence",
    "Phân phối confidence top-1.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)


def render_latest() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
