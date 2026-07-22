"""FastAPI serving gateway: upload ảnh -> preprocess -> Triton -> kết quả.

Endpoints:
  GET  /                -> redirect tới Swagger UI (/docs)
  GET  /health/live     -> gateway còn sống
  GET  /health/ready    -> Triton sẵn sàng + model đã nạp
  POST /predict         -> upload 1 ảnh, trả {predicted_class, confidence, ...}
  GET  /metrics         -> Prometheus exposition (job "gateway")

Swagger/OpenAPI tự sinh tại /docs (ăn điểm phần F).
"""

from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import UnidentifiedImageError

from gateway import metrics
from gateway.triton_client import CLASS_NAMES, MODEL_NAME, TritonInferenceClient

_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB — chặn upload quá lớn.

app = FastAPI(
    title="Skin Lesion Classifier Gateway",
    description=(
        "FastAPI gateway: nhận ảnh tổn thương da, tiền xử lý, gọi Triton "
        "Inference Server (ONNX) và trả về class + confidence trên 9 lớp ISIC."
    ),
    version="0.1.0",
)

# Một client dùng chung cho cả process (Triton HTTP client là thread-safe).
_triton = TritonInferenceClient()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect gốc tới Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    """Liveness: process còn phục vụ được request."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def ready() -> dict[str, object]:
    """Readiness: Triton sẵn sàng và model đã nạp xong."""
    if not _triton.is_ready():
        raise HTTPException(status_code=503, detail="Triton hoặc model chưa sẵn sàng.")
    return {"status": "ready", "model": MODEL_NAME}


@app.get("/classes", tags=["meta"])
def classes() -> dict[str, object]:
    """Danh sách 9 lớp theo đúng thứ tự index output."""
    return {"num_classes": len(CLASS_NAMES), "classes": list(CLASS_NAMES)}


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Phân loại một ảnh tổn thương da."""
    started = time.perf_counter()

    if file.content_type is None or not file.content_type.startswith("image/"):
        metrics.PREDICTION_ERRORS.labels(reason="bad_content_type").inc()
        raise HTTPException(status_code=415, detail="File phải là ảnh (image/*).")

    image_bytes = await file.read()
    if not image_bytes:
        metrics.PREDICTION_ERRORS.labels(reason="empty_file").inc()
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        metrics.PREDICTION_ERRORS.labels(reason="too_large").inc()
        raise HTTPException(status_code=413, detail="Ảnh vượt quá 10 MB.")

    try:
        prediction = _triton.infer(image_bytes)
    except UnidentifiedImageError:
        metrics.PREDICTION_ERRORS.labels(reason="bad_image").inc()
        raise HTTPException(status_code=400, detail="Không đọc được ảnh.") from None
    except Exception as exc:  # Triton lỗi / không kết nối được.
        metrics.PREDICTION_ERRORS.labels(reason="triton_error").inc()
        raise HTTPException(status_code=502, detail=f"Lỗi inference: {exc}") from exc

    metrics.PREDICTIONS.labels(predicted_class=prediction.predicted_class).inc()
    metrics.CONFIDENCE.observe(prediction.confidence)
    metrics.REQUEST_LATENCY.observe(time.perf_counter() - started)

    return {
        "predicted_class": prediction.predicted_class,
        "confidence": round(prediction.confidence, 4),
        "probabilities": {
            name: round(probability, 4)
            for name, probability in prediction.probabilities.items()
        },
    }


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition endpoint (scrape job 'gateway')."""
    payload, content_type = metrics.render_latest()
    return Response(content=payload, media_type=content_type)
