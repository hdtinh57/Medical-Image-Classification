"""FastAPI gateway — image upload → Triton inference → JSON response.

Nhẹ: không torch/timm. Preprocessing và Triton HTTP call nằm trong
``triton_client.py``; Prometheus custom metrics nằm trong ``metrics.py``.
Feedback loop và drift detection nằm trong ``monitoring/``.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiofiles
import httpx
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from gateway.metrics import (
    CONFIDENCE,
    PREDICTION_ERRORS,
    PREDICTIONS,
    REQUEST_LATENCY,
    render_latest,
)
from gateway.triton_client import CLASS_NAMES, NUM_CLASSES, TritonInferenceClient
from monitoring.drift import DriftDetector, DriftThresholds
from monitoring.feedback import FeedbackCollector

_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

_triton: TritonInferenceClient | None = None
_feedback: FeedbackCollector | None = None
_drift_detector: DriftDetector | None = None
_recent_classes: deque[str] = deque(maxlen=500)
_recent_confidences: deque[float] = deque(maxlen=500)


def _get_triton() -> TritonInferenceClient:
    """Return the shared Triton client, raising 503 when uninitialised."""
    if _triton is None:
        raise HTTPException(status_code=503, detail="Triton client chưa khởi tạo.")
    return _triton


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialise the Triton client, feedback collector, and drift detector."""
    global _triton, _feedback, _drift_detector  # noqa: PLW0603
    _triton = TritonInferenceClient()
    _feedback = FeedbackCollector(class_names=list(CLASS_NAMES))

    # Initialise drift detector with uniform reference distribution.
    # In production, load from validation predictions via load_reference_from_predictions().
    uniform_dist = [1.0 / NUM_CLASSES] * NUM_CLASSES
    _drift_detector = DriftDetector(
        class_names=list(CLASS_NAMES),
        reference_class_dist=uniform_dist,
        reference_confidences=[0.5] * 100,  # placeholder baseline
        thresholds=DriftThresholds(),
    )
    yield
    _triton = None
    _feedback = None
    _drift_detector = None


app = FastAPI(
    title="Skin Cancer Classification Gateway",
    description="API gateway cho mô hình phân loại tổn thương da 9 lớp qua Triton.",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount UI static files
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


# ── Health ───────────────────────────────────────────────────────────────


@app.get(
    "/health",
    summary="Server và Triton readiness",
    tags=["Health"],
    response_model=dict[str, Any],
)
async def health() -> dict[str, Any]:
    """Kiểm tra gateway đang chạy và Triton model đã sẵn sàng.

    Returns **200** khi cả hai đều healthy, **503** khi Triton chưa sẵn sàng.
    """
    triton_ready = _get_triton().is_ready()
    status = "healthy" if triton_ready else "degraded"
    payload: dict[str, Any] = {
        "status": status,
        "gateway": "running",
        "triton": "ready" if triton_ready else "not ready",
        "model_classes": NUM_CLASSES,
    }
    if not triton_ready:
        return JSONResponse(content=payload, status_code=503)
    return payload


# ── Predict ──────────────────────────────────────────────────────────────


@app.post(
    "/predict",
    summary="Phân loại ảnh tổn thương da",
    tags=["Inference"],
    response_model=dict[str, Any],
)
async def predict(
    file: UploadFile = File(..., description="Ảnh da liễu (JPEG, PNG, BMP, TIFF, WebP)"),
) -> dict[str, Any]:
    """Upload một ảnh và nhận kết quả phân loại 9 lớp.

    **Response** gồm:
    - ``predicted_class``: tên lớp có xác suất cao nhất
    - ``confidence``: xác suất top-1
    - ``probabilities``: phân phối xác suất toàn bộ 9 lớp
    """
    image_bytes = await _read_and_validate_upload(file)
    client = _get_triton()

    if not client.is_ready():
        PREDICTION_ERRORS.labels(reason="triton_unavailable").inc()
        raise HTTPException(status_code=503, detail="Triton model chưa sẵn sàng.")

    start = time.perf_counter()
    try:
        result = client.infer(image_bytes)
    except Exception as exc:
        PREDICTION_ERRORS.labels(reason="inference_error").inc()
        raise HTTPException(
            status_code=502,
            detail=f"Triton inference lỗi: {exc}",
        ) from exc
    elapsed = time.perf_counter() - start

    REQUEST_LATENCY.observe(elapsed)
    PREDICTIONS.labels(predicted_class=result.predicted_class).inc()
    CONFIDENCE.observe(result.confidence)

    # Track for drift detection
    _recent_classes.append(result.predicted_class)
    _recent_confidences.append(result.confidence)

    # Register for feedback matching
    prediction_id = uuid.uuid4().hex[:12]
    if _feedback is not None:
        _feedback.register_prediction(prediction_id, result.predicted_class)

    return {
        "prediction_id": prediction_id,
        "predicted_class": result.predicted_class,
        "confidence": round(result.confidence, 6),
        "probabilities": {name: round(prob, 6) for name, prob in result.probabilities.items()},
        "latency_seconds": round(elapsed, 4),
    }


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    """Read and validate an uploaded image file."""
    _validate_content_type(file)
    image_bytes = await file.read()
    _validate_size(image_bytes)
    _validate_image(image_bytes)
    return image_bytes


def _validate_content_type(file: UploadFile) -> None:
    """Reject files with an unsupported MIME type."""
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        PREDICTION_ERRORS.labels(reason="bad_content_type").inc()
        raise HTTPException(
            status_code=415,
            detail=f"Content type '{file.content_type}' không được hỗ trợ. "
            f"Chấp nhận: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}",
        )


def _validate_size(image_bytes: bytes) -> None:
    """Reject uploads that exceed the size limit."""
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        PREDICTION_ERRORS.labels(reason="file_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail=f"File vượt giới hạn {_MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )


def _validate_image(image_bytes: bytes) -> None:
    """Reject uploads that Pillow cannot decode as a valid image."""
    try:
        with Image.open(__import__("io").BytesIO(image_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, Exception):
        PREDICTION_ERRORS.labels(reason="bad_image").inc()
        raise HTTPException(
            status_code=422,
            detail="File không phải ảnh hợp lệ hoặc ảnh bị hỏng.",
        )


# ── Prometheus metrics ───────────────────────────────────────────────────


@app.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    tags=["Monitoring"],
    include_in_schema=False,
)
async def metrics() -> Response:
    """Trả về metrics dạng Prometheus exposition format.

    Được Prometheus scrape tự động (``monitoring/prometheus/prometheus.yml``).
    """
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


# ── Feedback & Monitoring ────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Ground-truth feedback for a previous prediction."""

    prediction_id: str
    actual_class: str


@app.post(
    "/feedback",
    summary="Submit ground-truth feedback",
    tags=["Monitoring"],
    response_model=dict[str, Any],
)
async def feedback(body: FeedbackRequest) -> dict[str, Any]:
    """Submit the actual diagnosis for a previous prediction.

    Use the ``prediction_id`` from the ``/predict`` response to match.
    Enables production accuracy monitoring via ``GET /performance``.
    """
    if _feedback is None:
        raise HTTPException(status_code=503, detail="Feedback collector chưa sẵn sàng.")

    if body.actual_class not in CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Class '{body.actual_class}' không hợp lệ. "
            f"Classes: {', '.join(CLASS_NAMES)}",
        )

    record = _feedback.submit_feedback(body.prediction_id, body.actual_class)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction ID '{body.prediction_id}' không tìm thấy hoặc đã expired.",
        )

    return {
        "status": "accepted",
        "correct": record.correct,
        "predicted_class": record.predicted_class,
        "actual_class": record.actual_class,
    }


@app.get(
    "/performance",
    summary="Production accuracy from feedback",
    tags=["Monitoring"],
    response_model=dict[str, Any],
)
async def performance() -> dict[str, Any]:
    """Rolling production accuracy computed from ground-truth feedback."""
    if _feedback is None:
        raise HTTPException(status_code=503, detail="Feedback collector chưa sẵn sàng.")
    snapshot = _feedback.get_snapshot()
    return {
        "window_size": snapshot.window_size,
        "total_feedback": snapshot.total_feedback,
        "accuracy": snapshot.accuracy,
        "per_class_accuracy": snapshot.per_class_accuracy,
        "top_confusion_pairs": snapshot.confusion_pairs,
    }


@app.get(
    "/drift",
    summary="Drift detection status",
    tags=["Monitoring"],
    response_model=dict[str, Any],
)
async def drift() -> dict[str, Any]:
    """Statistical drift analysis on recent predictions.

    Uses PSI (Population Stability Index) on class distribution and
    KS (Kolmogorov-Smirnov) test on confidence values.
    """
    if _drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector chưa sẵn sàng.")

    recent_cls = list(_recent_classes)
    recent_conf = list(_recent_confidences)

    if len(recent_cls) < 10:
        return {"status": "insufficient_data", "message": "Cần ít nhất 10 predictions."}

    report = _drift_detector.evaluate(recent_cls, recent_conf)
    return {
        "status": report.overall_status,
        "window_size": report.window_size,
        "class_distribution_psi": {
            "value": report.class_distribution_psi.psi_value,
            "status": report.class_distribution_psi.status,
        },
        "confidence_ks_test": {
            "statistic": report.confidence_ks.statistic,
            "p_value": report.confidence_ks.p_value,
            "drift_detected": report.confidence_ks.drift_detected,
        },
        "confidence_shift": {
            "reference_mean": report.confidence_shift.reference_mean,
            "current_mean": report.confidence_shift.current_mean,
            "drift_detected": report.confidence_shift.drift_detected,
        },
        "per_class_drift": report.per_class_drift,
        "recommendations": report.recommendations,
    }


# ── Admin ──────────────────────────────────────────────────────────────────


@app.post(
    "/admin/retrain",
    summary="Trigger model retraining pipeline",
    tags=["Admin"],
    response_model=dict[str, Any],
)
async def trigger_retrain() -> dict[str, Any]:
    """Trigger GitHub Actions scheduled-retrain workflow.

    Requires GITHUB_TOKEN and GITHUB_REPO environment variables.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPO", "Dat-V/Medical-Image-Classification")

    if not token:
        # Mock success for local demo if no token is provided
        return {
            "status": "mock_success",
            "message": "GITHUB_TOKEN not found. Mocking retrain trigger.",
        }

    url = f"https://api.github.com/repos/{repo}/actions/workflows/scheduled-retrain.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"ref": "main", "inputs": {"reason": "drift_detected"}}

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload, timeout=10.0)

    if resp.status_code != 204:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi gọi GitHub API: {resp.status_code} {resp.text}",
        )

    return {"status": "success", "message": "Đã trigger retrain workflow thành công."}


@app.post(
    "/admin/data",
    summary="Upload new training data (zip)",
    tags=["Admin"],
    response_model=dict[str, Any],
)
async def upload_new_data(
    file: UploadFile = File(..., description="ZIP archive containing new images"),
) -> dict[str, Any]:
    """Upload new data for the next retraining cycle."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ upload file .zip")

    # Save to mounted volume /app/data/new_uploads
    upload_dir = Path("/app/data/new_uploads")
    if not upload_dir.exists():
        # Fallback to local if not running in docker with volume
        upload_dir = Path(__file__).parents[1] / "data" / "new_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / f"upload_{uuid.uuid4().hex[:8]}.zip"
    async with aiofiles.open(dest_path, "wb") as out_file:
        while content := await file.read(1024 * 1024):
            await out_file.write(content)

    return {
        "status": "success",
        "message": f"Đã lưu file {file.filename} vào thư mục chờ retrain.",
        "path": str(dest_path.name),
    }


# ── Root ─────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def redirect_to_ui():
    """Chuyển hướng trang chủ sang UI."""
    return RedirectResponse(url="/ui/")


@app.get(
    "/api/info",
    summary="API info",
    tags=["Info"],
    response_model=dict[str, Any],
)
async def root() -> dict[str, Any]:
    """Thông tin tổng quan API."""
    return {
        "service": "Skin Cancer Classification Gateway",
        "version": "1.1.0",
        "model": "skin_classifier (9-class ONNX via Triton)",
        "classes": list(CLASS_NAMES),
        "endpoints": {
            "predict": "POST /predict",
            "feedback": "POST /feedback",
            "health": "GET /health",
            "performance": "GET /performance",
            "drift": "GET /drift",
            "docs": "GET /docs",
            "redoc": "GET /redoc",
            "metrics": "GET /metrics",
        },
    }
