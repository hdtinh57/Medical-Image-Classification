"""Integration tests for the FastAPI gateway endpoints.

Tests use httpx.ASGITransport so they run without a live Triton server;
the TritonInferenceClient is patched to return deterministic predictions.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from gateway.main import app
from gateway.triton_client import CLASS_NAMES, Prediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DUMMY_PREDICTION = Prediction(
    predicted_class="melanoma",
    confidence=0.85,
    probabilities={name: (0.85 if name == "melanoma" else 0.01875) for name in CLASS_NAMES},
)


def _create_test_image(fmt: str = "JPEG") -> bytes:
    """Create a minimal valid image in the requested format."""
    buffer = BytesIO()
    Image.new("RGB", (224, 224), color=(128, 64, 32)).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture()
def mock_triton_client():
    """Patch the gateway's Triton client with a mock that returns deterministic results."""
    mock_client = MagicMock()
    mock_client.is_ready.return_value = True
    mock_client.infer.return_value = _DUMMY_PREDICTION
    with patch("gateway.main._triton", mock_client):
        yield mock_client


@pytest.fixture()
def mock_triton_unavailable():
    """Patch the gateway's Triton client to simulate an unavailable server."""
    mock_client = MagicMock()
    mock_client.is_ready.return_value = False
    with patch("gateway.main._triton", mock_client):
        yield mock_client


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRoot:
    """Tests for the root info endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_service_info(self, mock_triton_client: MagicMock) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/")

        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "Skin Cancer Classification Gateway"
        assert "predict" in body["endpoints"]
        assert len(body["classes"]) == 9


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for the health endpoint."""

    @pytest.mark.asyncio
    async def test_health_ok_when_triton_ready(self, mock_triton_client: MagicMock) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["triton"] == "ready"

    @pytest.mark.asyncio
    async def test_health_503_when_triton_not_ready(
        self,
        mock_triton_unavailable: MagicMock,
    ) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for the image classification endpoint."""

    @pytest.mark.asyncio
    async def test_predict_valid_jpeg(self, mock_triton_client: MagicMock) -> None:
        image_bytes = _create_test_image("JPEG")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] == "melanoma"
        assert body["confidence"] == pytest.approx(0.85, abs=1e-4)
        assert len(body["probabilities"]) == 9
        assert "latency_seconds" in body

    @pytest.mark.asyncio
    async def test_predict_valid_png(self, mock_triton_client: MagicMock) -> None:
        image_bytes = _create_test_image("PNG")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("test.png", image_bytes, "image/png")},
            )

        assert response.status_code == 200
        assert response.json()["predicted_class"] == "melanoma"

    @pytest.mark.asyncio
    async def test_predict_rejects_unsupported_content_type(
        self,
        mock_triton_client: MagicMock,
    ) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("test.txt", b"not an image", "text/plain")},
            )

        assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_predict_rejects_corrupt_image(self, mock_triton_client: MagicMock) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("corrupt.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_503_when_triton_unavailable(
        self,
        mock_triton_unavailable: MagicMock,
    ) -> None:
        image_bytes = _create_test_image("JPEG")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_predict_502_when_triton_inference_fails(
        self,
        mock_triton_client: MagicMock,
    ) -> None:
        mock_triton_client.infer.side_effect = RuntimeError("Triton connection refused")
        image_bytes = _create_test_image("JPEG")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_predict_missing_file_returns_422(self, mock_triton_client: MagicMock) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/predict")

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for the Prometheus metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(
        self,
        mock_triton_client: MagicMock,
    ) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        body = response.text
        assert "gateway_predictions_total" in body or "# HELP" in body
