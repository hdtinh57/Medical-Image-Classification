"""Triton inference client for the 9-class skin-lesion classifier.

Giữ gateway NHẸ: preprocess bằng Pillow + NumPy thuần (không torch/timm), rồi
gọi Triton qua HTTP v2 inference protocol. Contract dưới đây PHẢI khớp với
model_repository/skin_classifier/config.pbtxt (output logits[9]) và với
preprocessing đã dùng lúc train — sai preprocess thì predict lệch âm thầm.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException

# --- Serving contract (mirror training/model.py + config.pbtxt) ----------------
MODEL_NAME = "skin_classifier"
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
IMAGE_SIZE = 224
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Tên tensor phải khớp config.pbtxt (sai tên -> Triton trả lỗi).
INPUT_NAME = "input"
OUTPUT_NAME = "logits"

# Resize theo training/preprocess.py::build_eval_transform: shorter side -> 256,
# rồi CenterCrop 224 (giữ đúng preprocess lúc train).
_RESIZE_SHORTER = round(IMAGE_SIZE / 0.875)  # = 256

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
_INFER_TIMEOUT_S = float(os.getenv("TRITON_TIMEOUT", "10"))


@dataclass(frozen=True)
class Prediction:
    """Single-image inference result."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def _resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    """Resize preserving aspect ratio so the shorter side equals ``target``."""
    width, height = image.size
    scale = target / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop a centered ``size`` x ``size`` square."""
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Turn raw image bytes into a normalized ``[1, 3, 224, 224]`` FP32 batch.

    Khớp deterministic eval transform (training/preprocess.py): Resize shorter
    side -> 256 -> CenterCrop(224) -> scale [0, 1] -> Normalize(ImageNet) -> CHW.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _resize_shorter_side(image, _RESIZE_SHORTER)
    image = _center_crop(image, IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
    array = (array - IMAGE_MEAN) / IMAGE_STD  # normalize theo channel
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(array[np.newaxis, :], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


class TritonInferenceClient:
    """Thin wrapper around the Triton HTTP client for one classifier model."""

    def __init__(self, url: str = TRITON_URL, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._client = httpclient.InferenceServerClient(url=url)

    def is_ready(self) -> bool:
        """Return True only when the server is live and the model is loaded."""
        try:
            return self._client.is_server_ready() and self._client.is_model_ready(
                self._model_name
            )
        except InferenceServerException:
            return False

    def infer(self, image_bytes: bytes) -> Prediction:
        """Run one image through Triton and return the decoded prediction."""
        batch = preprocess(image_bytes)

        infer_input = httpclient.InferInput(INPUT_NAME, batch.shape, "FP32")
        infer_input.set_data_from_numpy(batch)
        requested_output = httpclient.InferRequestedOutput(OUTPUT_NAME)

        response = self._client.infer(
            self._model_name,
            inputs=[infer_input],
            outputs=[requested_output],
            timeout=_INFER_TIMEOUT_S,
        )
        logits = response.as_numpy(OUTPUT_NAME)[0]  # shape [9]
        probabilities = _softmax(logits)

        best = int(np.argmax(probabilities))
        return Prediction(
            predicted_class=CLASS_NAMES[best],
            confidence=float(probabilities[best]),
            probabilities={
                name: float(probability)
                for name, probability in zip(CLASS_NAMES, probabilities)
            },
        )
