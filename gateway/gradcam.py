"""Grad-CAM visual explanations for the 9-class skin-lesion classifier.

Generates class-activation heatmaps showing which image regions the model
focuses on — a critical explainability tool for clinician trust and
regulatory compliance in medical imaging.

The implementation hooks into the last convolutional layer of timm models
(ResNet18, ConvNeXtV2-Tiny) and produces a blended overlay of the
activation map on the original input image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Import model contract (only needed for CLI; gateway can use this module standalone).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_DIR = _PROJECT_ROOT / "training"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.model import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_SIZE,
    load_model_from_checkpoint,
)
from training.preprocess import build_eval_transform  # noqa: E402


class GradCAM:
    """Compute Grad-CAM heatmaps for a timm classifier.

    Hooks the last convolutional feature-map layer to capture activations
    and their gradients with respect to a target class logit.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._model.eval()
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hook_handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward/backward hooks to the last feature-extraction layer."""
        target_layer = self._find_target_layer()

        def forward_hook(
            _module: torch.nn.Module,
            _input: Any,
            output: torch.Tensor,
        ) -> None:
            self._activations = output.detach()

        def backward_hook(
            _module: torch.nn.Module,
            _grad_input: Any,
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self._gradients = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def _find_target_layer(self) -> torch.nn.Module:
        """Find the last convolutional layer across supported timm architectures."""
        # ResNet family: model.layer4[-1]
        if hasattr(self._model, "layer4"):
            return self._model.layer4[-1]
        # ConvNeXt family: model.stages[-1].blocks[-1]
        if hasattr(self._model, "stages"):
            return self._model.stages[-1].blocks[-1]
        # Generic fallback: walk backward through modules for the last Conv2d
        last_conv = None
        for module in self._model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError("Không tìm được convolutional layer phù hợp cho Grad-CAM.")

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for one image.

        Parameters
        ----------
        input_tensor:
            Preprocessed tensor ``[1, 3, H, W]``.
        target_class:
            Class index to explain. Defaults to the predicted class.

        Returns
        -------
        np.ndarray
            Heatmap array ``[H, W]`` in range ``[0, 1]``.
        """
        input_tensor = input_tensor.requires_grad_(True)
        logits = self._model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self._model.zero_grad()
        target_logit = logits[0, target_class]
        target_logit.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Grad-CAM hooks không capture được gradients/activations.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input image dimensions
        cam_resized = (
            np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (input_tensor.shape[3], input_tensor.shape[2]),
                    Image.Resampling.BILINEAR,
                )
            ).astype(np.float32)
            / 255.0
        )

        return cam_resized

    def remove_hooks(self) -> None:
        """Remove registered hooks to free memory."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, Image.Image]:
    """Load and preprocess an image for Grad-CAM analysis."""
    original = Image.open(image_path).convert("RGB")
    transform = build_eval_transform(IMAGE_SIZE)
    tensor = transform(original).unsqueeze(0)
    return tensor, original


def overlay_heatmap(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image."""
    original_resized = original.resize((heatmap.shape[1], heatmap.shape[0]))
    original_array = np.array(original_resized).astype(np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    blended = (1 - alpha) * original_array + alpha * heatmap_colored.astype(np.float32)
    blended = np.clip(blended, 0, 1)
    return blended


def generate_explanation(
    checkpoint_path: Path,
    image_path: Path,
    output_dir: Path,
    target_class: int | None = None,
) -> dict[str, Any]:
    """Generate a complete Grad-CAM explanation for one image.

    Returns a metadata dict with predicted class, confidence, and file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    input_tensor, original = preprocess_image(image_path)

    # Get prediction
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
    predicted_idx = int(np.argmax(probabilities))
    explain_class = target_class if target_class is not None else predicted_idx

    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(input_tensor.clone(), target_class=explain_class)
    gradcam.remove_hooks()

    # Save artifacts
    overlay = overlay_heatmap(original, heatmap)
    _save_explanation_figure(
        original,
        heatmap,
        overlay,
        predicted_class=CLASS_NAMES[predicted_idx],
        explained_class=CLASS_NAMES[explain_class],
        confidence=float(probabilities[predicted_idx]),
        output_path=output_dir / f"gradcam_{image_path.stem}.png",
    )

    return {
        "image": str(image_path),
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "explained_class": CLASS_NAMES[explain_class],
        "explained_index": explain_class,
        "probabilities": {
            name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probabilities)
        },
    }


def _save_explanation_figure(
    original: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    *,
    predicted_class: str,
    explained_class: str,
    confidence: float,
    output_path: Path,
) -> None:
    """Save a three-panel Grad-CAM explanation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    heatmap_display = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM: {explained_class}")
    axes[1].axis("off")
    fig.colorbar(heatmap_display, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Pred: {predicted_class}, {confidence:.1%})")
    axes[2].axis("off")

    fig.suptitle(
        f"Grad-CAM Explanation — Skin Cancer Classification\n"
        f"Predicted: {predicted_class} ({confidence:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone Grad-CAM generation."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to a self-describing .pth checkpoint"
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/responsible_ai"),
        help="Output directory for Grad-CAM artifacts",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class index to explain (default: predicted class)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        result = generate_explanation(
            args.checkpoint,
            args.image,
            args.output,
            args.target_class,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Grad-CAM error: {exc}", file=sys.stderr)
        return 1
    print(f"Predicted: {result['predicted_class']} ({result['confidence']:.4f})")
    print(f"Explained: {result['explained_class']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
