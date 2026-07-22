"""Export a self-describing training checkpoint to an ONNX Triton repository."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import torch
from botocore.client import Config
from botocore.exceptions import ClientError

from training.model import CLASS_NAMES, MODEL_NAME, NUM_CLASSES, load_model_from_checkpoint

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
REGISTRY_BUCKET = os.getenv("REGISTRY_BUCKET", "model-registry")
TRITON_BUCKET = os.getenv("TRITON_BUCKET", "models")
OPSET = 18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model_repository" / MODEL_NAME
CONFIG_FILE = MODEL_DIR / "config.pbtxt"
LABELS_FILE = MODEL_DIR / "labels.txt"


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for local export and optional MinIO publication."""

    version: int | None = None
    local_checkpoint: Path | None = None
    upload: bool = False


def parse_args() -> ExportConfig:
    """Parse checkpoint export options."""
    parser = argparse.ArgumentParser(description="Export a 9-class checkpoint to ONNX.")
    parser.add_argument(
        "--version", type=int, help="Checkpoint/model version; mặc định latest hoặc 1."
    )
    parser.add_argument(
        "--local-checkpoint",
        "--local-weights",
        dest="local_checkpoint",
        type=Path,
        help="Dùng checkpoint local thay vì tải từ model-registry.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload Triton repository lên MinIO sau khi verify local.",
    )
    args = parser.parse_args()
    return ExportConfig(
        version=args.version,
        local_checkpoint=(
            args.local_checkpoint.expanduser().resolve() if args.local_checkpoint else None
        ),
        upload=args.upload,
    )


def s3_client() -> Any:
    """Create one MinIO-compatible S3 client."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create an output bucket only when it does not already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        print(f"  + tạo bucket '{bucket}'")


def latest_version(client: Any, bucket: str, model_name: str) -> int:
    """Return the highest integer checkpoint version in a registry prefix."""
    prefix = f"{model_name}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    versions = []
    for common_prefix in response.get("CommonPrefixes", []):
        part = common_prefix["Prefix"][len(prefix) :].strip("/")
        if part.isdigit():
            versions.append(int(part))
    if not versions:
        raise RuntimeError(f"Không có checkpoint version trong s3://{bucket}/{model_name}/")
    return max(versions)


def resolve_checkpoint(config: ExportConfig, temporary_dir: Path) -> tuple[Path, int]:
    """Resolve a local checkpoint or download one explicit registry version."""
    if config.local_checkpoint is not None:
        if not config.local_checkpoint.is_file():
            raise FileNotFoundError(config.local_checkpoint)
        return config.local_checkpoint, config.version or 1

    client = s3_client()
    version = config.version or latest_version(client, REGISTRY_BUCKET, MODEL_NAME)
    checkpoint_path = temporary_dir / "best_checkpoint.pth"
    key = f"{MODEL_NAME}/{version}/best_checkpoint.pth"
    print(f"[export] tải checkpoint: s3://{REGISTRY_BUCKET}/{key}")
    client.download_file(REGISTRY_BUCKET, key, str(checkpoint_path))
    return checkpoint_path, version


def export_onnx(checkpoint_path: Path, output_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Export one checkpoint and return the parity-test input and PyTorch logits."""
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    image_size = int(checkpoint["image_size"])
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        pytorch_logits = model(dummy).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dimension = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: batch_dimension},),
        opset_version=OPSET,
        dynamo=True,
    )
    return dummy, pytorch_logits


def verify_onnx(
    onnx_path: Path,
    dummy: torch.Tensor,
    pytorch_logits: np.ndarray,
) -> float:
    """Verify output contract and numerical parity with ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": dummy.numpy()})[0]
    if onnx_logits.shape != (1, NUM_CLASSES):
        raise RuntimeError(f"ONNX output shape không hợp lệ: {onnx_logits.shape}")
    np.testing.assert_allclose(onnx_logits, pytorch_logits, rtol=1e-3, atol=1e-4)
    max_difference = float(np.max(np.abs(onnx_logits - pytorch_logits)))
    print(f"[export] ONNX Runtime parity OK; max_abs_diff={max_difference:.8f}")
    return max_difference


def write_labels() -> None:
    """Write the canonical labels file consumed by Triton."""
    LABELS_FILE.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")


def upload_repository(version: int, onnx_path: Path) -> None:
    """Upload verified Triton repository files to MinIO."""
    client = s3_client()
    ensure_bucket(client, TRITON_BUCKET)
    uploads = [
        (CONFIG_FILE, f"{MODEL_NAME}/config.pbtxt"),
        (LABELS_FILE, f"{MODEL_NAME}/labels.txt"),
        (onnx_path, f"{MODEL_NAME}/{version}/model.onnx"),
    ]
    external_data = onnx_path.with_name(f"{onnx_path.name}.data")
    if external_data.exists():
        uploads.append((external_data, f"{MODEL_NAME}/{version}/{external_data.name}"))

    print(f"[export] upload Triton repository -> s3://{TRITON_BUCKET}/{MODEL_NAME}/")
    for local_path, key in uploads:
        client.upload_file(str(local_path), TRITON_BUCKET, key)
        print(f"  ↑ s3://{TRITON_BUCKET}/{key}")


def run_export(config: ExportConfig) -> Path:
    """Resolve, export, verify, and optionally upload one checkpoint."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path, version = resolve_checkpoint(config, Path(temporary_directory))
        output_path = MODEL_DIR / str(version) / "model.onnx"
        print(f"[export] checkpoint: {checkpoint_path}")
        print(f"[export] ONNX output: {output_path}")
        dummy, pytorch_logits = export_onnx(checkpoint_path, output_path)
        write_labels()
        verify_onnx(output_path, dummy, pytorch_logits)

    if config.upload:
        upload_repository(version, output_path)
    else:
        print("[export] local verify hoàn tất; bỏ qua MinIO vì chưa truyền --upload.")
    return output_path


def main() -> int:
    """CLI entrypoint."""
    try:
        output_path = run_export(parse_args())
    except (AssertionError, ClientError, OSError, RuntimeError, ValueError) as exc:
        print(f"[export] lỗi: {exc}")
        return 1
    print(f"[export] DONE: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
