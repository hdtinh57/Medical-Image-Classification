"""Serving step: KÉO checkpoint .pth từ MinIO (do team train đẩy lên) ->
convert ONNX -> ĐẨY ONNX vào Triton model repository trên MinIO (bucket 'models').
Triton đọc bucket 'models' qua S3 và tự nạp (poll 30s).

Đây là nhiệm vụ của vai Serving/Infra: CHỈ kéo model có sẵn xuống + convert + phục vụ.
KHÔNG train và KHÔNG đẩy .pth gốc (team train lo bước đó).

Version ONNX trong Triton == version .pth trong registry.

Cài đặt:
    pip install torch timm onnx onnxruntime boto3

Chạy (khi MinIO đang chạy):
    python training/export_triton.py                    # lấy version .pth mới nhất trong registry
    python training/export_triton.py --version 2         # đúng version 2
    python training/export_triton.py --local-weights model/best_weights.pth   # test, không cần registry
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch
import timm
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# --- Cấu hình MinIO (khớp docker-compose; có thể override qua env) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
REGISTRY_BUCKET = os.getenv("REGISTRY_BUCKET", "model-registry")  # .pth  (team train đẩy lên)
TRITON_BUCKET = os.getenv("TRITON_BUCKET", "models")              # .onnx (Triton đọc)

# --- Model ---
MODEL_NAME = "skin_classifier"
ARCH = "convnextv2_tiny.fcmae_ft_in22k_in1k"
NUM_CLASSES = 6
CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
OPSET = 17

ROOT = Path(__file__).resolve().parents[1]  # .../Medical-Image-Classification
MODEL_DIR = ROOT / "model_repository" / MODEL_NAME
CONFIG_FILE = MODEL_DIR / "config.pbtxt"
LABELS_FILE = MODEL_DIR / "labels.txt"


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"  + tạo bucket '{bucket}'")


def latest_version(s3, bucket: str, model_name: str) -> int:
    """Version .pth mới nhất (int) trong registry."""
    prefix = f"{model_name}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    versions = []
    for cp in resp.get("CommonPrefixes", []):
        part = cp["Prefix"][len(prefix):].strip("/")
        if part.isdigit():
            versions.append(int(part))
    if not versions:
        raise RuntimeError(
            f"Chưa có version nào trong s3://{bucket}/{model_name}/ — team train chưa đẩy .pth?"
        )
    return max(versions)


def load_state_dict(path):
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:  # checkpoint bọc thêm 1 lớp
        state = state["state_dict"]
    return state


def export_onnx(weights_path, out_onnx: Path) -> torch.Tensor:
    model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(load_state_dict(weights_path))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    return dummy


def verify_onnx(onnx_path: Path, dummy: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] (bỏ qua verify: chưa cài onnxruntime)")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    assert out.shape == (1, NUM_CLASSES), out.shape
    print(f"[export] onnxruntime OK, output shape = {out.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None, help="version .pth trong registry; mặc định mới nhất")
    ap.add_argument("--local-weights", default=None, help="dùng .pth local, bỏ qua bước pull (để test)")
    args = ap.parse_args()

    s3 = s3_client()
    ensure_bucket(s3, TRITON_BUCKET)

    with tempfile.TemporaryDirectory() as tmp:
        # 1) KÉO .pth (từ registry, hoặc dùng local để test)
        if args.local_weights:
            version = args.version or 1
            weights_path = args.local_weights
            print(f"[export] weights local: {weights_path} (version {version})")
        else:
            version = args.version or latest_version(s3, REGISTRY_BUCKET, MODEL_NAME)
            weights_path = Path(tmp) / "best_weights.pth"
            key = f"{MODEL_NAME}/{version}/best_weights.pth"
            print(f"[export] kéo weights: s3://{REGISTRY_BUCKET}/{key}")
            s3.download_file(REGISTRY_BUCKET, key, str(weights_path))

        # 2) CONVERT sang ONNX vào đúng thư mục version của Triton
        out_onnx = MODEL_DIR / str(version) / "model.onnx"
        print(f"[export] convert ONNX -> {out_onnx}")
        dummy = export_onnx(weights_path, out_onnx)
        LABELS_FILE.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        verify_onnx(out_onnx, dummy)

    # 3) ĐẨY Triton repo (config + labels + onnx) lên bucket 'models'
    print(f"[export] đẩy Triton repo -> s3://{TRITON_BUCKET}/{MODEL_NAME}/")
    for local, key in [
        (CONFIG_FILE, f"{MODEL_NAME}/config.pbtxt"),
        (LABELS_FILE, f"{MODEL_NAME}/labels.txt"),
        (out_onnx, f"{MODEL_NAME}/{version}/model.onnx"),
    ]:
        s3.upload_file(str(local), TRITON_BUCKET, key)
        print(f"  ↑ s3://{TRITON_BUCKET}/{key}")

    print(f"[export] DONE — Triton sẽ tự nạp version {version} trong ~30s (poll).")


if __name__ == "__main__":
    main()
