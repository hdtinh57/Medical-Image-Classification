"""Download the Skin Cancer ISIC dataset from Kaggle when it is missing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "nodoubttome/skin-cancer9-classesisic"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "skin-cancer9-classesisic"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARKER_FILE = ".download_complete.json"


def count_images(directory: Path) -> int:
    """Count image files recursively in a directory."""
    if not directory.is_dir():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_marker(directory: Path, image_count: int) -> None:
    """Record a successful download for later checks."""
    marker = {
        "dataset": DATASET_ID,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "image_count": image_count,
    }
    (directory / MARKER_FILE).write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )


def load_kaggle_api():
    """Import and authenticate the Kaggle API with a helpful error message."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài package 'kaggle'. Chạy: pip install -r requirements.txt"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Không xác thực được Kaggle. Hãy đặt kaggle.json tại "
            "%USERPROFILE%\\.kaggle\\kaggle.json hoặc cấu hình "
            "KAGGLE_USERNAME và KAGGLE_KEY."
        ) from exc

    return api


def download_dataset(output_dir: Path, force: bool = False) -> None:
    """Download and atomically install the dataset into output_dir."""
    existing_images = count_images(output_dir)
    if existing_images > 0 and not force:
        print(
            f"Dataset đã tồn tại tại: {output_dir} "
            f"({existing_images:,} ảnh). Bỏ qua tải lại."
        )
        return

    if output_dir.exists() and not force:
        raise RuntimeError(
            f"Thư mục {output_dir} đã tồn tại nhưng không chứa ảnh. "
            "Xóa thư mục hoặc chạy lại với --force."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.download-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False)

    try:
        print(f"Đang tải Kaggle dataset: {DATASET_ID}")
        print(f"Thư mục đích: {output_dir}")

        api = load_kaggle_api()
        api.dataset_download_files(
            DATASET_ID,
            path=str(temp_dir),
            unzip=True,
            quiet=False,
        )

        downloaded_images = count_images(temp_dir)
        if downloaded_images == 0:
            raise RuntimeError(
                "Tải xong nhưng không tìm thấy file ảnh; không thay đổi data hiện tại."
            )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
        write_marker(output_dir, downloaded_images)

        print(f"Hoàn tất: {downloaded_images:,} ảnh tại {output_dir}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tự động tải và giải nén dataset Skin Cancer ISIC từ Kaggle "
            "nếu data chưa tồn tại."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu dataset (mặc định: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại và thay thế dataset hiện có sau khi tải thành công.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()

    try:
        download_dataset(output_dir, force=args.force)
    except RuntimeError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Lỗi không mong đợi: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
