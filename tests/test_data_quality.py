"""Data-quality tests cho pipeline ingest (phần D của rubric).

Hai nhóm:
1. Test logic ingest thuần (count_images / write_marker) — chạy được mọi lúc,
   dùng tmp_path, không cần dataset thật.
2. Test chất lượng dataset thật — chỉ chạy khi data/ đã tải về, tự skip trong CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.ingest import (
    DEFAULT_OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    MARKER_FILE,
    count_images,
    write_marker,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------
# 1. Logic ingest — luôn chạy
# --------------------------------------------------------------------------


def test_count_images_tra_ve_0_khi_thu_muc_khong_ton_tai(tmp_path):
    assert count_images(tmp_path / "khong-co-that") == 0


def test_count_images_dem_de_quy_va_bo_qua_file_khong_phai_anh(tmp_path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "nested" / "b.PNG")  # đuôi viết hoa vẫn phải tính
    _touch(tmp_path / "nested" / "sau" / "c.jpeg")
    _touch(tmp_path / "readme.txt")  # không phải ảnh
    _touch(tmp_path / "metadata.csv")  # không phải ảnh

    assert count_images(tmp_path) == 3


@pytest.mark.parametrize("ext", sorted(IMAGE_EXTENSIONS))
def test_count_images_nhan_moi_duoi_anh_khai_bao(tmp_path, ext):
    _touch(tmp_path / f"anh{ext}")
    assert count_images(tmp_path) == 1


def test_write_marker_ghi_json_doc_lai_duoc(tmp_path):
    write_marker(tmp_path, image_count=1234)

    marker = json.loads((tmp_path / MARKER_FILE).read_text(encoding="utf-8"))
    assert marker["image_count"] == 1234
    assert marker["dataset"]
    assert marker["downloaded_at"]


# --------------------------------------------------------------------------
# 2. Chất lượng dataset thật — skip khi chưa có data (CI luôn skip)
# --------------------------------------------------------------------------

requires_dataset = pytest.mark.skipif(
    count_images(DEFAULT_OUTPUT_DIR) == 0,
    reason="Chưa tải dataset về data/ — chạy `python training/ingest.py` trước",
)


def class_distribution(dataset_dir: Path) -> dict[str, int]:
    """Số ảnh mỗi lớp; mỗi thư mục con cấp 1 = một lớp."""
    return {sub.name: count_images(sub) for sub in sorted(dataset_dir.iterdir()) if sub.is_dir()}


# TODO(bạn viết): định nghĩa ngưỡng "dataset đủ tốt để train".
# Đây là quyết định domain, không phải code máy móc — xem phần giải thích ở chat.
#
# def check_dataset_quality(distribution: dict[str, int]) -> list[str]:
#     """Trả về danh sách vi phạm; list rỗng = dataset đạt.
#
#     Gợi ý cân nhắc:
#       - số ảnh tối thiểu mỗi lớp (train nổi không?)
#       - tỉ lệ mất cân bằng tối đa (lớp lớn nhất / lớp nhỏ nhất)
#       - lớp ác tính MEL/BCC/SCC có cần ngưỡng chặt hơn lớp lành tính không?
#     """
#     violations: list[str] = []
#     ...
#     return violations


@requires_dataset
def test_dataset_co_du_lop_va_khong_co_lop_rong():
    distribution = class_distribution(DEFAULT_OUTPUT_DIR)

    assert distribution, f"Không tìm thấy thư mục lớp nào trong {DEFAULT_OUTPUT_DIR}"
    empty = [name for name, count in distribution.items() if count == 0]
    assert not empty, f"Lớp rỗng: {empty}"


# CỐ Ý không assert labels.txt khớp tên thư mục dataset: model đang serve là bản
# 6 lớp (ACK/BCC/MEL/NEV/SCC/SEK) fine-tune trên ISIC 2019, còn dataset tải về là
# bộ 9 lớp. Hai thứ khác nhau có chủ đích — xem README §CI/CD.
