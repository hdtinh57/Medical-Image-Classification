"""Contract test cho Triton model artifacts.

Bắt lỗi nguy hiểm nhất của serving: `config.pbtxt`, `labels.txt` và hằng số trong
`export_triton.py` lệch nhau. Khi lệch, Triton VẪN load thành công nhưng trả sai
nhãn — sai im lặng, không exception nào để lần ra.

Test này CHỈ đọc file (ast + regex), KHÔNG import export_triton.py, để job test
trong CI không phải cài torch/timm.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "training" / "export_triton.py"
MODEL_DIR = ROOT / "model_repository" / "skin_classifier"
CONFIG_FILE = MODEL_DIR / "config.pbtxt"
LABELS_FILE = MODEL_DIR / "labels.txt"


def _module_constants(path: Path) -> dict[str, object]:
    """Đọc hằng số top-level của module Python mà không import nó.

    Tránh import vì export_triton.py kéo theo torch/timm/boto3.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass  # giá trị động (vd. os.getenv) — bỏ qua
    return constants


@pytest.fixture(scope="module")
def export_constants() -> dict[str, object]:
    assert EXPORT_SCRIPT.is_file(), f"Thiếu {EXPORT_SCRIPT}"
    return _module_constants(EXPORT_SCRIPT)


def _config_text() -> str:
    assert CONFIG_FILE.is_file(), f"Thiếu {CONFIG_FILE}"
    return CONFIG_FILE.read_text(encoding="utf-8")


def _labels() -> list[str]:
    assert LABELS_FILE.is_file(), f"Thiếu {LABELS_FILE}"
    return [ln.strip() for ln in LABELS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_num_classes_khop_giua_ba_noi(export_constants):
    """NUM_CLASSES == len(CLASSES) == số dòng labels.txt == output dims config.pbtxt.

    Model phục vụ là bản ConvNeXtV2 6 lớp (ACK/BCC/MEL/NEV/SCC/SEK) fine-tune trên
    ISIC 2019 — KHÁC số lớp của dataset 9 lớp trong data/. Bốn chỗ trên phải khớp
    nhau theo model đang serve, không theo dataset.
    """
    num_classes = export_constants["NUM_CLASSES"]
    classes = export_constants["CLASSES"]
    labels = _labels()

    assert (
        len(classes) == num_classes
    ), f"CLASSES có {len(classes)} phần tử nhưng NUM_CLASSES={num_classes}"
    assert (
        len(labels) == num_classes
    ), f"labels.txt có {len(labels)} dòng nhưng NUM_CLASSES={num_classes}"

    output_block = re.search(r"output\s*\[(.*?)\n\]", _config_text(), re.DOTALL)
    assert output_block, "Không tìm thấy khối output trong config.pbtxt"
    dims = re.search(r"dims:\s*\[\s*(\d+)\s*\]", output_block.group(1))
    assert dims, "Không đọc được output dims trong config.pbtxt"
    assert (
        int(dims.group(1)) == num_classes
    ), f"config.pbtxt output dims={dims.group(1)} nhưng NUM_CLASSES={num_classes}"


def test_labels_khop_dung_thu_tu_voi_classes(export_constants):
    """Sai thứ tự nhãn = dự đoán map sang sai bệnh, model vẫn 'chạy tốt'."""
    assert _labels() == list(export_constants["CLASSES"])


def test_ten_input_output_khop_export_va_config():
    """Lệch tên tensor -> Triton lỗi lúc infer, không phải lúc load."""
    source = EXPORT_SCRIPT.read_text(encoding="utf-8")
    assert 'input_names=["input"]' in source
    assert 'output_names=["logits"]' in source

    config = _config_text()
    assert 'name: "input"' in config
    assert 'name: "logits"' in config


def test_config_dung_ten_model_va_backend_cpu():
    config = _config_text()
    assert 'name: "skin_classifier"' in config
    assert 'platform: "onnxruntime_onnx"' in config
    assert "KIND_CPU" in config, "Đề chạy không GPU — instance_group phải là KIND_CPU"


def test_labels_khong_trung_lap():
    labels = _labels()
    assert len(labels) == len(set(labels)), f"labels.txt có nhãn trùng: {labels}"
