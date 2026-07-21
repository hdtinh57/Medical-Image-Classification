# Medical Image Classification

[![CI](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/ci.yml)
[![Release model](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/release-model.yml/badge.svg)](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/release-model.yml)

## Tải dataset tự động từ Kaggle

Dataset: [Skin Cancer ISIC — 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data)

### 1. Cài dependency

```powershell
python -m pip install -r requirements.txt
```

### 2. Cấu hình Kaggle API

1. Mở [Kaggle Settings](https://www.kaggle.com/settings), tìm mục **API** và chọn **Create New Token** để tải `kaggle.json`.
2. Trên Windows, đặt file token vào `%USERPROFILE%\.kaggle\kaggle.json`:

```powershell
New-Item -ItemType Directory -Force "$HOME\.kaggle" | Out-Null
Copy-Item "$HOME\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json"
```

Hoặc tạo file `.env` tại thư mục gốc project:

```dotenv
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Script tự đọc file này; biến môi trường của hệ điều hành hoặc CI/CD sẽ được ưu tiên nếu đã tồn tại.

> Không commit `.env` hoặc `kaggle.json` lên Git. Các file này chứa credential cá nhân và đã được thêm vào `.gitignore`.

### 3. Tải và giải nén dataset

```powershell
python training/ingest.py
```

Mặc định data được lưu tại:

```text
data/raw/skin-cancer9-classesisic/
```

Script sẽ:

- bỏ qua việc tải nếu thư mục đã chứa ảnh;
- tải vào thư mục tạm trước để tránh để lại dataset lỗi/dở dang;
- tự giải nén và tạo `.download_complete.json` sau khi tải thành công;
- không đưa data vào Git vì toàn bộ `data/` đã được ignore.

Các tùy chọn:

```powershell
# Chọn thư mục khác
python training/ingest.py --output D:\datasets\skin-cancer

# Tải lại và thay thế data hiện tại sau khi bản mới tải thành công
python training/ingest.py --force
```

## CI/CD (GitHub Actions)

| Workflow | Chạy khi | Làm gì |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | mọi push + PR vào `main`/`dev` | `ruff` lint + format · `pytest` (unit, data-quality, model-validation) + coverage · validate `docker-compose.yml`, Prometheus rules, Grafana dashboard |
| [`release-model.yml`](.github/workflows/release-model.yml) | PR vào `main` · tag `model-v*` · chạy tay | Export `.pth` → ONNX → đẩy lên MinIO → verify round-trip bằng `onnxruntime` → upload `model.onnx` làm artifact. Job `triton-smoke` (tuỳ chọn) chạy Triton thật và gọi infer. |

MinIO của nhóm chạy ở `localhost` nên runner GitHub không kết nối được. CI vì vậy dựng **MinIO ephemeral** trong job để kiểm chứng trọn pipeline export → upload → load; file ONNX được đẩy lên GitHub artifact để tải về dùng thật.

CI **không train model** (theo PLAN §1.3) — job `release-model` sinh checkpoint random đúng kiến trúc chỉ để kiểm tra đường ống.

> **Số lớp: model 6 ≠ dataset 9.** Model đang serve là [`conan17970/convnextv2-skin-cancer-isic2019`](https://huggingface.co/conan17970/convnextv2-skin-cancer-isic2019) — ConvNeXtV2-tiny fine-tune trên ISIC 2019, **6 lớp** `ACK · BCC · MEL · NEV · SCC · SEK`. Dataset `ingest.py` tải về là bộ Kaggle **9 lớp**, dùng cho EDA và train lại sau này. `test_model_validation.py` canh cho `NUM_CLASSES`, `CLASSES`, `labels.txt` và `config.pbtxt` khớp nhau **theo model đang serve**, không theo dataset.

### Chạy trước khi push (giống hệt CI)

```powershell
python -m pip install -r requirements-dev.txt

ruff check .
ruff format --check .
pytest
```

Hoặc để `pre-commit` tự lo phần lint:

```powershell
pre-commit install
pre-commit run --all-files
```
