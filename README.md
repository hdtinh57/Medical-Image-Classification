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

Luồng branch: `feat/**` → `dev` → `prod`. `main` là default branch, giữ mốc ổn định và docs.

| Workflow | Chạy khi | Làm gì |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push lên `main`/`dev`/`prod`/`feat/**` · PR vào `main`/`dev`/`prod` | `ruff` lint + format · `pytest` (unit, data-quality, model-validation) + coverage · validate `docker-compose.yml`, Prometheus rules, Grafana dashboard |
| [`release-model.yml`](.github/workflows/release-model.yml) | PR vào `prod` · tag `model-v*` · chạy tay | Export checkpoint → ONNX → đẩy lên MinIO → verify round-trip bằng `onnxruntime` → upload `model.onnx` làm artifact. Job `triton-smoke` (tuỳ chọn) chạy Triton thật và gọi infer. |

`release-model.yml` gắn vào PR `dev` → `prod` vì đó là cổng cuối trước khi code được coi là chạy thật.

MinIO của nhóm chạy ở `localhost` nên runner GitHub không kết nối được. CI vì vậy dựng **MinIO ephemeral** trong job để kiểm chứng trọn pipeline export → upload → load; file ONNX được đẩy lên GitHub artifact để tải về dùng thật.

CI **không train model** (theo PLAN §1.3) — job `release-model` sinh checkpoint random đúng kiến trúc chỉ để kiểm tra đường ống.

> **Contract 9 lớp.** `training/model.py` là nguồn duy nhất định nghĩa `CLASS_NAMES` (9 lớp ISIC Kaggle), `NUM_CLASSES`, `IMAGE_SIZE`. `labels.txt` và `config.pbtxt` phải khớp; `validate_checkpoint()` chặn mọi checkpoint lệch class order hoặc input size, nên checkpoint sai contract sẽ fail ngay ở bước export chứ không lọt xuống Triton.

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

### Lưu ý UTF-8 trên Windows

Nếu terminal cũ không in được tiếng Việt, chỉ cần bật UTF-8 cho tiến trình Python hiện tại:

```powershell
# PowerShell
$env:PYTHONUTF8 = "1"
python training/ingest.py
```

```bash
# Git Bash hoặc terminal của Claude Code
PYTHONUTF8=1 python training/ingest.py
```

Các lệnh trên vẫn gọi Windows Python; Git Bash chỉ cung cấp cú pháp shell.

## Chạy EDA

EDA đọc trực tiếp dataset do `training/ingest.py` tạo, kiểm tra chất lượng toàn bộ ảnh,
phát hiện các nhóm **same-content** theo SHA-256 và ghi nhận overlap giữa raw `Train`/`Test`,
rồi lưu thống kê và biểu đồ tái lập. Raw Kaggle benchmark luôn được giữ nguyên: EDA không tự
xóa, deduplicate hoặc relabel ảnh. Hash chỉ được dùng để nhóm dữ liệu khi tách Validation nội bộ.

```powershell
python training/eda_skin_cancer.py
```

Output mặc định nằm tại:

```text
artifacts/eda/
├── dataset_index.csv
├── summary.json
├── class_distribution.png
├── split_class_distribution.png
├── sample_images.png
├── image_size_distribution.png
├── aspect_ratio_distribution.png
├── color_brightness_by_class.png
├── file_size_distribution.png
└── duplicate_report.csv        # chỉ có khi phát hiện duplicate
```

Tùy chọn thường dùng:

```powershell
# Chọn dataset/output khác
python training/eda_skin_cancer.py `
  --data-dir D:\datasets\skin-cancer `
  --output-dir D:\reports\skin-cancer-eda

# Thay đổi số ảnh mỗi lớp dùng cho thống kê pixel
python training/eda_skin_cancer.py --pixel-sample-per-class 50 --seed 42

# Hiển thị biểu đồ sau khi lưu
python training/eda_skin_cancer.py --show
```

`artifacts/` được ignore để output sinh tự động và dữ liệu nhạy cảm không bị đưa vào Git.

## Phát triển model 9 lớp

Pipeline model sử dụng class contract chung trong `training/model.py`. Kaggle `Test/` được giữ
nguyên; Validation được tách từ `Train/` bằng `StratifiedGroupKFold` và group theo SHA-256 để
cùng một nội dung ảnh không xuất hiện ở cả Train và Validation.

### 1. Tạo manifest Train/Validation/Test

```powershell
python -m training.dataset --n-splits 5 --fold 0 --seed 42
```

Artifacts mặc định:

```text
artifacts/training/
├── data_manifest.csv
└── data_manifest_summary.json
```

### 2. Smoke test CUDA

```powershell
python -m training.train `
  --epochs 1 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --run-name smoke
```

### 3. Train baseline ResNet18

```powershell
python -m training.train `
  --arch resnet18 `
  --epochs 10 `
  --batch-size 32 `
  --run-name resnet18-baseline
```

Model chính có thể chạy bằng ConvNeXtV2-Tiny:

```powershell
python -m training.train `
  --arch convnextv2_tiny.fcmae_ft_in22k_in1k `
  --epochs 20 `
  --batch-size 16 `
  --run-name convnextv2-tiny
```

Mỗi run lưu checkpoint, history, class mapping và MLflow artifacts dưới `artifacts/`.
MLflow vẫn được dùng mặc định; local tracking metadata nằm trong SQLite
`artifacts/mlflow/mlflow.db`, còn artifacts nằm trong `artifacts/mlflow/artifacts/`.
Model được chọn theo Validation macro F1; không dùng Kaggle Test để tune hyperparameter.

### 4. Evaluate checkpoint

```powershell
python -m training.evaluate `
  --checkpoint artifacts\training\runs\<run>\best_checkpoint.pth `
  --split test
```

Evaluation sinh `metrics.json`, `predictions.csv`, per-class report và confusion matrix.

### 5. Export ONNX/Triton local

```powershell
python -m training.export_triton `
  --local-checkpoint artifacts\training\runs\<run>\best_checkpoint.pth `
  --version 1
```

Lệnh trên chỉ export và verify PyTorch/ONNX Runtime parity. Chỉ thêm `--upload` khi chủ động
muốn publish model repository lên MinIO.
