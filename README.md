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

## CI/CD và MLOps automation

| Workflow | Trigger | Vai trò |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push/PR vào `main`, `dev`, `feat/**` | Ruff, pytest/coverage, Docker Compose, Prometheus và Grafana config validation. Không train model. |
| [`train-model.yml`](.github/workflows/train-model.yml) | trusted push vào `main` hoặc chạy tay | Chạy trên self-hosted CUDA runner: ingest → EDA → grouped manifest → train → Validation/Test evaluation → MLflow → Validation quality gate → candidate artifact. |
| [`release-model.yml`](.github/workflows/release-model.yml) | successful `Train candidate model` run trên `main`, hoặc manual source run ID | Xác minh checksum/gate + ONNX parity trên GitHub-hosted CPU; sau GitHub Environment approval, deploy immutable model version vào MinIO, chờ Triton ready và smoke inference `(1, 9)`. |

`ci.yml` là CI nhẹ. Training không chạy trên GitHub-hosted runner và chỉ được thực hiện bởi runner tin cậy có labels `self-hosted`, `Windows`, `X64`, `mlops-train`. Release production dùng runner `mlops-deploy` và GitHub Environment `model-production` với required reviewers.

### Cấu hình GitHub bắt buộc trước khi bật Continuous Training/Deployment

1. Đăng ký self-hosted training runner có CUDA với labels `self-hosted`, `Windows`, `X64`, `mlops-train`; cài Python, CUDA PyTorch và dependencies project.
2. Đăng ký self-hosted deploy runner có network access tới MinIO/Triton với labels `self-hosted`, `Windows`, `X64`, `mlops-deploy`.
3. Đặt repository secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`; không in hoặc commit các giá trị này.
4. Tạo Environment `model-production`, bật required reviewers và prevent self-review; đặt **environment secrets** `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `TRITON_HTTP_URL`.
5. Tùy chọn repository variables: `MLOPS_DATA_DIR` (persistent dataset path) và `MLFLOW_TRACKING_URI` (remote tracking server). Không cấu hình thì workflow dùng đường dẫn/artifact mặc định local trên runner.

Pipeline không overwrite model version; deploy yêu cầu version mới lớn hơn mọi version đã có trong registry/Triton để `latest` routing không quay về model cũ. Candidate chỉ được promotion khi Validation vượt quality gate mặc định `accuracy >= 0.50`, `balanced_accuracy >= 0.50`, `macro_f1 >= 0.50`, và không giảm macro F1 quá `0.02` so với champion. Kaggle Test chỉ phục vụ audit/report sau lựa chọn model, không gate promotion.

### Chạy kiểm tra trước khi push

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

### 6. Chạy candidate pipeline end-to-end tại local

Lệnh này dùng cùng orchestration với `train-model.yml`. Nó download/EDA trừ khi dữ liệu đã có,
train, evaluate Validation/Test, tạo `quality_gate.json` **theo Validation**, và sinh immutable
candidate bundle. Lệnh trả exit code `1` khi candidate bị gate từ chối nhưng vẫn giữ diagnostics.

```powershell
python -m training.pipeline `
  --version 2 `
  --arch resnet18 `
  --epochs 1 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --min-accuracy 0 `
  --min-balanced-accuracy 0 `
  --min-macro-f1 0
```

Output nằm tại `artifacts/pipeline/candidate/v2/`; gồm checkpoint self-describing,
`quality_gate.json`, Validation/Test reports, manifest/EDA evidence, checksums và provenance.
Không truyền MinIO credentials hoặc `--upload` ở bước này.

### 7. Manual recovery deploy (chỉ khi hạ tầng đã bật và credentials được cấp)

Bình thường `release-model.yml` xử lý deploy có approval. Lệnh local bên dưới chỉ dành cho operator
được ủy quyền, cần environment variables `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY`, `TRITON_HTTP_URL`; không fallback sang credential demo.

```powershell
python -m training.deploy --bundle-dir artifacts\pipeline\candidate\v2
```

Deploy kiểm tra checksum/gate, đăng ký checkpoint versioned, export/upload ONNX, chờ Triton ready tối
đa 60 giây và smoke-test logits `(1, 9)`. Nếu candidate Triton version lỗi, nó chỉ xóa prefix model
candidate vừa publish và giữ previous production version cùng audit metadata.
