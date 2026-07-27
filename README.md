# Medical Image Classification — MLOps Platform

[![CI](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/ci.yml)
[![Release model](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/release-model.yml/badge.svg)](https://github.com/hdtinh57/Medical-Image-Classification/actions/workflows/release-model.yml)

End-to-end MLOps platform phân loại tổn thương da (9 lớp ISIC) theo kiến trúc production-grade: training pipeline → model registry → Triton serving → FastAPI gateway → monitoring + continuous retraining.

> **Lưu ý:** Đây là prototype hỗ trợ quyết định lâm sàng, không phải thiết bị chẩn đoán y tế.

---

## Mục lục

- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Quick Start — Docker Stack](#quick-start--docker-stack)
- [Access URLs](#access-urls)
- [Cấu hình môi trường](#cấu-hình-môi-trường)
- [Dataset — Kaggle](#dataset--kaggle)
- [Training Pipeline](#training-pipeline)
- [Gateway API](#gateway-api)
- [Admin UI](#admin-ui)
- [Monitoring](#monitoring)
- [CI/CD & GitHub Actions](#cicd--github-actions)
- [Scheduled Retraining](#scheduled-retraining)
- [Phát triển local](#phát-triển-local)

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions                                │
│  ci.yml (lint/test) → train-model.yml → release-model.yml      │
│  scheduled-retrain.yml (weekly / drift trigger / manual)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ self-hosted runner (local Docker)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                              │
│  Kaggle → ingest → EDA → manifest → train → quality gate        │
│  → candidate bundle → ONNX export → MinIO model-registry        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Serving Stack (Docker Compose)                 │
│                                                                  │
│  Client → FastAPI Gateway :8080                                  │
│             ├── /predict  → Triton :8001 (gRPC)                 │
│             ├── /ui/      → Admin & Prediction UI                │
│             ├── /metrics  → Prometheus :9090                     │
│             └── /admin/*  → GitHub Actions trigger               │
│                                                                  │
│  MinIO :9000/9001  ←→  Triton :8000-8002  →  Prometheus :9090  │
│                                                   ↓              │
│                                             Grafana :3000        │
│  Loki :3100  ←  Promtail (log collector)                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start — Docker Stack

### Yêu cầu

- Docker Desktop đang chạy
- Các port chưa bị chiếm: `8080`, `9000`, `9001`, `8000–8002`, `9090`, `3000`, `3100`

### Khởi động

```powershell
# Copy và điền thông tin vào .env
cp .env.example .env   # hoặc tạo mới theo hướng dẫn bên dưới

# Build và khởi động toàn bộ stack
docker compose up -d --build

# Kiểm tra trạng thái
docker compose ps
```

Trạng thái mong đợi:

| Container | Trạng thái |
|---|---|
| `minio` | Running |
| `minio-init` | Exited (0) — đã tạo bucket xong |
| `triton` | Running — có thể trả 404 cho model nếu chưa có model |
| `gateway` | Running |
| `prometheus` | Running |
| `grafana` | Running |
| `loki` | Running |
| `promtail` | Running |
| `github-runner` | Running — đang lắng nghe job |

### Dừng stack

```powershell
docker compose down       # giữ lại volumes (MinIO, Grafana data)
docker compose down -v    # xóa toàn bộ volumes (mất dữ liệu local)
```

---

## Access URLs

| Service | URL | Thông tin đăng nhập |
|---|---|---|
| **Prediction & Admin UI** | http://localhost:8080/ui/ | — |
| **API Docs (Swagger)** | http://localhost:8080/docs | — |
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin` |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **Prometheus** | http://localhost:9090 | — |
| **Loki** | http://localhost:3100 | — |
| **Triton REST** | http://localhost:8000 | — |
| **Triton metrics** | http://localhost:8002/metrics | — |

---

## Cấu hình môi trường

Tạo file `.env` tại thư mục gốc project:

```dotenv
# Kaggle credentials (để tải dataset)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# MinIO (S3-compatible object store)
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# GitHub (để trigger retraining từ Admin UI)
GITHUB_TOKEN=ghp_your_personal_access_token
GITHUB_REPO=YourOrg/Medical-Image-Classification

# GitHub Self-hosted Runner registration token
# Lấy tại: Repo → Settings → Actions → Runners → New runner
GITHUB_RUNNER_TOKEN=your_runner_registration_token
```

> ⚠️ **Không commit `.env` lên Git.** File đã được thêm vào `.gitignore`.

---

## Dataset — Kaggle

Dataset: [Skin Cancer ISIC — 9 Classes](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data)

### 1. Cài dependency

```powershell
python -m pip install -r requirements.txt
```

### 2. Tải dataset

```powershell
python training/ingest.py
```

Data được lưu tại `data/raw/skin-cancer9-classesisic/`. Script tự bỏ qua nếu dữ liệu đã tồn tại.

```powershell
# Tải lại (force overwrite)
python training/ingest.py --force

# Lưu vào thư mục khác
python training/ingest.py --output D:\datasets\skin-cancer
```

---

## Training Pipeline

### 1. Chạy EDA

```powershell
python training/eda_skin_cancer.py
```

Output tại `artifacts/eda/`: phân phối class, duplicate report, thống kê ảnh.

### 2. Tạo manifest Train/Validation/Test

```powershell
python -m training.dataset --n-splits 5 --fold 0 --seed 42
```

Validation được tách từ Train bằng `StratifiedGroupKFold` + SHA-256 group để tránh data leakage.

### 3. Train model

```powershell
# Smoke test (kiểm tra CUDA)
python -m training.train --epochs 1 --max-train-batches 2 --max-val-batches 2 --run-name smoke

# Baseline ResNet18
python -m training.train --arch resnet18 --epochs 10 --batch-size 32 --run-name resnet18-baseline

# Model chính ConvNeXtV2-Tiny
python -m training.train --arch convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 20 --batch-size 16 --run-name convnextv2-tiny
```

MLflow tracking lưu tại `artifacts/mlflow/`. Model được chọn theo Validation macro F1.

### 4. Evaluate

```powershell
python -m training.evaluate --checkpoint artifacts\training\runs\<run>\best_checkpoint.pth --split test
```

### 5. Export ONNX / publish lên Triton

```powershell
# Export local (không upload)
python -m training.export_triton --local-checkpoint artifacts\training\runs\<run>\best_checkpoint.pth --version 1

# Export + upload lên MinIO
python -m training.export_triton --local-checkpoint artifacts\training\runs\<run>\best_checkpoint.pth --version 1 --upload
```

### 6. Pipeline end-to-end (candidate bundle)

```powershell
python -m training.pipeline \
  --version 2 \
  --arch resnet18 \
  --epochs 10 \
  --seed 42
```

Output tại `artifacts/pipeline/candidate/v2/`:

```text
best_checkpoint.pth        checkpoint self-describing
quality_gate.json          kết quả quality gate + metrics Validation
candidate.json             Git SHA, MLflow run ID, checkpoint SHA-256
checksums.json             SHA-256 toàn bộ artifacts
evaluations/val/           metrics/predictions/confusion matrix (Validation)
evaluations/test/          benchmark report (Test — không dùng để chọn model)
manifest_summary.json      split/leakage evidence
eda_summary.json           data-quality evidence
```

Quality gate mặc định: `accuracy ≥ 0.50`, `balanced_accuracy ≥ 0.50`, `macro_f1 ≥ 0.50`, không giảm macro F1 quá `0.02` so với champion.

### 7. Deploy thủ công (authorized operator only)

```powershell
$env:MINIO_ENDPOINT    = "http://localhost:9000"
$env:MINIO_ACCESS_KEY  = "minioadmin"
$env:MINIO_SECRET_KEY  = "minioadmin"
$env:TRITON_HTTP_URL   = "http://localhost:8000"

python -m training.deploy --bundle-dir artifacts\pipeline\candidate\v2
```

---

## Gateway API

FastAPI gateway chạy tại `http://localhost:8080`, kết nối Triton qua gRPC.

| Endpoint | Method | Mô tả |
|---|---|---|
| `/` | GET | Redirect về `/ui/` |
| `/ui/` | GET | Prediction UI (HTML) |
| `/ui/admin.html` | GET | Admin Dashboard (HTML) |
| `/predict` | POST | Upload ảnh da liễu → kết quả phân loại 9 lớp |
| `/feedback` | POST | Gửi ground-truth để theo dõi production accuracy |
| `/performance` | GET | Production accuracy từ feedback (rolling window) |
| `/drift` | GET | Drift detection (PSI + KS test) |
| `/health` | GET | Gateway + Triton readiness |
| `/metrics` | GET | Prometheus exposition format |
| `/admin/retrain` | POST | Trigger GitHub Actions retrain workflow |
| `/admin/pipeline-status` | GET | Trạng thái run mới nhất trên GitHub Actions |
| `/admin/data` | POST | Upload ZIP dataset mới cho lần retrain tiếp theo |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

### Ví dụ gọi predict

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@skin_lesion.jpg"
```

Response:

```json
{
  "prediction_id": "a1b2c3d4e5f6",
  "predicted_class": "melanoma",
  "confidence": 0.871234,
  "probabilities": { "melanoma": 0.871234, "nevus": 0.102345, ... },
  "latency_seconds": 0.0412
}
```

---

## Admin UI

Truy cập tại **http://localhost:8080/ui/admin.html**

| Tính năng | Mô tả |
|---|---|
| **Clinical Feedback** | Xem production accuracy và top confusion pairs từ feedback của bác sĩ |
| **Ingest New Data** | Upload file `.zip` dataset mới vào hàng chờ retrain |
| **Continuous Training** | Xem trạng thái pipeline GitHub Actions và trigger retraining thủ công |

Để **Launch Retraining** hoạt động, cần cấu hình `GITHUB_TOKEN` và `GITHUB_REPO` trong `.env`.

---

## Monitoring

### Grafana (http://localhost:3000)

Dashboard được provision tự động khi khởi động stack. Bao gồm:
- Triton inference latency, throughput, error rate
- Gateway request metrics
- Model confidence distribution

### Prometheus (http://localhost:9090)

Scrape Triton metrics mỗi 15 giây và gateway metrics mỗi 15 giây.

### Loki + Promtail

Promtail thu thập logs tất cả container Docker và đẩy vào Loki. Grafana đọc từ Loki để hiển thị log.

### Drift Detection

`GET /drift` thực hiện:
- **PSI (Population Stability Index)** trên phân phối class prediction
- **KS test (Kolmogorov-Smirnov)** trên confidence scores
- Cần ít nhất **10 predictions** gần nhất trong sliding window 500 requests

---

## CI/CD & GitHub Actions

| Workflow | Trigger | Vai trò |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | Push/PR vào `main`, `dev`, `feat/**` | Ruff lint, pytest, Docker Compose validation, Prometheus/Grafana config check |
| [`train-model.yml`](.github/workflows/train-model.yml) | Trusted push vào `main` hoặc manual | Self-hosted CUDA runner: ingest → EDA → train → quality gate → candidate artifact |
| [`release-model.yml`](.github/workflows/release-model.yml) | Successful train run trên `main` hoặc manual | Checksum verify + ONNX parity → GitHub Environment approval → deploy MinIO + Triton |
| [`scheduled-retrain.yml`](.github/workflows/scheduled-retrain.yml) | Weekly (Mon 03:00 UTC), manual, drift webhook | Drift check → retrain → quality gate → auto-deploy nếu pass |

### Cấu hình bắt buộc (GitHub)

1. **Repository secrets**: `KAGGLE_USERNAME`, `KAGGLE_KEY`
2. **Self-hosted runner** với labels `self-hosted, Linux, X64` (container `github-runner` trong Compose)
3. **GitHub Environment** `model-production` với required reviewers (cho release workflow)
4. **Environment secrets** (trong `model-production`): `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `TRITON_HTTP_URL`

---

## Scheduled Retraining

Workflow `scheduled-retrain.yml` tự động:

1. **Check drift** — query `/drift` endpoint; bỏ qua nếu stable và là scheduled run
2. **Retrain** — chạy `training.pipeline` với arch và epoch cấu hình sẵn
3. **Deploy** — tự deploy lên MinIO/Triton nếu quality gate pass
4. **Notify** — ghi summary vào GitHub Step Summary

Trigger thủ công từ Admin UI: **http://localhost:8080/ui/admin.html** → **Launch Retraining**.

### Cài đặt torch GPU cho runner

Runner sử dụng CUDA 12.4 (`cu124`), tương thích với CUDA driver ≤ 12.x:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Phát triển local

### Cài dependency dev

```powershell
python -m pip install -r requirements-dev.txt
```

### Kiểm tra code

```powershell
ruff check .
ruff format --check .
pytest
```

### Pre-commit hooks

```powershell
pre-commit install
pre-commit run --all-files
```

### Lưu ý UTF-8 trên Windows

```powershell
$env:PYTHONUTF8 = "1"
python training/ingest.py
```

### Build lại gateway sau khi thay đổi code

```powershell
docker compose up -d --build gateway
```

### Xem logs real-time

```powershell
docker compose logs -f gateway
docker compose logs -f triton
docker compose logs -f github-runner
```

---

## Cấu trúc thư mục

```text
Medical-Image-Classification/
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml
│   ├── train-model.yml
│   ├── release-model.yml
│   └── scheduled-retrain.yml
├── gateway/                    # FastAPI gateway
│   ├── main.py                 # API endpoints + UI mount
│   ├── triton_client.py        # Triton gRPC inference client
│   ├── gradcam.py              # Grad-CAM visualization
│   ├── metrics.py              # Prometheus custom metrics
│   ├── static/                 # Frontend UI (HTML/CSS/JS)
│   └── Dockerfile
├── training/                   # Training pipeline
│   ├── ingest.py               # Kaggle dataset download
│   ├── eda_skin_cancer.py      # EDA + data validation
│   ├── dataset.py              # Train/Val split (GroupKFold)
│   ├── train.py                # CUDA training loop + MLflow
│   ├── evaluate.py             # Evaluation + confusion matrix
│   ├── model.py                # Model architecture + checkpoint
│   ├── pipeline.py             # End-to-end candidate pipeline
│   ├── quality_gate.py         # Promotion criteria
│   ├── export_triton.py        # ONNX export + MinIO upload
│   ├── deploy.py               # Protected deployment transaction
│   └── model_registry.py      # MinIO registry operations
├── monitoring/                 # Observability config
│   ├── prometheus/
│   ├── grafana/
│   ├── loki/
│   ├── promtail/
│   ├── drift.py                # PSI + KS drift detection
│   └── feedback.py             # Feedback collector
├── model_repository/           # Triton model config template
├── tests/                      # Pytest test suite
├── docker-compose.yml          # Full stack orchestration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/test dependencies
├── .env                        # Local secrets (không commit)
├── ARCHITECTURE.md             # Technical architecture details
└── INFRA.md                    # Infra & deployment operations
```
