# INFRA — Serving & Observability (MinIO · Triton · Prometheus · Grafana)

Slice của **vai Serving/Infra + Observability**.
Nhiệm vụ: **KÉO model có sẵn từ MinIO → convert ONNX → Triton serve**, và dựng monitoring.
Bước train + đẩy `.pth` lên MinIO là của **team train** (không thuộc slice này).

## Phân chia bucket trên MinIO

| Bucket | Chứa | Ai ghi |
|---|---|---|
| `model-registry` | Checkpoint `.pth` theo version: `skin_classifier/<N>/best_weights.pth` | **Team train** đẩy lên |
| `models` | Triton repo (ONNX): `skin_classifier/config.pbtxt`, `labels.txt`, `<N>/model.onnx` | **export_triton.py** (slice này) |

Version ONNX trong Triton = version `.pth` trong registry (1:1).

## Luồng

```
[team train] ──push .pth──►  MinIO: model-registry/skin_classifier/<N>/best_weights.pth
                                          │
                                          │  export_triton.py  (KÉO xuống)
                                          ▼
                                   convert -> ONNX
                                          │  (ĐẨY lên)
                                          ▼
                             MinIO: models/skin_classifier/<N>/model.onnx
                                          │  (Triton đọc qua S3, poll 30s)
                                          ▼
   client ──► Triton :8000 (REST) ──► kết quả ;  :8002/metrics ──► Prometheus :9090 ──► Grafana :3000
```

## 0. Yêu cầu
- **Docker Desktop** đang chạy.
- **Python 3.10+** + deps để convert: `pip install torch timm onnx onnxruntime boto3`
- Image `nvcr.io/nvidia/tritonserver:24.08-py3` khá nặng (vài GB) — lần đầu pull sẽ lâu.

## 1. Bật cụm hạ tầng
```bash
docker compose up -d
docker compose ps      # minio/triton/prometheus/grafana 'running'; minio-init 'exited (0)' là đúng
```
Lúc này bucket `models` còn trống → Triton chạy `--exit-on-error=false`, đứng chờ (poll), chưa có model. Bình thường.

## 2. Đưa model vào Triton (nhiệm vụ chính của slice)

**Trường hợp thật:** team train đã đẩy `.pth` lên `model-registry`. Bạn chỉ chạy:
```bash
python training/export_triton.py            # lấy version mới nhất
# hoặc chỉ định: python training/export_triton.py --version 2
```

**Test độc lập** (team train chưa đẩy) — dùng file `.pth` local, bỏ qua bước pull:
```bash
python training/export_triton.py --local-weights model/best_weights.pth
```

Script sẽ: kéo `.pth` → convert `model.onnx` → đẩy `config.pbtxt` + `labels.txt` + `model.onnx` lên `s3://models/skin_classifier/`. Triton tự nạp trong ~30s.

## 3. Cổng & truy cập

| Service | URL | Ghi chú |
|---|---|---|
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |
| Triton REST | http://localhost:8000 | v2 inference protocol |
| Triton metrics | http://localhost:8002/metrics | Prometheus scrape |
| Prometheus | http://localhost:9090 | `/targets`, `/alerts` |
| Grafana | http://localhost:3000 | admin / admin → dashboard *Triton — Skin Classifier Serving* |

## 4. Kiểm tra (verify)
```bash
curl http://localhost:8000/v2/health/ready                    # server sẵn sàng -> 200
curl http://localhost:8000/v2/models/skin_classifier/ready    # model đã nạp -> 200
curl http://localhost:8000/v2/models/skin_classifier          # metadata input/output/labels
curl -s http://localhost:8002/metrics | grep nv_inference_request_success
```
Smoke-test suy luận (tensor ngẫu nhiên — tiền xử lý ảnh thật nằm ở gateway):
```bash
pip install "tritonclient[http]" numpy
```
```python
import numpy as np, tritonclient.http as http
c = http.InferenceServerClient(url="localhost:8000")
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
inp = http.InferInput("input", x.shape, "FP32"); inp.set_data_from_numpy(x)
out = c.infer("skin_classifier", [inp], outputs=[http.InferRequestedOutput("logits")])
print(out.as_numpy("logits").shape)   # -> (1, 6)
```

## 5. Cập nhật model mới (retrain)
1. Team train đẩy version mới (vd. `2`) lên `model-registry`.
2. Bạn chạy `python training/export_triton.py --version 2` (hoặc để mặc định lấy mới nhất).
3. Triton poll thấy `skin_classifier/2/` → nạp và phục vụ version mới. Không cần restart.

## 6. Dừng / dọn
```bash
docker compose down        # giữ dữ liệu
docker compose down -v      # xoá luôn volume MinIO/Grafana
```

## 7. Cách Triton đọc từ MinIO
```
--model-repository=s3://http://minio:9000/models
                        └scheme┘ └host:port┘ └bucket┘
```
Credentials lấy từ env `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (= user/pass MinIO trong compose). Bucket `models` là gốc model repository.

## 8. Troubleshooting

| Triệu chứng | Xử lý |
|---|---|
| `export_triton` báo "chưa có version trong registry" | Team train chưa đẩy `.pth`. Test tạm bằng `--local-weights`. |
| Triton `/v2/models/skin_classifier/ready` = 404 | Chưa export, hoặc chờ poll (30s). Xem `docker compose logs -f triton`. |
| Triton không kết nối MinIO | Sai `AWS_*` env / endpoint; kiểm tra bucket `models` trong MinIO console. |
| `pull access denied` khi kéo Triton | Image NGC nặng/mạng chậm; thử lại hoặc đổi tag mới hơn. |
| Prometheus `gateway` DOWN | Bình thường — gateway do team serving làm sau. `triton` phải UP. |
| ONNX export lỗi ở opset 17 | ConvNeXtV2 có lớp GRN; thử `OPSET=16` hoặc `18` trong `export_triton.py`. |

## 9. Phối hợp / việc liên quan
- **Team train:** viết bước đẩy `.pth` (versioned) lên `model-registry`.
- **Gateway (serving):** FastAPI nhận ảnh → tiền xử lý (resize 224 + normalize ImageNet) → gọi Triton → Grad-CAM. Đã chừa `job=gateway` trong Prometheus + panel Grafana.
- **CI/CD:** GitHub Actions có thể gọi `export_triton.py` để tự đẩy ONNX khi có model mới.
- **Đổi credentials** MinIO/Grafana trước khi để repo public.
