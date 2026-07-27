# FastAPI gateway — image nhẹ (không torch/timm; predict đi qua Triton).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Cài dependency trước để tận dụng layer cache.
COPY gateway/requirements.txt ./gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

# Copy source gateway and monitoring.
COPY gateway/ ./gateway/
COPY monitoring/ ./monitoring/

EXPOSE 8000

# Trong compose, TRITON_URL=triton:8000 (cùng network). Mặc định localhost để chạy tay.
ENV TRITON_URL=localhost:8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
