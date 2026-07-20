# Medical Image Classification

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
