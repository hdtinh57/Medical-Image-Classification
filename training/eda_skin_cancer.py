"""
EDA cho dataset: Skin Cancer 9 Classes ISIC
Nguồn: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

Cấu trúc dữ liệu điển hình sau khi tải & giải nén từ Kaggle:
    Skin cancer ISIC The International Skin Imaging Collaboration/
        Train/
            actinic keratosis/
            basal cell carcinoma/
            dermatofibroma/
            melanoma/
            nevus/
            pigmented benign keratosis/
            seborrheic keratosis/
            squamous cell carcinoma/
            vascular lesion/
        Test/
            (tương tự Train)

Nếu bạn tải bằng Kaggle API:
    pip install kaggle
    kaggle datasets download -d nodoubttome/skin-cancer9-classesisic
    unzip skin-cancer9-classesisic.zip -d skin_cancer_data

=> Chỉnh lại biến DATA_DIR bên dưới cho khớp với đường dẫn thực tế trên máy bạn.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN — SỬA LẠI CHO ĐÚNG VỚI MÁY BẠN
# ============================================================
# Đường dẫn được tính tương đối theo VỊ TRÍ FILE SCRIPT này (không phụ thuộc
# việc bạn chạy lệnh `python ...` từ thư mục nào).
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    SCRIPT_DIR
    / "raw"
    / "skin-cancer9-classesisic"
    / "Skin cancer ISIC The International Skin Imaging Collaboration"
)
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# 2. HÀM TIỆN ÍCH: QUÉT DỮ LIỆU THÀNH DATAFRAME
# ============================================================
def build_index(root_dir: Path, split_name: str) -> pd.DataFrame:
    """Quét toàn bộ ảnh trong root_dir (mỗi subfolder = 1 class) -> DataFrame."""
    records = []
    if not root_dir.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục: {root_dir}")
        return pd.DataFrame(columns=["filepath", "class", "split"])

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                records.append({"filepath": str(f), "class": class_dir.name, "split": split_name})
    return pd.DataFrame(records)


print("Đang quét thư mục dữ liệu...")
df_train = build_index(TRAIN_DIR, "train")
df_test = build_index(TEST_DIR, "test")
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"Tổng số ảnh: {len(df)}")
print(f"  - Train: {len(df_train)}")
print(f"  - Test : {len(df_test)}")

if df.empty:
    print("\n[LỖI] Không tìm thấy ảnh nào! Kiểm tra lại đường dẫn DATA_DIR/TRAIN_DIR/TEST_DIR.")
    print(f"  DATA_DIR  = {DATA_DIR.resolve()}  (tồn tại: {DATA_DIR.exists()})")
    print(f"  TRAIN_DIR = {TRAIN_DIR.resolve()} (tồn tại: {TRAIN_DIR.exists()})")
    print(f"  TEST_DIR  = {TEST_DIR.resolve()}  (tồn tại: {TEST_DIR.exists()})")
    if DATA_DIR.exists():
        print("  Các thư mục con thực tế bên trong DATA_DIR:")
        for p in DATA_DIR.iterdir():
            print("   -", p.name)
    raise SystemExit(
        "Dừng chương trình: hãy sửa lại biến DATA_DIR/TRAIN_DIR/TEST_DIR ở đầu script "
        "cho khớp với cấu trúc thư mục thật, rồi chạy lại."
    )

print(f"Số lớp (class): {df['class'].nunique()}")
print(df["class"].unique())


# ============================================================
# 3. PHÂN BỐ SỐ LƯỢNG ẢNH THEO LỚP
# ============================================================
def plot_class_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tổng thể
    order = df["class"].value_counts().index
    sns.countplot(data=df, y="class", order=order, ax=axes[0], palette="viridis")
    axes[0].set_title("Số lượng ảnh theo lớp (toàn bộ dataset)")
    axes[0].set_xlabel("Số lượng ảnh")
    axes[0].set_ylabel("Lớp bệnh")

    # Theo Train/Test
    ct = pd.crosstab(df["class"], df["split"]).astype(int)
    ct = ct.loc[order]
    if ct.shape[1] == 0:
        axes[1].text(0.5, 0.5, "Không có dữ liệu split", ha="center", va="center")
        axes[1].axis("off")
    else:
        ct.plot(kind="barh", stacked=True, ax=axes[1])
        axes[1].set_title("Phân bố Train / Test theo từng lớp")
        axes[1].set_xlabel("Số lượng ảnh")
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("class_distribution.png", bbox_inches="tight")
    plt.show()
    print("\nBảng số lượng ảnh theo lớp:")
    print(ct.assign(Total=ct.sum(axis=1)).sort_values("Total", ascending=False))


plot_class_distribution(df)


# ============================================================
# 4. KIỂM TRA MẤT CÂN BẰNG DỮ LIỆU (CLASS IMBALANCE)
# ============================================================
counts = df["class"].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(f"\nTỉ lệ mất cân bằng (lớp nhiều nhất / lớp ít nhất): {imbalance_ratio:.2f}")


# ============================================================
# 5. HIỂN THỊ ẢNH MẪU CHO MỖI LỚP
# ============================================================
def show_sample_images(df: pd.DataFrame, n_per_class: int = 3):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 3, len(classes) * 3))

    for i, cls in enumerate(classes):
        subset = df[df["class"] == cls].sample(
            min(n_per_class, len(df[df["class"] == cls])), random_state=42
        )
        for j, (_, row) in enumerate(subset.iterrows()):
            ax = axes[i, j] if n_per_class > 1 else axes[i]
            try:
                img = Image.open(row["filepath"])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Lỗi ảnh", ha="center", va="center")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, fontsize=9)
        # Đặt tiêu đề hàng
        axes[i, 0].set_title(cls, fontsize=10, loc="left") if n_per_class > 1 else None

    plt.suptitle("Ảnh mẫu cho từng lớp bệnh", fontsize=14)
    plt.tight_layout()
    plt.savefig("sample_images.png", bbox_inches="tight")
    plt.show()


show_sample_images(df, n_per_class=3)


# ============================================================
# 6. PHÂN TÍCH KÍCH THƯỚC ẢNH (WIDTH, HEIGHT, ASPECT RATIO)
# ============================================================
def analyze_image_sizes(df: pd.DataFrame, sample_size: int = 500):
    """Lấy mẫu ngẫu nhiên để phân tích kích thước (tránh đọc hết toàn bộ ảnh cho nhanh)."""
    sample_df = df.sample(min(sample_size, len(df)), random_state=42).copy()

    widths, heights, modes, corrupted = [], [], [], []
    for fp in sample_df["filepath"]:
        try:
            with Image.open(fp) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                modes.append(img.mode)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            modes.append(None)
            corrupted.append(fp)

    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["mode"] = modes
    sample_df["aspect_ratio"] = sample_df["width"] / sample_df["height"]

    print(f"\nSố ảnh lỗi/không đọc được trong mẫu: {len(corrupted)}")
    if corrupted:
        print("Ví dụ ảnh lỗi:", corrupted[:5])

    print("\nThống kê kích thước ảnh (mẫu {} ảnh):".format(len(sample_df)))
    print(sample_df[["width", "height", "aspect_ratio"]].describe())

    print("\nPhân bố định dạng màu (mode):")
    print(sample_df["mode"].value_counts())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(sample_df["width"].dropna(), bins=30, ax=axes[0], color="steelblue")
    axes[0].set_title("Phân bố chiều rộng (width)")

    sns.histplot(sample_df["height"].dropna(), bins=30, ax=axes[1], color="salmon")
    axes[1].set_title("Phân bố chiều cao (height)")

    sns.scatterplot(
        data=sample_df, x="width", y="height", hue="class", ax=axes[2], legend=False, alpha=0.6
    )
    axes[2].set_title("Tương quan width vs height")

    plt.tight_layout()
    plt.savefig("image_size_distribution.png", bbox_inches="tight")
    plt.show()

    return sample_df


size_df = analyze_image_sizes(df, sample_size=500)


# ============================================================
# 7. PHÂN TÍCH MÀU SẮC / ĐỘ SÁNG TRUNG BÌNH THEO LỚP
# ============================================================
def analyze_color_brightness(df: pd.DataFrame, sample_per_class: int = 30):
    records = []
    for cls in sorted(df["class"].unique()):
        subset = df[df["class"] == cls].sample(
            min(sample_per_class, len(df[df["class"] == cls])), random_state=42
        )
        for fp in subset["filepath"]:
            try:
                img = Image.open(fp).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                records.append(
                    {
                        "class": cls,
                        "mean_R": arr[:, :, 0].mean(),
                        "mean_G": arr[:, :, 1].mean(),
                        "mean_B": arr[:, :, 2].mean(),
                        "brightness": arr.mean(),
                    }
                )
            except Exception:
                continue

    color_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=color_df, x="class", y="brightness", ax=axes[0], palette="magma")
    axes[0].set_title("Độ sáng trung bình theo lớp")
    axes[0].tick_params(axis="x", rotation=75)

    color_melt = color_df.melt(
        id_vars=["class"],
        value_vars=["mean_R", "mean_G", "mean_B"],
        var_name="channel",
        value_name="value",
    )
    sns.boxplot(data=color_melt, x="class", y="value", hue="channel", ax=axes[1])
    axes[1].set_title("Giá trị kênh màu RGB trung bình theo lớp")
    axes[1].tick_params(axis="x", rotation=75)

    plt.tight_layout()
    plt.savefig("color_brightness_by_class.png", bbox_inches="tight")
    plt.show()

    return color_df


color_df = analyze_color_brightness(df, sample_per_class=30)


# ============================================================
# 8. KIỂM TRA ẢNH TRÙNG LẶP / DUNG LƯỢNG FILE (TÙY CHỌN)
# ============================================================
def analyze_file_sizes(df: pd.DataFrame):
    sizes_kb = []
    for fp in df["filepath"]:
        try:
            sizes_kb.append(os.path.getsize(fp) / 1024)
        except Exception:
            sizes_kb.append(np.nan)
    df = df.copy()
    df["size_kb"] = sizes_kb

    plt.figure(figsize=(8, 5))
    sns.histplot(df["size_kb"].dropna(), bins=40, color="teal")
    plt.title("Phân bố dung lượng file ảnh (KB)")
    plt.xlabel("Dung lượng (KB)")
    plt.tight_layout()
    plt.savefig("file_size_distribution.png", bbox_inches="tight")
    plt.show()

    print("\nThống kê dung lượng file:")
    print(df["size_kb"].describe())
    return df


df_with_size = analyze_file_sizes(df)


# ============================================================
# 9. TÓM TẮT KẾT QUẢ EDA
# ============================================================
print("\n" + "=" * 60)
print("TÓM TẮT EDA")
print("=" * 60)
print(f"- Tổng số ảnh: {len(df)}")
print(f"- Số lớp: {df['class'].nunique()}")
print(f"- Tỉ lệ mất cân bằng lớp: {imbalance_ratio:.2f}x")
print(
    f"- Kích thước ảnh trung bình (mẫu): "
    f"{size_df['width'].mean():.0f} x {size_df['height'].mean():.0f} px"
)
print("- Các biểu đồ đã lưu: class_distribution.png, sample_images.png,")
print("  image_size_distribution.png, color_brightness_by_class.png, file_size_distribution.png")
