# ==================================================
# 🚗 Dataset Generator for Text Super-Resolution Network (TSRN)
# ==================================================
import os, cv2, random
import numpy as np
from tqdm import tqdm

# ==================================================
# 🔹 Folder sumber dan tujuan
# ==================================================
SOURCE_DIR = "dataset_hr"       # folder gambar plat asli hasil crop
DATASET_DIR = "dataset_tsrn_hd"  # folder output dataset
os.makedirs(DATASET_DIR, exist_ok=True)

# ==================================================
# 🔹 Proporsi dataset
# ==================================================
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ==================================================
# 🔹 Ukuran HR target (semua disamakan)
# ==================================================
HR_WIDTH = 128
HR_HEIGHT = 64

# ==================================================
# 🔹 Fungsi peningkatan kualitas gambar (HD Enhancement)
# ==================================================
def enhance_hd(img):
    """Tajamkan dan bersihkan gambar agar siap jadi HR."""
    # Resize ke ukuran target
    img = cv2.resize(img, (HR_WIDTH, HR_HEIGHT), interpolation=cv2.INTER_CUBIC)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Denoising ringan
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Tambah sedikit kontras
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    return img

# ==================================================
# 🔹 Fungsi membuat gambar LR dari HR
# ==================================================
def make_lr_image(img, scale=4):
    """Membuat gambar LR dengan simulasi blur/downscale lalu upscale lagi."""
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr_up

# ==================================================
# 🔹 Siapkan struktur folder
# ==================================================
for split in ["train", "val", "test"]:
    for sub in ["HR", "LR"]:
        os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

# ==================================================
# 🔹 Ambil dan acak gambar
# ==================================================
images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(images)

n_total = len(images)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
train_files = images[:n_train]
val_files = images[n_train:n_train + n_val]
test_files = images[n_train + n_val:]

# ==================================================
# 🔹 Fungsi utama untuk proses per split
# ==================================================
def process_split(files, split):
    for file in tqdm(files, desc=f"Processing {split}"):
        src_path = os.path.join(SOURCE_DIR, file)
        img = cv2.imread(src_path)
        if img is None:
            print(f"⚠️ Gagal baca {file}")
            continue

        # Buat HR (enhanced)
        img_hr = enhance_hd(img)

        # Buat LR dari HR
        img_lr = make_lr_image(img_hr, scale=4)

        # Simpan hasil
        cv2.imwrite(os.path.join(DATASET_DIR, split, "HR", file), img_hr)
        cv2.imwrite(os.path.join(DATASET_DIR, split, "LR", file), img_lr)

# ==================================================
# 🔹 Jalankan semua split
# ==================================================
process_split(train_files, "train")
process_split(val_files, "val")
process_split(test_files, "test")

print("\n✅ Dataset TSRN siap digunakan!")
print(f"📂 Lokasi: {os.path.abspath(DATASET_DIR)}")
