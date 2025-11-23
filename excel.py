# ==================================================
# 🚗 Text Enhancement Batch (Folder Mode)
# ==================================================
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# ==================================================
# 🔹 Pilih Folder Input & Siapkan Output Folder
# ==================================================
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="📁 Pilih folder gambar plat nomor")
if not folder_path:
    raise Exception("❌ Tidak ada folder yang dipilih.")
print(f"📂 Folder input: {folder_path}")

output_dir = os.path.join(folder_path, "output_sharpened")
os.makedirs(output_dir, exist_ok=True)
print(f"💾 Folder output: {output_dir}")

# ==================================================
# 🔹 Fungsi Edge Enhance dan Combo8 (versi perbaikan HR)
# ==================================================
def edge_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img, 1.2, edges, 0.3, 0)

def combo8(img):
    """Edge Enhance ➜ Denoise ➜ Strong Sharpen ➜ Refine HR"""
    # 1️⃣ Edge Enhancement
    step1 = edge_enhance(img)

    # 2️⃣ Denoise ringan
    step2 = cv2.fastNlMeansDenoisingColored(step1, None, 5, 5, 7, 21)

    # 3️⃣ Sharpening kuat tapi seimbang
    kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]
    ]) / 8.0
    step3 = cv2.filter2D(step2, -1, kernel)

    # 4️⃣ CLAHE untuk perbaikan kontras
    lab = cv2.cvtColor(step3, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_hr = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # 5️⃣ Detail Enhancement
    final_hr = cv2.detailEnhance(final_hr, sigma_s=12, sigma_r=0.25)

    return final_hr

# ==================================================
# 🔹 Proses Semua Gambar dalam Folder
# ==================================================
allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(allowed_ext):
        file_path = os.path.join(folder_path, filename)
        print(f"🔧 Memproses: {filename}")

        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️ Gagal membaca: {filename}")
            continue

        # Ubah ke RGB dan perbesar resolusi (HD)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # Terapkan peningkatan (Combo8)
        sharpened = combo8(img)

        # Simpan hasil
        save_path = os.path.join(output_dir, f"sharp_{filename}")
        cv2.imwrite(save_path, cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))
        print(f"✅ Disimpan: {save_path}")

print("\n🎉 Semua gambar berhasil ditingkatkan dan disimpan di:", output_dir)
