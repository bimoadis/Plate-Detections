# ==================================================
# 🚗 Text Enhancement & OCR (Lokal, Non-Colab)
# ==================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog
import os
import math

# ==================================================
# 🔹 Pilih Gambar Plat Nomor dari Laptop
# ==================================================
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Pilih gambar plat nomor",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
)
if not file_path:
    raise Exception("❌ Tidak ada file yang dipilih.")
print(f"📂 File dipilih: {file_path}")

img = cv2.imread(file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)  # HD upscale

# ==================================================
# 🔹 Definisi Fungsi Sharpening
# ==================================================
def sharpen_basic(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def sharpen_strong(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def unsharp_mask(img, blur_size=(5,5), amount=1.5):
    blur = cv2.GaussianBlur(img, blur_size, 0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def clahe_sharpen(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return sharpen_basic(img_clahe)

def laplacian_sharpen(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(img - 0.7 * lap)
    return sharp

def high_boost(img, A=1.7):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    mask = cv2.subtract(img, blur)
    boosted = cv2.add(img, cv2.multiply(mask, np.array([A])))
    return np.clip(boosted, 0, 255).astype(np.uint8)

def edge_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img, 1.2, edges, 0.3, 0)

def bilateral_sharpen(img):
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    return cv2.addWeighted(img, 1.5, smooth, -0.5, 0)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def denoise_strong(img):
    dn = denoise(img)
    return sharpen_strong(dn)

def denoise_high_boost(img):
    dn = denoise(img)
    return high_boost(dn)

def clahe_unsharp(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return unsharp_mask(img_clahe)

# ==================================================
# 🔹 Kombinasi (Utama + Alternatif)
# ==================================================
def combo1(img):
    """CLAHE ➜ High Boost ➜ Edge Enhance"""
    step1 = clahe_sharpen(img)
    step2 = high_boost(step1)
    step3 = edge_enhance(step2)
    return step3

def combo2(img):
    """Denoise ➜ CLAHE ➜ Unsharp Mask"""
    step1 = denoise(img)
    step2 = clahe_unsharp(step1)
    return unsharp_mask(step2)

def combo3(img):
    """Laplacian ➜ Bilateral Sharpen ➜ Strong Sharpen"""
    step1 = laplacian_sharpen(img)
    step2 = bilateral_sharpen(step1)
    return sharpen_strong(step2)

def combo4(img):
    """Laplacian ➜ Denoise ➜ Basic Sharpen"""
    step1 = laplacian_sharpen(img)
    step2 = denoise(step1)
    return sharpen_basic(step2)

# 🔸 Alternatif tambahan hasil evaluasi visual:
def combo5(img):
    """CLAHE ➜ Bilateral ➜ Unsharp"""
    step1 = clahe_sharpen(img)
    step2 = bilateral_sharpen(step1)
    return unsharp_mask(step2)

def combo6(img):
    """High Boost ➜ Edge Enhance ➜ Denoise"""
    step1 = high_boost(img)
    step2 = edge_enhance(step1)
    return denoise(step2)

def combo7(img):
    """CLAHE ➜ Laplacian ➜ Strong Sharpen"""
    step1 = clahe_sharpen(img)
    step2 = laplacian_sharpen(step1)
    return sharpen_strong(step2)

def combo8(img):
    """CLAHE ➜ Laplacian ➜ Strong Sharpen"""
    step1 = edge_enhance(img)
    step2 = denoise(step1)
    return sharpen_strong(step2)

# ==================================================
# 🔹 Gabungkan Semua Hasil Sharpening
# ==================================================
results = {
    "Original": img,
    "Basic Sharpen": sharpen_basic(img),
    "Strong Sharpen": sharpen_strong(img),
    "Unsharp Mask": unsharp_mask(img),
    "CLAHE + Sharpen": clahe_sharpen(img),
    "Laplacian Sharpen": laplacian_sharpen(img),
    "High Boost": high_boost(img),
    "Edge Enhance": edge_enhance(img),
    "Bilateral Sharpen": bilateral_sharpen(img),
    "Denoise + Strong Sharpen": denoise_strong(img),
    "Denoise + High Boost": denoise_high_boost(img),
    "CLAHE + Unsharp": clahe_unsharp(img),
    "Combo 1 (CLAHE→HB→Edge)": combo1(img),
    "Combo 2 (Denoise→CLAHE→Unsharp)": combo2(img),
    "Combo 3 (Lap→Bilateral→Strong)": combo3(img),
    "Combo 4 (Lap→Denoise→Basic)": combo4(img),
    "Combo 5 (CLAHE→Bilateral→Unsharp)": combo5(img),
    "Combo 6 (HB→Edge→Denoise)": combo6(img),
    "Combo 7 (CLAHE→Lap→Strong)": combo7(img),
    "Combo 8 (Edge→Strong)": combo8(img)
}

# ==================================================
# 🔹 Load Custom OCR Model (Lokal)
# ==================================================
model_ocr = YOLO("OCRCUSTOMV4/content/runs/detect/train/weights/best.pt")

def run_custom_ocr(image):
    """Mendeteksi karakter dari gambar menggunakan YOLO OCR custom."""
    results_ocr = model_ocr.predict(source=image, conf=0.3, device="cpu")[0]
    boxes = results_ocr.boxes.xyxy.cpu().numpy()
    classes = results_ocr.boxes.cls.cpu().numpy()
    names = results_ocr.names
    detections = []
    for box2, cls2 in zip(boxes, classes):
        x1c, y1c, x2c, y2c = box2
        detections.append((x1c, names[int(cls2)]))
    detections = sorted(detections, key=lambda x: x[0])
    text = "".join([ch for _, ch in detections])
    return text.strip()

# ==================================================
# 🔹 Jalankan OCR untuk Setiap Hasil Sharpening
# ==================================================
ocr_results = {}
for name, image in results.items():
    try:
        text = run_custom_ocr(image)
    except Exception as e:
        text = f"(error: {e})"
    ocr_results[name] = text

# ==================================================
# 🔹 Tampilkan Semua Gambar + OCR
# ==================================================
total = len(results)
cols = 4
rows = math.ceil(total / cols)

plt.figure(figsize=(22, rows * 4.5))
for i, (title, image) in enumerate(results.items()):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.title(f"{title}\nOCR: {ocr_results[title]}", fontsize=9)
    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=1.0)
plt.show()
