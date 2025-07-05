from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
import cv2
import easyocr
import numpy as np
from datetime import datetime
import csv

# === 1. PILIH GAMBAR ===
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Pilih Gambar",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.png")]
)

if not image_path:
    print("[ERROR] Tidak ada gambar dipilih.")
    exit()

# === 2. PERSIAPAN FOLDER OUTPUT ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = os.path.join("hasil_deteksi_output", f"run_{timestamp}")
crop_folder = os.path.join(base_output_dir, "crop")
hd_folder = os.path.join(base_output_dir, "hd")
thresh_folder = os.path.join(base_output_dir, "threshold")
csv_file = os.path.join(base_output_dir, "hasil_deteksi.csv")

os.makedirs(crop_folder, exist_ok=True)
os.makedirs(hd_folder, exist_ok=True)
os.makedirs(thresh_folder, exist_ok=True)

# === 3. LOAD MODEL YOLO ===
model = YOLO("runs11s/detect/train/weights/best.pt")

# === 4. DETEKSI PLAT NOMOR ===
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)

# === 5. INISIALISASI OCR DAN CSV ===
reader = easyocr.Reader(['en'], gpu=cv2.cuda.getCudaEnabledDeviceCount() > 0)
img_cv = cv2.imread(image_path)

csv_header = ["Tanggal", "Nama File", "Hasil YOLO", "Confidence YOLO",
              "Path Crop", "Path HD", "Path Threshold",
              "OCR dari HD", "Conf HD", "OCR dari Threshold", "Conf Threshold"]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

# === 6. PROSES SETIAP HASIL DETEKSI ===
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf_yolo = float(box.conf[0])
    cls = int(box.cls[0])
    label = model.names[cls]

    # === Crop ===
    plate_crop = img_cv[y1:y2, x1:x2]
    crop_path = os.path.join(crop_folder, f"crop_{i+1}.png")
    cv2.imwrite(crop_path, plate_crop)

    # === Auto-Rotate ===
    # gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     all_cnt = np.vstack(contours)
    #     rect = cv2.minAreaRect(all_cnt)
    #     angle = rect[-1]
    #     if angle < -45:
    #         angle += 90
    #     elif angle > 45:
    #         angle -= 90

    #     (h, w) = plate_crop.shape[:2]
    #     M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    #     rotated = cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # else:
    #     rotated = plate_crop.copy()

    # === HD Enhancement ===
    upscaled = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    hd = cv2.filter2D(enhanced, -1, sharpen_kernel)
    hd_path = os.path.join(hd_folder, f"hd_{i+1}.png")
    cv2.imwrite(hd_path, hd)

    # === Thresholding ===
    gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_hd, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 41, 15)
    h_crop = thresh.shape[0]
    thresh = thresh[:int(h_crop * 0.75), :]
    thresh_path = os.path.join(thresh_folder, f"thresh_{i+1}.png")
    cv2.imwrite(thresh_path, thresh)

    # === OCR HD ===
    texts_hd, confs_hd = [], []
    for result in reader.readtext(hd, detail=1):
        if len(result) == 3:
            _, text, conf = result
            if conf > 0.5:
                texts_hd.append(text)
                confs_hd.append(conf)
    ocr_text_hd = ' '.join(texts_hd)
    conf_avg_hd = sum(confs_hd) / len(confs_hd) if confs_hd else 0

    # === OCR Threshold ===
    texts_th, confs_th = [], []
    for result in reader.readtext(thresh, detail=1):
        if len(result) == 3:
            _, text, conf = result
            if conf > 0.5:
                texts_th.append(text)
                confs_th.append(conf)
    ocr_text_th = ' '.join(texts_th)
    conf_avg_th = sum(confs_th) / len(confs_th) if confs_th else 0

    # === Tampilkan Semua ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
    plt.title(f"Crop #{i+1}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(hd, cv2.COLOR_BGR2RGB))
    plt.title("HD Enhanced")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title("Threshold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # === Print ke Terminal ===
    print(f"\n[INFO] Box #{i+1} | YOLO: {label} ({conf_yolo:.2f})")
    print(f"[HD OCR] {ocr_text_hd} (Conf: {conf_avg_hd:.2f})")
    print(f"[Threshold OCR] {ocr_text_th} (Conf: {conf_avg_th:.2f})")

    # === Simpan ke CSV ===
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            os.path.basename(image_path),
            label, f"{conf_yolo:.2f}",
            crop_path, hd_path, thresh_path,
            ocr_text_hd, f"{conf_avg_hd:.2f}",
            ocr_text_th, f"{conf_avg_th:.2f}"
        ])