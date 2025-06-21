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
    print("❌ Tidak ada gambar dipilih.")
    exit()

# === 2. PERSIAPAN FOLDER OUTPUT ===
base_output_dir = "hasil_deteksi_output"
crop_folder = os.path.join(base_output_dir, "crop")
hd_folder = os.path.join(base_output_dir, "hd")
threshold_folder = os.path.join(base_output_dir, "threshold")
csv_file = os.path.join(base_output_dir, "hasil_deteksi.csv")

os.makedirs(crop_folder, exist_ok=True)
os.makedirs(hd_folder, exist_ok=True)
os.makedirs(threshold_folder, exist_ok=True)

# === 3. LOAD MODEL YOLO ===
model = YOLO("runs2/detect/train/weights/best.pt")

# === 4. DETEKSI PLAT NOMOR ===
results = model.predict(
    source=image_path, 
    conf=0.25, 
    save=True, 
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)

# === 5. TAMPILKAN HASIL DETEKSI (opsional) ===
save_dir = results[0].save_dir
pred_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.png'))]
if pred_images:
    img_path = os.path.join(save_dir, pred_images[0])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Hasil Deteksi YOLOv8")
    plt.show()

# === 6. INISIALISASI OCR DAN CSV ===
reader = easyocr.Reader(['en'], gpu=cv2.cuda.getCudaEnabledDeviceCount() > 0)
img_cv = cv2.imread(image_path)

csv_header = ["Tanggal", "Nama File", "Hasil YOLO", "Confidence YOLO",
              "Path Crop", "Path HD", "Path Threshold",
              "OCR dari HD", "Conf HD", "OCR dari Threshold", "Conf Threshold"]

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# === 7. PROSES SETIAP BOX DETEKSI ===
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf_yolo = float(box.conf[0])
    cls = int(box.cls[0])
    label = model.names[cls]

    # Crop
    plate_crop = img_cv[y1:y2, x1:x2]
    crop_path = os.path.join(crop_folder, f"crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png")
    cv2.imwrite(crop_path, plate_crop)

    # === HD Enhancement ===
    plate_upscaled = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(plate_upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    plate_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5,-1],
                             [0, -1, 0]])
    plate_hd = cv2.filter2D(plate_clahe, -1, kernel_sharp)
    hd_path = os.path.join(hd_folder, f"hd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png")
    cv2.imwrite(hd_path, plate_hd)

    # === Threshold
    plate_gray = cv2.cvtColor(plate_hd, cv2.COLOR_BGR2GRAY)
    plate_blur = cv2.GaussianBlur(plate_gray, (5, 5), 0)
    plate_thresh = cv2.adaptiveThreshold(
        plate_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 15)
    h = plate_thresh.shape[0]
    plate_thresh = plate_thresh[:int(h * 0.75), :]
    thresh_path = os.path.join(threshold_folder, f"thresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png")
    cv2.imwrite(thresh_path, plate_thresh)

    # === OCR dari HD
    texts_hd, confs_hd = [], []
    for res in reader.readtext(plate_hd, detail=1):
        if len(res) == 3:
            _, text, conf = res
            if conf > 0.5:
                texts_hd.append(text)
                confs_hd.append(conf)
    ocr_text_hd = ' '.join(texts_hd)
    conf_avg_hd = sum(confs_hd) / len(confs_hd) if confs_hd else 0

    # === OCR dari Threshold
    texts_t, confs_t = [], []
    for res in reader.readtext(plate_thresh, detail=1):
        if len(res) == 3:
            _, text, conf = res
            if conf > 0.5:
                texts_t.append(text)
                confs_t.append(conf)
    ocr_text_thresh = ' '.join(texts_t)
    conf_avg_thresh = sum(confs_t) / len(confs_t) if confs_t else 0

    # === TAMPILKAN GAMBAR (crop, hd, threshold)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
    plt.title(f"Crop #{i+1}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(plate_hd, cv2.COLOR_BGR2RGB))
    plt.title("HD Enhanced")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(plate_thresh, cmap='gray')
    plt.title("Thresholded")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # === LOG KE TERMINAL
    print(f"\n📌 Box #{i+1} | YOLO: {label} ({conf_yolo:.2f})")
    print(f"🧾 OCR (HD): {ocr_text_hd} (Conf: {conf_avg_hd:.2f})")
    print(f"🧾 OCR (Threshold): {ocr_text_thresh} (Conf: {conf_avg_thresh:.2f})")

    # === SIMPAN CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            os.path.basename(image_path),
            label, f"{conf_yolo:.2f}",
            crop_path, hd_path, thresh_path,
            ocr_text_hd, f"{conf_avg_hd:.2f}",
            ocr_text_thresh, f"{conf_avg_thresh:.2f}"
        ])
