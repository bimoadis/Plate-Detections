from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
import cv2
import easyocr
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

# === 2. LOAD MODEL ===
model = YOLO("runs2/detect/train/weights/best.pt")

# === 3. PREDIKSI DENGAN YOLO ===
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)

# === 4. TAMPILKAN HASIL DETEKSI ===
save_dir = results[0].save_dir
pred_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.png'))]

if pred_images:
    img_path = os.path.join(save_dir, pred_images[0])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Hasil Deteksi YOLOv8")
    plt.show()
else:
    print("❌ Tidak ada gambar hasil deteksi ditemukan.")
    exit()

# === 5. OCR (EasyOCR + Preprocessing) ===
reader = easyocr.Reader(['en'], gpu=cv2.cuda.getCudaEnabledDeviceCount() > 0)
img_cv = cv2.imread(image_path)

print("\n📌 Hasil Pembacaan Teks Plat (Setelah Preprocessing):\n")

# Setup output CSV
output_dir = "output_csv_log"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "hasil_deteksi.csv")
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow([
            'datetime',
            'image_name',
            'yolo_label',
            'yolo_confidence',
            'crop_img',
            'preproc_img',
            'ocr_text',
            'ocr_confidence'
        ])

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img_cv[y1:y2, x1:x2]

        # === PREPROCESSING ===
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        plate_blur = cv2.GaussianBlur(plate_gray, (3, 3), 0)
        plate_thresh = cv2.adaptiveThreshold(
            plate_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 15
        )
        h = plate_thresh.shape[0]
        plate_thresh = plate_thresh[:int(h * 0.75), :]
        plate_thresh = cv2.resize(plate_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # === SIMPAN GAMBAR ===
        crop_filename = f"crop_{i+1}_{os.path.basename(image_path)}"
        preproc_filename = f"preproc_{i+1}_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(output_dir, crop_filename), plate_crop)
        cv2.imwrite(os.path.join(output_dir, preproc_filename), plate_thresh)

        # === TAMPILKAN CROP & PREPROC ===
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        plt.title(f'Crop Plat #{i+1}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(plate_thresh, cmap='gray')
        plt.title('Preprocessed')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # === OCR EASYOCR ===
        texts = []
        confs = []
        ocr_result = reader.readtext(plate_thresh, detail=1)
        for result in ocr_result:
            if len(result) == 3:
                _, text, conf = result
                if conf > 0.5:
                    texts.append(text)
                    confs.append(conf)

        merged_text = ' '.join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0

        # === LABEL & CONFIDENCE YOLO ===
        try:
            class_id = int(results[0].boxes.cls[i])
            label = results[0].names[class_id]
        except:
            label = "Unknown"

        try:
            conf_yolo = float(results[0].boxes.conf[i])
        except:
            conf_yolo = 0.0

        # === SIMPAN KE CSV ===
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            os.path.basename(image_path),
            label,
            f"{conf_yolo:.2f}",
            crop_filename,
            preproc_filename,
            merged_text,
            f"{avg_conf:.2f}"
        ])

        # === CETAK KE TERMINAL ===
        print(f"Teks Plat Terdeteksi: {merged_text} (Confidence: {avg_conf:.2f})\n")

