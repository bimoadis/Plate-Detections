from ultralytics import YOLO
from tkinter import filedialog, Tk
import cv2
import os
import numpy as np
import easyocr
import csv
from datetime import datetime

# === PILIH VIDEO DARI FILE EXPLORER ===
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Pilih Video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("❌ Tidak ada video dipilih.")
    exit()

# === FOLDER OUTPUT ===
output_dir = "hasil_deteksi_video"
crop_dir = os.path.join(output_dir, "crop")
hd_dir = os.path.join(output_dir, "hd")
thresh_dir = os.path.join(output_dir, "threshold")
os.makedirs(crop_dir, exist_ok=True)
os.makedirs(hd_dir, exist_ok=True)
os.makedirs(thresh_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "hasil_video.csv")

# === INISIALISASI ===
model = YOLO("runs2/detect/train/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# === CSV HEADER ===
header = ["Waktu", "Frame", "Label YOLO", "Conf YOLO",
          "Crop Path", "HD Path", "Threshold Path",
          "OCR HD", "Conf HD", "OCR Threshold", "Conf Threshold"]

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# === BUKA VIDEO ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
frame_interval = 10  # proses setiap 10 frame untuk efisiensi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    results = model.predict(frame, conf=0.3, device="cpu")[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_yolo = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        crop = frame[y1:y2, x1:x2]
        crop_name = f"crop_{frame_count}_{i+1}.png"
        crop_path = os.path.join(crop_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # Auto Rotate
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rotated = crop.copy()
        if cnts:
            all_cnt = np.vstack(cnts)
            rect = cv2.minAreaRect(all_cnt)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            h, w = crop.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # HD Enhancement
        upscale = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(upscale, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
        hd = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        hd = cv2.filter2D(hd, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

        hd_name = f"hd_{frame_count}_{i+1}.png"
        hd_path = os.path.join(hd_dir, hd_name)
        cv2.imwrite(hd_path, hd)

        # Threshold
        gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_hd, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 41, 15)
        h = thresh.shape[0]
        thresh = thresh[:int(h * 0.75), :]

        thresh_name = f"thresh_{frame_count}_{i+1}.png"
        thresh_path = os.path.join(thresh_dir, thresh_name)
        cv2.imwrite(thresh_path, thresh)

        # OCR HD
        result_hd = reader.readtext(hd)
        texts_hd = [t[1] for t in result_hd if t[2] > 0.5]
        confs_hd = [t[2] for t in result_hd if t[2] > 0.5]
        text_hd = ' '.join(texts_hd)
        avg_hd = sum(confs_hd) / len(confs_hd) if confs_hd else 0

        # OCR Threshold
        result_thresh = reader.readtext(thresh)
        texts_t = [t[1] for t in result_thresh if t[2] > 0.5]
        confs_t = [t[2] for t in result_thresh if t[2] > 0.5]
        text_t = ' '.join(texts_t)
        avg_t = sum(confs_t) / len(confs_t) if confs_t else 0

        # Log
        print(f"[Frame {frame_count}] Label: {label} ({conf_yolo:.2f}) | OCR-HD: {text_hd} ({avg_hd:.2f}) | OCR-TH: {text_t} ({avg_t:.2f})")

        # Save to CSV
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_count, label, f"{conf_yolo:.2f}",
                crop_path, hd_path, thresh_path,
                text_hd, f"{avg_hd:.2f}", text_t, f"{avg_t:.2f}"
            ])

cap.release()
