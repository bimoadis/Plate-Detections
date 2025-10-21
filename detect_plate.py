from ultralytics import YOLO
import cv2, os, numpy as np, csv
from datetime import datetime

# =========================================
# 🔹 Load Model YOLO
# =========================================
model_plate = YOLO("runs11s/detect/train/weights/best.pt")   # deteksi plat nomor
model_ocr = YOLO("OCRCUSTOM2/content/runs/detect/train/weights/best.pt")  # deteksi karakter OCR

# =========================================
# 🔹 Fungsi peningkatan kualitas (HD)
# =========================================
def enhance_hd(img):
    """Meningkatkan kualitas gambar agar lebih jelas untuk OCR."""
    # Resize ke ukuran lebih besar
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # Sharpening (penajaman)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Denoise ringan
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img

# =========================================
# 🔹 Fungsi baca OCR YOLO
# =========================================
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

    # Urutkan karakter dari kiri ke kanan
    detections = sorted(detections, key=lambda x: x[0])
    text = "".join([ch for _, ch in detections])
    return text.strip()

# =========================================
# 🔹 Fungsi utama
# =========================================
def process_video(video_path: str, output_dir: str = "hasil_deteksi_video") -> str:
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, "crop")
    hd_dir = os.path.join(output_dir, "hd")
    deteksi_dir = os.path.join(output_dir, "deteksi")

    for d in [crop_dir, hd_dir, deteksi_dir]:
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(output_dir, "hasil_video.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Waktu", "Frame", "Label Plat", "Conf YOLO",
                "OCR Crop", "OCR HD", "Crop Path", "HD Path"
            ])

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_interval = 25  # proses tiap 25 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # =========================================
        # 1️⃣ DETEKSI PLAT NOMOR
        # =========================================
        results_plate = model_plate.predict(frame, conf=0.5, device="cpu")[0]
        if not results_plate.boxes:
            continue

        for i, box in enumerate(results_plate.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_plate.names[cls]

            # =========================================
            # 2️⃣ Simpan crop
            # =========================================
            crop = frame[y1:y2, x1:x2]
            crop_path = os.path.join(crop_dir, f"crop_{frame_count}_{i+1}.png")
            cv2.imwrite(crop_path, crop)

            # =========================================
            # 3️⃣ Buat versi HD dan simpan
            # =========================================
            hd_crop = enhance_hd(crop)
            hd_path = os.path.join(hd_dir, f"hd_{frame_count}_{i+1}.png")
            cv2.imwrite(hd_path, hd_crop)

            # =========================================
            # 4️⃣ Jalankan OCR pada crop & HD crop
            # =========================================
            plate_text_crop = run_custom_ocr(crop)
            plate_text_hd = run_custom_ocr(hd_crop)

            # =========================================
            # 5️⃣ Simpan hasil ke CSV
            # =========================================
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_count, label, f"{conf_yolo:.2f}",
                    plate_text_crop, plate_text_hd,
                    crop_path, hd_path
                ])

            # =========================================
            # 6️⃣ Simpan frame hasil deteksi
            # =========================================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_disp = plate_text_hd if plate_text_hd else plate_text_crop
            cv2.putText(frame, text_disp, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            deteksi_path = os.path.join(deteksi_dir, f"deteksi_{frame_count}.jpg")
            cv2.imwrite(deteksi_path, frame)

            print(f"[Frame {frame_count}] Plat: {text_disp} | Conf: {conf_yolo:.2f}")

    cap.release()
    print("\n✅ Selesai! Hasil disimpan di:", csv_path)
    return csv_path
