from ultralytics import YOLO
import cv2, os, csv
from datetime import datetime

# =========================================
# 🔹 Load Model YOLO untuk deteksi plat
# =========================================
model_plate = YOLO("runs2/detect/train/weights/best.pt")  # ganti dengan path modelmu

# =========================================
# 🔹 Fungsi utama: deteksi & crop dari folder
# =========================================
def detect_and_crop_folder(input_folder: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, "crops")
    deteksi_dir = os.path.join(output_dir, "deteksi")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(deteksi_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "hasil_crop.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Waktu", "Nama File", "Label", "Confidence", "Crop Path"])

    # =========================================
    # 🔹 Loop semua gambar dalam folder
    # =========================================
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Gagal membaca: {filename}")
            continue

        # =========================================
        # 🔹 Deteksi plat nomor
        # =========================================
        results = model_plate.predict(source=image, conf=0.5, device="cpu")[0]

        if not results.boxes:
            print(f"🚫 Tidak ada plat pada {filename}")
            continue

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_plate.names[cls]

            # Crop plat
            crop = image[y1:y2, x1:x2]
            crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i+1}.png"
            crop_path = os.path.join(crop_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            # Simpan hasil deteksi ke CSV
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    filename, label, f"{conf_yolo:.2f}", crop_path
                ])

            # Gambar bounding box di gambar asli
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf_yolo:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Simpan gambar hasil deteksi
        deteksi_path = os.path.join(deteksi_dir, f"deteksi_{filename}")
        cv2.imwrite(deteksi_path, image)

        print(f"✅ {filename} → {len(results.boxes)} plat terdeteksi")

    print("\n🎯 Selesai! Semua hasil disimpan di folder:", output_dir)
    return csv_path


# =========================================
# 🔹 Jalankan fungsi
# =========================================
if __name__ == "__main__":
    input_folder = "Plat-putih"   # ganti dengan folder gambar kamu
    detect_and_crop_folder(input_folder)
