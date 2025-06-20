from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
import cv2
import easyocr

# 1. Pilih gambar dari file explorer
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Pilih Gambar",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.png")]
)

# 2. Validasi
if not image_path:
    print("❌ Tidak ada gambar dipilih.")
    exit()

# 3. Load model YOLOv8
model = YOLO("runs2/detect/train/weights/best.pt")

# 4. Deteksi plat nomor
results = model.predict(source=image_path, conf=0.25, save=True, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu")

# 5. Tampilkan hasil deteksi (gambar lengkap)
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

# 6. OCR (EasyOCR) dengan preprocessing
reader = easyocr.Reader(['en'], gpu=cv2.cuda.getCudaEnabledDeviceCount() > 0)
img_cv = cv2.imread(image_path)

print("\n📌 Hasil Pembacaan Teks Plat (Setelah Preprocessing):\n")

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_crop = img_cv[y1:y2, x1:x2]

    # === Preprocessing yang Lebih Baik ===
    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_eq = cv2.equalizeHist(plate_gray)
    _, plate_thresh = cv2.threshold(plate_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Opsional: perbesar gambar agar OCR lebih akurat
    plate_thresh = cv2.resize(plate_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Tampilkan hasil crop + preprocessing
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

    # OCR
    ocr_result = reader.readtext(plate_thresh)

    texts = []
    confs = []
    for (_, text, conf) in ocr_result:
        if conf > 0.5:
            texts.append(text)
            confs.append(conf)

    merged_text = ' '.join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0

    print(f"Teks Plat Terdeteksi: {merged_text} (Confidence: {avg_conf:.2f})")
