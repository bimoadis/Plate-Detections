# ============================================================
# 🚗 TSRN Inference + OCR Custom (Local PC Version — 5 Foto Random)
# ============================================================
import torch
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms
from tkinter import Tk, filedialog
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 🧱 Residual Block + TSRN
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class TSRN(nn.Module):
    def __init__(self, num_residuals=5):
        super().__init__()
        self.entry = nn.Conv2d(3, 64, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])
        self.exit = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.entry(x))
        x = self.res_blocks(x)
        x = self.exit(x)
        return torch.sigmoid(x)

# ============================================================
# ⚙️ Setup Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Menggunakan device: {device}")

# ============================================================
# 🧠 Load Semua Model TSRN
# ============================================================
model_dir = "models_tsrn"
model_paths = {
    "tsrn1": os.path.join(model_dir, "tsrn_plate_final1.pth"),
    "tsrn2": os.path.join(model_dir, "tsrn_plate_final2.pth"),
    "tsrn3": os.path.join(model_dir, "tsrn_plate_final3.pth"),
    "tsrn4": os.path.join(model_dir, "tsrn_plate_final4.pth"),
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        model = TSRN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[name] = model
        print(f"✅ Model {name} berhasil dimuat dari {path}")
    else:
        print(f"⚠️ Model {name} tidak ditemukan!")

if not models:
    raise FileNotFoundError("❌ Tidak ada model TSRN ditemukan!")

# ============================================================
# 🔤 Load Custom OCR YOLO Model
# ============================================================
ocr_model = YOLO("OCRCUSTOMV4/content/runs/detect/train/weights/best.pt")
print("✅ Model OCR berhasil dimuat.")

def run_custom_ocr(image):
    """OCR karakter menggunakan YOLO custom."""
    results = ocr_model.predict(source=image, conf=0.25, device="cpu")[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    cls = results.boxes.cls.cpu().numpy()
    names = results.names

    detections = []
    for box2, c in zip(boxes, cls):
        x1, _, _, _ = box2
        detections.append((x1, names[int(c)]))

    detections = sorted(detections, key=lambda x: x[0])
    return "".join([ch for _, ch in detections]).strip()

# ============================================================
# 📸 Pilih Folder Gambar Input
# ============================================================
Tk().withdraw()
input_folder = filedialog.askdirectory(title="Pilih Folder Gambar LR")
if not input_folder:
    raise ValueError("❌ Tidak ada folder dipilih.")

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
selected_files = random.sample(image_files, min(5, len(image_files)))

print("\n📂 Gambar yang diproses:", selected_files)

transform = transforms.Compose([transforms.ToTensor()])

# ============================================================
# 🔄 Proses Setiap Gambar
# ============================================================
for filename in selected_files:
    img_path = os.path.join(input_folder, filename)
    image = Image.open(img_path).convert("RGB")
    lr_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    tensor = transform(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    sr_results = {}
    ocr_results = {}

    # TSRN Inference
    for name, model in models.items():
        with torch.no_grad():
            sr_tensor = model(tensor)

        sr_img = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_img = (sr_img * 255).clip(0, 255).astype("uint8")

        sr_results[name] = sr_img

        # Simpan hasil
        out_dir = f"sr_output_{name}"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

        # Jalankan OCR pada SR model ini
        ocr_results[name] = run_custom_ocr(sr_img)

    # ============================================================
    # 🎨 Tampilkan LR + Semua SR + OCR
    # ============================================================
    plt.figure(figsize=(18, 4))

    plt.subplot(1, len(models) + 1, 1)
    plt.imshow(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
    plt.title("LR (Input)")
    plt.axis("off")

    idx = 2
    for name in models.keys():
        plt.subplot(1, len(models) + 1, idx)
        plt.imshow(sr_results[name])
        plt.title(f"{name.upper()}\nOCR: {ocr_results[name]}", fontsize=9)
        plt.axis("off")
        idx += 1

    plt.suptitle(f"Perbandingan TSRN + OCR — {filename}", fontsize=14)
    plt.show()

print("\n🎉 Semua proses selesai — TSRN + OCR sukses!")
