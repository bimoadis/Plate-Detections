import cv2
import os
from glob import glob
import numpy as np
import torch

# Install: pip install realesrgan basicsr

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install dengan: pip install realesrgan basicsr")
    exit(1)

# ==============================================================================
# KONFIGURASI
# ==============================================================================

model_path = "weights/net_g_5000.pth"  # Path ke model hasil training
lr_folder = "dataset_lr"               # Folder input LR
sr_folder = "dataset_sr"               # Folder output SR
scale = 4                              # Scale factor (4x)

# ==============================================================================
# LOAD MODEL
# ==============================================================================

print("=" * 50)
print("LR → SR dengan Real-ESRGAN")
print("=" * 50)

# Deteksi device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {device}")

print(f"\n📥 Loading model dari: {model_path}")

# Buat model RRDBNet (arsitektur RealESRGAN_x4plus)
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=scale
)

# Buat upsampler dengan RealESRGANer
upsampler = RealESRGANer(
    scale=scale,
    model_path=model_path,
    model=model,
    tile=0,           # 0 = tidak pakai tile, atau set 256/512 untuk gambar besar
    tile_pad=10,
    pre_pad=0,
    half=False,       # True jika GPU support FP16
    device=device
)

print("✅ Model berhasil dimuat!")

# ==============================================================================
# PROSES FOLDER LR → SR
# ==============================================================================

os.makedirs(sr_folder, exist_ok=True)

# Ambil semua gambar LR
lr_list = glob(os.path.join(lr_folder, "*.png")) + \
          glob(os.path.join(lr_folder, "*.jpg")) + \
          glob(os.path.join(lr_folder, "*.jpeg"))

print(f"\n📁 Ditemukan {len(lr_list)} gambar LR di {lr_folder}")
print("-" * 50)

# Proses setiap gambar
for i, img_path in enumerate(sorted(lr_list), 1):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    out_path = os.path.join(sr_folder, f"{base_name}.png")
    
    print(f"[{i}/{len(lr_list)}] Processing: {filename}...", end=" ")
    
    try:
        # Baca gambar LR (BGR format untuk OpenCV)
        lr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if lr_image is None:
            print(f"❌ Gagal membaca gambar")
            continue
        
        # Super Resolution dengan Real-ESRGAN
        sr_image, _ = upsampler.enhance(lr_image, outscale=scale)
        
        # Simpan hasil
        cv2.imwrite(out_path, sr_image)
        print(f"✅ Saved: {out_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print(f"🎉 Selesai! Hasil SR disimpan di: {sr_folder}")
