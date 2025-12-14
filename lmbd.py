from PIL import Image
import os
from glob import glob
import math

src_dir = 'dataset_hr'           # folder berisi gambar asli (sumber)
lr_output_dir = 'dataset_hr64'   # folder output LR (min 64x64, kelipatan 4)
hr_output_dir = 'dataset_hr_4x'  # folder output HR (4x dari LR)

MIN_SIZE = 64  # ukuran minimum untuk LR
SCALE_FACTOR = 4  # HR = SCALE_FACTOR x LR

# Buat folder output jika belum ada
os.makedirs(lr_output_dir, exist_ok=True)
os.makedirs(hr_output_dir, exist_ok=True)

def round_to_multiple(value, multiple):
    """Bulatkan ke atas ke kelipatan tertentu"""
    return math.ceil(value / multiple) * multiple

def get_new_dimensions(w, h, min_size, scale_factor):
    """
    Hitung dimensi baru yang:
    1. Minimal = min_size
    2. Merupakan kelipatan scale_factor (agar HR tepat 4x)
    """
    # Pastikan minimal min_size
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    
    # Bulatkan ke kelipatan scale_factor
    new_w = round_to_multiple(new_w, scale_factor)
    new_h = round_to_multiple(new_h, scale_factor)
    
    return new_w, new_h

# Cari semua file gambar (png, jpg, jpeg)
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(f'{src_dir}/{ext}'))

print(f"📁 Ditemukan {len(image_paths)} gambar di folder '{src_dir}'")
print(f"📐 Ukuran minimum LR: {MIN_SIZE}x{MIN_SIZE}")
print(f"📐 Dimensi LR akan dibulatkan ke kelipatan {SCALE_FACTOR}")
print(f"🔍 HR akan tepat {SCALE_FACTOR}x lebih besar dari LR")
print("-" * 60)

for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    filename = os.path.basename(img_path)
    
    # Hitung dimensi LR baru (min 64, kelipatan 4)
    new_lr_w, new_lr_h = get_new_dimensions(w, h, MIN_SIZE, SCALE_FACTOR)
    
    # Resize ke ukuran LR baru
    lr_img = img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
    
    # Buat HR tepat 4x lebih besar dari LR
    hr_w = new_lr_w * SCALE_FACTOR
    hr_h = new_lr_h * SCALE_FACTOR
    hr_img = lr_img.resize((hr_w, hr_h), Image.BICUBIC)
    
    # Verifikasi rasio (harus tepat 4x)
    assert hr_w == new_lr_w * SCALE_FACTOR, f"Width mismatch: {hr_w} != {new_lr_w * SCALE_FACTOR}"
    assert hr_h == new_lr_h * SCALE_FACTOR, f"Height mismatch: {hr_h} != {new_lr_h * SCALE_FACTOR}"
    
    # Simpan LR
    lr_out_path = os.path.join(lr_output_dir, filename)
    lr_img.save(lr_out_path)
    
    # Simpan HR
    hr_out_path = os.path.join(hr_output_dir, filename)
    hr_img.save(hr_out_path)
    
    print(f"✓ {filename}: {w}x{h} → LR: {new_lr_w}x{new_lr_h} → HR: {hr_w}x{hr_h}")

print("-" * 60)
print(f"✅ Selesai!")
print(f"   📂 LR tersimpan di: '{lr_output_dir}' (min {MIN_SIZE}x{MIN_SIZE}, kelipatan {SCALE_FACTOR})")
print(f"   📂 HR tersimpan di: '{hr_output_dir}' (tepat {SCALE_FACTOR}x dari LR)")
