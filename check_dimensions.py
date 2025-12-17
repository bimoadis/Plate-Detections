from PIL import Image
import os
from glob import glob

lr_dir = 'dataset_lr'
hr_dir = 'dataset_hr_4x'

print("=" * 60)
print("Cek Dimensi: HR harus 4x lebih besar dari LR")
print("=" * 60)

lr_files = glob(f"{lr_dir}/*.png") + glob(f"{lr_dir}/*.jpg") + glob(f"{lr_dir}/*.jpeg")

total = 0
valid = 0
invalid = 0
missing = 0

for lr_path in sorted(lr_files):
    filename = os.path.basename(lr_path)
    base_name = os.path.splitext(filename)[0]
    
    # Cari file HR dengan nama yang sama (cek berbagai ekstensi)
    hr_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(hr_dir, base_name + ext)
        if os.path.exists(candidate):
            hr_path = candidate
            break
    
    total += 1
    
    if hr_path is None:
        print(f"❌ MISSING: {filename} tidak ditemukan di {hr_dir}")
        missing += 1
        continue
    
    # Buka kedua gambar
    lr_img = Image.open(lr_path)
    hr_img = Image.open(hr_path)
    
    lr_w, lr_h = lr_img.size
    hr_w, hr_h = hr_img.size
    
    # Hitung rasio
    ratio_w = hr_w / lr_w if lr_w > 0 else 0
    ratio_h = hr_h / lr_h if lr_h > 0 else 0
    
    # Cek apakah 4x
    is_4x = (ratio_w == 4.0 and ratio_h == 4.0)
    
    if is_4x:
        print(f"✅ OK: {filename} - LR({lr_w}x{lr_h}) → HR({hr_w}x{hr_h}) = 4x")
        valid += 1
    else:
        print(f"❌ WRONG: {filename} - LR({lr_w}x{lr_h}) → HR({hr_w}x{hr_h}) = {ratio_w:.2f}x, {ratio_h:.2f}x")
        invalid += 1

print("\n" + "=" * 60)
print("HASIL:")
print(f"  Total file LR  : {total}")
print(f"  ✅ Valid (4x)   : {valid}")
print(f"  ❌ Invalid      : {invalid}")
print(f"  ❌ Missing HR   : {missing}")
print("=" * 60)

if invalid == 0 and missing == 0:
    print("🎉 Semua file sudah benar! HR = 4x LR")
else:
    print("⚠️  Ada file yang tidak sesuai, periksa output di atas.")

