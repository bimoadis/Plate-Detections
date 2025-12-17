from PIL import Image
import os
from glob import glob

lr_dir = 'dataset_lr'
hr_dir = 'dataset_hr_4x'

print("=" * 60)
print("Fix Dimensi: Resize HR menjadi 4x dari LR")
print("=" * 60)

lr_files = glob(f"{lr_dir}/*.png") + glob(f"{lr_dir}/*.jpg") + glob(f"{lr_dir}/*.jpeg")

total = 0
already_ok = 0
resized = 0
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
    
    # Target size = 4x LR
    target_w = lr_w * 4
    target_h = lr_h * 4
    
    # Cek apakah sudah 4x
    if hr_w == target_w and hr_h == target_h:
        print(f"✅ OK: {filename} - sudah 4x ({hr_w}x{hr_h})")
        already_ok += 1
    else:
        # Resize HR ke 4x ukuran LR
        hr_resized = hr_img.resize((target_w, target_h), Image.LANCZOS)
        hr_resized.save(hr_path)
        print(f"🔧 RESIZED: {filename} - ({hr_w}x{hr_h}) → ({target_w}x{target_h})")
        resized += 1

print("\n" + "=" * 60)
print("HASIL:")
print(f"  Total file LR   : {total}")
print(f"  ✅ Sudah OK      : {already_ok}")
print(f"  🔧 Di-resize     : {resized}")
print(f"  ❌ Missing HR    : {missing}")
print("=" * 60)

if resized > 0:
    print(f"🎉 {resized} file HR berhasil di-resize menjadi 4x LR!")
else:
    print("✅ Semua file sudah benar, tidak ada yang perlu di-resize.")

